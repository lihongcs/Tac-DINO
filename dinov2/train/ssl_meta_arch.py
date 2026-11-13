# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified for Tac-DINO.


import math
import torch
import logging
from torch import nn
from functools import partial
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.models import build_model_from_cfg
from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import (
    ShardedGradScaler,
    get_fsdp_modules,
    get_fsdp_wrapper,
    is_fsdp,
    reshard_fsdp_model,
)

from dinov2.models.vision_transformer import BlockChunk

try:
    from xformers.ops import fmha
except ImportError:
    fmha = None


logger = logging.getLogger("dinov2")


def convert_chunked_state_dict(model, state_dict):
    """
    将标准 ViT 权重的 key 转换为 Chunked ViT 的格式。
    标准格式: blocks.0.attn...
    Chunked 格式: blocks.0.0.attn... (其中 blocks.{chunk_idx}.{block_idx})
    """
    # 如果模型没有使用 chunked_blocks，直接返回原 state_dict
    if not getattr(model, "chunked_blocks", False):
        return state_dict

    # 1. 建立 block_idx 到 chunk_idx 的映射
    # 通过遍历模型参数，找到实际存在的 block 参数位置
    # nn.Identity (Padding) 没有参数，所以只会找到真实的 block
    block_map = {}
    for name, _ in model.named_parameters():
        parts = name.split(".")
        # 期望格式: blocks.{chunk_idx}.{block_idx}.param
        if len(parts) >= 3 and parts[0] == "blocks":
            try:
                chunk_idx = int(parts[1])
                block_idx = int(parts[2])
                block_map[block_idx] = chunk_idx
            except ValueError:
                continue

    if not block_map:
        return state_dict

    logger.info(f"Converting state_dict for chunked blocks. Mapped {len(block_map)} blocks.")
    
    new_state_dict = {}
    for k, v in state_dict.items():
        parts = k.split(".")
        # 检查是否为 blocks.N.params 格式
        if len(parts) >= 2 and parts[0] == "blocks":
            try:
                block_idx = int(parts[1])
                # 如果能在模型中找到对应的 block
                if block_idx in block_map:
                    chunk_idx = block_map[block_idx]
                    # 构造新 key: blocks.{chunk_idx}.{block_idx}.{rest}
                    # parts[1:] 包含 block_idx 和后续部分，例如 ['0', 'norm1', 'weight']
                    new_key = f"blocks.{chunk_idx}.{'.'.join(parts[1:])}"
                    new_state_dict[new_key] = v
                    continue
            except ValueError:
                pass
        
        # 其他 key (如 cls_token, pos_embed 等) 保持不变
        new_state_dict[k] = v
            
    return new_state_dict





class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.local_data = cfg.train.local_data
        if cfg.compute_precision.grad_scaler:
            grad_scaler_init_scale = float(
                cfg.compute_precision.grad_scaler_init_scale
            )
            if not math.isfinite(grad_scaler_init_scale) or grad_scaler_init_scale <= 0:
                raise ValueError(
                    "compute_precision.grad_scaler_init_scale must be finite and positive"
                )
            self.fp16_scaler = ShardedGradScaler(
                init_scale=grad_scaler_init_scale
            )
        else:
            self.fp16_scaler = None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            if "model" in chkpt:
                raw_state_dict = chkpt["model"]
            else:
                raw_state_dict = chkpt
            state_dict = convert_chunked_state_dict(student_backbone, raw_state_dict)
            student_backbone.load_state_dict(state_dict, strict=True)
        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        # The original DINOv2 PyTorch-2.0 FSDP implementation lazily creates a
        # private ``_streams`` mapping for every independent root and then
        # shares it after the first backward. Newer FSDP releases removed that
        # private mapping and synchronize each independent root's own streams.
        # Keep a one-shot, feature-detected compatibility check for both APIs.
        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(
        self,
        images,
        teacher_temp,
        backward_loss_scale=1.0,
        return_objective=False,
    ):
        n_global_crops = 2
        n_local_crops = self.cfg.crops.local_crops_number
        device = next(self.student.backbone.parameters()).device
        global_crops = images["collated_global_crops"].to(device, non_blocking=True)
        local_tactile_crops = images["collated_local_tactile_crops"].to(device, non_blocking=True)
        masks = images["collated_masks"].to(device, non_blocking=True)
        mask_indices_list = images["mask_indices_list"].to(device, non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].to(device, non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].to(device, non_blocking=True)

        do_dino = self.do_dino
        do_ibot = self.do_ibot
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            teacher_dino_softmaxed_centered_list = None
            masked_teacher_ibot_softmaxed_centered = None
            if do_dino or do_ibot:
                x, n_global_crops_teacher = global_crops, n_global_crops
                teacher_backbone_output_dict = self.teacher.backbone(
                    x, is_training=True
                )
                teacher_cls_tokens = teacher_backbone_output_dict[
                    "x_norm_clstoken"
                ]
                teacher_cls_tokens = teacher_cls_tokens.chunk(
                    n_global_crops_teacher
                )
                # Reverse A/B so each student global crop matches the other
                # teacher view in the standard DINO objective.
                teacher_cls_tokens = torch.cat(
                    (teacher_cls_tokens[1], teacher_cls_tokens[0])
                )
                ibot_teacher_patch_tokens = teacher_backbone_output_dict[
                    "x_norm_patchtokens"
                ]
                _dim = ibot_teacher_patch_tokens.shape[-1]
                n_cls_tokens = teacher_cls_tokens.shape[0]

                if do_ibot and not self.ibot_separate_head:
                    buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                        upperbound + n_cls_tokens, _dim
                    )
                    buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                    torch.index_select(
                        ibot_teacher_patch_tokens.flatten(0, 1),
                        dim=0,
                        index=mask_indices_list,
                        out=buffer_tensor_teacher[
                            n_cls_tokens : n_cls_tokens + n_masked_patches
                        ],
                    )
                    tokens_after_head = self.teacher.dino_head(
                        buffer_tensor_teacher
                    )
                    teacher_cls_tokens_after_head = tokens_after_head[
                        :n_cls_tokens
                    ]
                    masked_teacher_patch_tokens_after_head = tokens_after_head[
                        n_cls_tokens : n_cls_tokens + n_masked_patches
                    ]
                elif do_ibot and self.ibot_separate_head:
                    buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(
                        upperbound, _dim
                    )
                    torch.index_select(
                        ibot_teacher_patch_tokens.flatten(0, 1),
                        dim=0,
                        index=mask_indices_list,
                        out=buffer_tensor_teacher[:n_masked_patches],
                    )
                    teacher_cls_tokens_after_head = self.teacher.dino_head(
                        teacher_cls_tokens
                    )
                    masked_teacher_patch_tokens_after_head = (
                        self.teacher.ibot_head(buffer_tensor_teacher)[
                            :n_masked_patches
                        ]
                    )
                else:
                    teacher_cls_tokens_after_head = self.teacher.dino_head(
                        teacher_cls_tokens
                    )

                if self.cfg.train.centering == "centering":
                    teacher_dino_softmaxed_centered_list = (
                        self.dino_loss.softmax_center_teacher(
                            teacher_cls_tokens_after_head,
                            teacher_temp=teacher_temp,
                        ).view(
                            n_global_crops_teacher,
                            -1,
                            *teacher_cls_tokens_after_head.shape[1:],
                        )
                    )
                    self.dino_loss.update_center(teacher_cls_tokens_after_head)
                    if do_ibot:
                        masked_teacher_patch_tokens_after_head = (
                            masked_teacher_patch_tokens_after_head.unsqueeze(0)
                        )
                        masked_teacher_ibot_softmaxed_centered = (
                            self.ibot_patch_loss.softmax_center_teacher(
                                masked_teacher_patch_tokens_after_head[
                                    :, :n_masked_patches
                                ],
                                teacher_temp=teacher_temp,
                            ).squeeze(0)
                        )
                        self.ibot_patch_loss.update_center(
                            masked_teacher_patch_tokens_after_head[
                                :n_masked_patches
                            ]
                        )
                elif self.cfg.train.centering == "sinkhorn_knopp":
                    teacher_dino_softmaxed_centered_list = (
                        self.dino_loss.sinkhorn_knopp_teacher(
                            teacher_cls_tokens_after_head,
                            teacher_temp=teacher_temp,
                        ).view(
                            n_global_crops_teacher,
                            -1,
                            *teacher_cls_tokens_after_head.shape[1:],
                        )
                    )
                    if do_ibot:
                        masked_teacher_ibot_softmaxed_centered = (
                            self.ibot_patch_loss.sinkhorn_knopp_teacher(
                                masked_teacher_patch_tokens_after_head,
                                teacher_temp=teacher_temp,
                                n_masked_patches_tensor=n_masked_patches_tensor,
                            )
                        )
                else:
                    raise NotImplementedError
            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)
        loss_dict = {}
        loss_accumulator = 0

        # Global RGB and local tactile views use independent backbones.
        student_global_backbone_output_dict = self.student.backbone(
            global_crops, masks=masks, is_training=True)
        student_local_backbone_output_dict = self.student.tactile_backbone(
            local_tactile_crops, is_training=True)
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list = [student_local_cls_tokens.unsqueeze(0),
                                        student_global_cls_tokens.unsqueeze(0)]

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(
            inputs_for_student_head_list
        )
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if do_dino and n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss
        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        alignment_loss, alignment_logs = self.multiresource_objective(images)
        loss_accumulator = loss_accumulator + alignment_loss
        loss_dict.update(alignment_logs)

        if torch.isnan(loss_accumulator) or torch.isinf(loss_accumulator):
             logger.error(f"Loss accumulator is NaN/Inf: {loss_accumulator}")
             logger.error(f"Loss dict: {loss_dict}")
             # Print component losses
             if "dino_local_crops_loss" in loss_dict:
                 logger.error(f"dino_local_crops_loss: {loss_dict['dino_local_crops_loss']}")
             if "dino_global_crops_loss" in loss_dict:
                 logger.error(f"dino_global_crops_loss: {loss_dict['dino_global_crops_loss']}")
             if "ibot_loss" in loss_dict:
                 logger.error(f"ibot_loss: {loss_dict['ibot_loss']}")
             if "koleo_loss" in loss_dict:
                 logger.error(f"koleo_loss: {loss_dict['koleo_loss']}")
             raise ValueError("NaN/Inf loss accumulator")
        
        
        backward_loss_scale = float(backward_loss_scale)
        if not math.isfinite(backward_loss_scale) or backward_loss_scale <= 0:
            raise ValueError("backward_loss_scale must be finite and positive")
        # Keep the exact historical graph when accumulation is disabled. For
        # N microsteps, each backward contributes objective/N so the final
        # synchronized gradient is the mean over the effective global batch.
        backward_objective = (
            loss_accumulator
            if backward_loss_scale == 1.0
            else loss_accumulator * backward_loss_scale
        )
        self.backprop_loss(backward_objective)

        self.fsdp_synchronize_streams()

        if return_objective:
            return loss_dict, loss_accumulator.detach()
        return loss_dict

    def fsdp_synchronize_streams(self):
        if not self.need_to_synchronize_fsdp_streams:
            return

        reference = self.student.backbone
        fsdp_roots = {
            **{
                f"student.{name}": module
                for name, module in self.student.items()
            },
            **{
                f"teacher.{name}": module
                for name, module in self.teacher.items()
            },
        }
        trainable_root_names = set(self.get_trainable_fsdp_roots())

        # Legacy PyTorch/DINOv2 contract. The official DINOv2 implementation
        # performs this one-time assignment after lazy FSDP initialization.
        if hasattr(reference, "_streams"):
            if getattr(reference, "_is_root", None) is None:
                raise RuntimeError(
                    "legacy FSDP backbone stream state was not lazily initialized"
                )
            shared_streams = reference._streams
            initialized_roots = {
                name
                for name, module in fsdp_roots.items()
                if hasattr(module, "_streams")
                and getattr(module, "_is_root", None) is not None
            }
            missing_trainable = sorted(
                trainable_root_names - initialized_roots
            )
            if missing_trainable:
                raise RuntimeError(
                    "trainable legacy FSDP roots were not initialized before "
                    "stream sharing: " + ", ".join(missing_trainable)
                )
            torch.cuda.synchronize()
            for name in initialized_roots:
                fsdp_roots[name]._streams = shared_streams
            logger.info(
                "Legacy FSDP streams shared across independent roots: %s",
                ", ".join(sorted(initialized_roots)),
            )
        else:
            # Modern FSDP1 removed ``_streams`` and maintains explicit stream
            # attributes per root. Its root post-backward callback waits on
            # that root's post-backward/all-reduce streams, so overwriting the
            # private state across independent roots is neither required nor
            # supported. Still require every trainable root to have completed
            # lazy initialization on the first backward.
            modern_stream_attributes = (
                "_default_stream",
                "_unshard_stream",
                "_post_backward_stream",
                "_pre_unshard_stream",
                "_all_reduce_stream",
            )
            initialized_roots = {
                name
                for name, module in fsdp_roots.items()
                if getattr(module, "_is_root", None) is not None
                and all(
                    hasattr(module, attribute)
                    and getattr(module, attribute) is not None
                    for attribute in modern_stream_attributes
                )
            }
            missing_trainable = sorted(
                trainable_root_names - initialized_roots
            )
            if missing_trainable:
                raise RuntimeError(
                    "unknown or uninitialized FSDP stream API for trainable "
                    "roots: " + ", ".join(missing_trainable)
                )
            logger.info(
                "Modern FSDP per-root stream lifecycle detected; private "
                "stream reassignment is disabled"
            )
        self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            # Reshard cached teacher parameters before updating the EMA.
            reshard_fsdp_model(self.teacher)
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()
        return self

    def synchronize_auxiliary_buffers(self):
        return 0

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        return [group for module in self.student.values()
                for group in self.get_maybe_fused_params_for_submodel(module)]

    def get_trainable_fsdp_roots(self):
        """Return every disjoint FSDP root that owns trainable parameters."""
        candidates = [(f"student.{name}", module) for name, module in self.student.items()]
        roots = {}
        covered_parameters = {}
        for root_name, module in candidates:
            trainable_parameters = tuple(
                parameter
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            if not trainable_parameters:
                continue
            if not is_fsdp(module):
                raise RuntimeError(
                    f"trainable synchronization root {root_name!r} is not FSDP"
                )
            for parameter in trainable_parameters:
                parameter_id = id(parameter)
                if parameter_id in covered_parameters:
                    raise RuntimeError(
                        "trainable parameter belongs to multiple FSDP roots: "
                        f"{covered_parameters[parameter_id]!r} and {root_name!r}"
                    )
                covered_parameters[parameter_id] = root_name
            roots[root_name] = module

        all_trainable_parameters = {
            id(parameter): parameter_name
            for parameter_name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        uncovered = sorted(
            all_trainable_parameters[parameter_id]
            for parameter_id in all_trainable_parameters.keys()
            - covered_parameters.keys()
        )
        if uncovered:
            raise RuntimeError(
                "trainable parameters are outside all synchronized FSDP roots: "
                + ", ".join(uncovered)
            )
        if not roots:
            raise RuntimeError("model has no trainable FSDP roots")
        return roots

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
        synchronized_roots = self.get_trainable_fsdp_roots()
        logger.info(
            "DISTRIBUTED FSDP -- synchronized trainable roots: %s",
            ", ".join(synchronized_roots),
        )
