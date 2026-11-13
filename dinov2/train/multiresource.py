"""GeoTAC RGB, tactile, depth, and point cloud pretraining."""
import copy
import torch
from torch import nn
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.layers.paired_tactile_alignment import (
    PAIRED_TARGETS, RELATION_MODES)
from dinov2.layers.tactile_temporal import build_controlled_clip_head, clip_head_contract
from dinov2.tacdino.contact_clip import protocol
from dinov2.tacdino.pointbert_runtime import PointBertBackbone, PREPROCESSING, WEIGHT_SHA256, UPSTREAM_COMMIT, runtime_identity


class MultiResourceSSLMetaArch(SSLMetaArch):
    def __init__(self, cfg):
        if cfg.train.local_data != "tac_patchalign" or not cfg.touch.get("tactile_geometry_manifest"):
            raise ValueError("GeoTAC requires contact-aligned inputs and a certified tactile geometry manifest")
        super().__init__(cfg)
        self.student["tactile_backbone"] = copy.deepcopy(self.student.backbone)
        depth = copy.deepcopy(self.student.backbone)
        old = depth.patch_embed.proj
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight[:, :3].copy_(old.weight); new.weight[:, 3:].zero_()
            if old.bias is not None: new.bias.copy_(old.bias)
        depth.patch_embed.proj = new
        self.student["depth_backbone"] = depth
        self.student["point_backbone"] = PointBertBackbone(cfg.touch.pointbert_pretrained)
        self.student["clip_head"] = build_controlled_clip_head("temporal_regions", self.embed_dim)
        for source, width in (("rgb", self.embed_dim), ("depth", self.embed_dim), ("pc", 384)):
            self.student[f"{source}_projection"] = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256))
        for name, module in self.student.items():
            if name not in self.teacher:
                self.teacher[name] = copy.deepcopy(module).requires_grad_(False)
        self.relation_distillation_mode = cfg.touch.relation_distillation_mode
        if self.relation_distillation_mode not in RELATION_MODES:
            raise ValueError("Unknown relation distillation mode")
        self.relation_distillation_temperature = float(cfg.touch.relation_distillation_temperature)
        for name in ("backbone", "depth_backbone", "point_backbone"):
            self.student[name].requires_grad_(False)
        if cfg.train.get("multiresource_activation_checkpointing", False):
            from dinov2.models.vision_transformer import BlockChunk
            for module in self.student.modules():
                if isinstance(module, BlockChunk):
                    module.activation_checkpointing = True
        self.alignment_scale = float(cfg.touch.multiresource_weight)
        if float(cfg.touch.relation_distillation_weight) != 1.0:
            raise ValueError("The released GeoTAC relation objective uses weight 1")
        self.alignment_distillation_gradient_merge = cfg.touch.alignment_distillation_gradient_merge
        if self.alignment_distillation_gradient_merge not in ("none", "anchor_pcgrad"):
            raise ValueError("Unknown alignment/distillation gradient merge")
        self._alignment_gradient_comparisons = []
        self._diagnostic_gradients = {}

    def train(self):
        super().train()
        for name in ("backbone", "depth_backbone", "point_backbone"):
            self.student[name].eval()
        return self

    def prepare_for_distributed_training(self):
        from dinov2.fsdp import get_fsdp_wrapper
        # Keep the legacy ViT BatchNorm guard intact. The only new BN owner
        # is the explicitly FP32 PC root with the buffer lifecycle below.
        point_student = self.student.pop('point_backbone')
        point_teacher = self.teacher.pop('point_backbone')
        super().prepare_for_distributed_training()
        point_teacher.load_state_dict(point_student.state_dict(), strict=True)
        self.student['point_backbone'] = get_fsdp_wrapper(
            self.cfg.compute_precision.student.point_backbone, modules_to_wrap=set())(point_student)
        self.teacher['point_backbone'] = get_fsdp_wrapper(
            self.cfg.compute_precision.teacher.point_backbone, modules_to_wrap=set())(point_teacher)
        self.get_trainable_fsdp_roots()  # Every PC parameter must have one owner.

    @torch.no_grad()
    def synchronize_auxiliary_buffers(self):
        count = super().synchronize_auxiliary_buffers()
        # Official Point-BERT includes BatchNorm. FSDP does not synchronize
        # running statistics automatically. Match DDP's rank-0 buffer policy.
        if torch.distributed.is_initialized():
            for role in (self.student, self.teacher):
                for buffer in role.point_backbone.buffers():
                    torch.distributed.broadcast(buffer, src=0)
                    count += 1
        return count

    @torch.no_grad()
    def update_teacher(self, m):
        super().update_teacher(m)
        # Parameters use the existing EMA. Running stats/counters are copied
        # from canonical student buffers, not left at pretrained values.
        source = dict(self.student.point_backbone.named_buffers())
        target = dict(self.teacher.point_backbone.named_buffers())
        if source.keys() != target.keys():
            raise RuntimeError('Point-BERT student/EMA buffer ownership differs')
        for name, value in source.items():
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(value, src=0)
            target[name].copy_(value)

    def multiresource_objective(self, images):
        device = next(self.student.backbone.parameters()).device
        batch = {key: value.to(device, non_blocking=True) for key, value in images.items() if key.startswith("resource_")}
        clip = batch["resource_tactile"]
        b, frames, c, h, w = clip.shape
        token_key = "x_norm_patchtokens"
        tactile_output = self.student.tactile_backbone(clip.reshape(b*frames, c, h, w), is_training=True)
        t = tactile_output[token_key]
        t = t.reshape(b, frames, *t.shape[1:])
        visual_output = self.student.backbone(batch["resource_rgb"], is_training=True)
        v = visual_output[token_key]
        # The repeated tactile frame contributes its CLS token from the same forward.
        rgb_cls = (visual_output["x_norm_clstoken"],
                   tactile_output["x_norm_clstoken"].reshape(b, frames, -1)[:, 0])
        del tactile_output, visual_output
        depth_input = torch.cat([batch["resource_depth"], batch["resource_tactile_depth"]])
        pc_input = torch.cat([batch["resource_pc"], batch["resource_tactile_pc"]])
        d = self.student.depth_backbone(depth_input, is_training=True)[token_key]
        p = self.student.point_backbone(pc_input, return_regions=True)
        tactile = self.student.clip_head(t)
        return self.local_four_modality_objective(batch, t, tactile, v, d, p, depth_input, rgb_cls)

    def local_four_modality_objective(self, batch, t, tactile, v, d, p, depth_input, rgb_cls=None):
        from dinov2.layers.local_four_modality import objective
        b = len(tactile)
        projected = dict(rgb=self.student.rgb_projection(v), depth=self.student.depth_projection(d[:b]),
                         pc=self.student.pc_projection(p[:b]))
        mask = torch.nn.functional.avg_pool2d(depth_input[:, 3:4].float(), 14, 14).flatten(1) > 0
        from dinov2.layers.alignment_gradient_diagnostics import AlignmentGradientComparison, AnchorPreservingGradientMerge
        comparison = (AnchorPreservingGradientMerge()
            if self.alignment_distillation_gradient_merge == "anchor_pcgrad"
            else AlignmentGradientComparison())
        self._alignment_gradient_comparisons.append(comparison)
        loss, logs = objective(projected, tactile, batch["resource_id"], d, p,
            batch["resource_tactile_geometry_valid"], mask, "full", self._diagnostic_gradients,
            temperature=self.relation_distillation_temperature, alignment_scale=self.alignment_scale,
            gradient_comparison=comparison, relation_mode=self.relation_distillation_mode,
            metadata=(batch["resource_object_id"], batch["resource_material_set"]), rgb_cls=rgb_cls)
        for name, feature in (('rgb', v), ('depth', d), ('pc', p), ('tactile', t)):
            logs['resource_'+name+'_std'] = feature.detach().float().std(0, unbiased=False).mean()
        if t.requires_grad:
            def frame_gradient(grad):
                values = grad.detach().float().flatten(2).norm(dim=2).mean(0)
                self._diagnostic_gradients.update({'tactile_frame_'+str(i): value for i, value in enumerate(values)})
            t.register_hook(frame_gradient)
        logs['resource_clip_change'] = (t[:, 1:]-t[:, :-1]).detach().float().square().mean().sqrt()
        logs['region_tactile_output_std'] = tactile.detach().float().std(0, unbiased=False).mean()
        logs['region_tactile_spatial_std'] = tactile.detach().float().std(1, unbiased=False).mean()
        logs['resource_geometry_coverage'] = batch['resource_coverage'].mean()
        logs['resource_pc_input_points'] = loss.new_tensor(batch['resource_pc'].shape[1])
        logs['resource_tactile_pc_input_points'] = loss.new_tensor(batch['resource_tactile_pc'].shape[1])
        logs['resource_pc_unique_observations_mean'] = (batch['resource_coverage'] * 98 * 98).mean()
        logs['resource_local_slot'] = batch['resource_local_slot'].float().mean()
        logs['resource_geometry_attempts'] = batch['resource_geometry_attempts'].float().mean()
        logs['resource_tactile_geometry_available'] = batch['resource_tactile_geometry_valid'].float().mean()
        logs['resource_tactile_geometry_frame'] = batch['resource_tactile_geometry_frame'].float().mean()
        logs['region_depth_valid_fraction'] = mask[:b].float().mean()
        for key, module in (('backbone', 'backbone'), ('depth', 'depth_backbone'), ('point', 'point_backbone')):
            logs['resource_encoder_'+key+'_frozen'] = loss.new_tensor(
                float(not any(param.requires_grad for param in self.student[module].parameters())))
        return loss, logs

    def forward_backward(self, *args, **kwargs):
        self._diagnostic_gradients = {}
        self._alignment_gradient_comparisons = []
        result = super().forward_backward(*args, **kwargs)
        logs = result[0] if isinstance(result, tuple) else result
        logs.update({"alignment_grad_"+k: v for k, v in self._diagnostic_gradients.items()})
        if len(self._alignment_gradient_comparisons) != 1:
            raise RuntimeError("Expected one common tactile feature tensor per microbatch")
        logs.update({"alignment_vs_distillation_" + key: value for key, value in
                     self._alignment_gradient_comparisons[0].statistics().items()})
        self._alignment_gradient_comparisons.clear()
        return result


def architecture_contract(cfg):
    """Record the objective and upstream initialization in exported checkpoints."""
    from dinov2.layers.local_four_modality import contract
    if cfg.student.arch != "vit_large":
        raise ValueError("GeoTAC checkpoint export requires the ViT-L backbone")
    loss = contract("full", float(cfg.touch.multiresource_weight),
                    relation_mode=cfg.touch.relation_distillation_mode, extra_cls_vt_alignment=True)
    return dict(
        protocol=protocol(), loss="GeoTAC", pointbert=dict(source_commit=UPSTREAM_COMMIT,
            official_weight_sha256=WEIGHT_SHA256, runtime_sha256=runtime_identity(),
            preprocessing=PREPROCESSING),
        clip_head=clip_head_contract("temporal_regions", 1024), local_four_modality_ablation=loss,
        alignment_targets=PAIRED_TARGETS, alignment_pairs=loss["alignment_pairs"],
        frozen_resource_encoders=True, extra_cls_vt_alignment=True, contact_consistent=True,
        relation_distillation_mode=cfg.touch.relation_distillation_mode,
        relation_distillation_temperature=float(cfg.touch.relation_distillation_temperature),
        alignment_distillation_gradient_merge=cfg.touch.alignment_distillation_gradient_merge,
        geometry_manifest_sha256=cfg.touch.tactile_geometry_manifest_sha256,
    )
