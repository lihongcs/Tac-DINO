# Copyright (c) Meta Platforms, Inc. and affiliates.
# #
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified for Tac-DINO.

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import time

import torch

from functools import partial
from fvcore.common.checkpoint import PeriodicCheckpointer
from dinov2.data import SamplerType, make_data_loader, make_dataset
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import (
    resolve_gradient_accumulation_steps,
    setup,
)
from dinov2.utils.utils import CosineScheduler, get_sha

import wandb
from omegaconf import OmegaConf

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.data import collate_data_and_cast, MaskingGenerator, TacDataAugmentationDINO
from dinov2.train.training_diagnostics import (objective_ledger, schedule_metrics,
    diagnostic_due, update_snapshot, update_statistics, capture_temporal_batch)


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def _sampler_seed_and_advance(
    seed,
    batch_per_gpu,
    start_iter,
    gradient_accumulation_steps=1,
):
    """Return the deterministic per-rank sampler resume state."""
    seed = int(seed)
    batch_per_gpu = int(batch_per_gpu)
    start_iter = int(start_iter)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    if batch_per_gpu < 1 or start_iter < 0 or gradient_accumulation_steps < 1:
        raise ValueError(
            "batch_per_gpu/gradient_accumulation_steps must be positive and "
            "start_iter non-negative"
        )
    return seed, start_iter * batch_per_gpu * gradient_accumulation_steps


def _optimizer_step_microbatches(iterable, gradient_accumulation_steps):
    """Group exactly N dataloader batches into one optimizer-step input."""

    gradient_accumulation_steps = int(gradient_accumulation_steps)
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    iterator = iter(iterable)
    while True:
        group = []
        for _ in range(gradient_accumulation_steps):
            try:
                group.append(next(iterator))
            except StopIteration:
                if group:
                    raise RuntimeError(
                        "dataloader ended inside a gradient-accumulation step"
                    )
                return
        yield tuple(group)


def _student_no_sync_context(student_modules, enabled):
    """Disable reduction on every independent trainable FSDP root."""

    if not enabled:
        return contextlib.nullcontext()
    stack = contextlib.ExitStack()
    entered = []
    try:
        for module_name, module in student_modules.items():
            if not any(parameter.requires_grad for parameter in module.parameters()):
                continue
            no_sync = getattr(module, "no_sync", None)
            if not callable(no_sync):
                raise RuntimeError(
                    "gradient accumulation requires every trainable FSDP root "
                    f"to expose FSDP.no_sync(); missing on {module_name!r}"
                )
            stack.enter_context(no_sync())
            entered.append(module_name)
        if not entered:
            raise RuntimeError(
                "gradient accumulation found no trainable FSDP roots"
            )
        return stack
    except Exception:
        stack.close()
        raise


def _validate_optimizer_parameter_inventory(optimizer, fsdp_roots):
    """Require one optimizer entry for every trainable FSDP-owned parameter."""

    expected = {}
    for root_name, module in fsdp_roots.items():
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in expected:
                raise RuntimeError(
                    "trainable parameter is exposed by multiple FSDP roots: "
                    f"{expected[parameter_id]!r} and {root_name!r}"
                )
            expected[parameter_id] = root_name

    actual = {}
    duplicates = []
    for group_index, parameter_group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(parameter_group["params"]):
            parameter_id = id(parameter)
            location = f"group={group_index},parameter={parameter_index}"
            if parameter_id in actual:
                duplicates.append((actual[parameter_id], location))
            else:
                actual[parameter_id] = location
    if duplicates:
        raise RuntimeError(
            "optimizer contains duplicate parameter entries: "
            + "; ".join(f"{first}/{second}" for first, second in duplicates)
        )

    missing = sorted(
        expected[parameter_id]
        for parameter_id in expected.keys() - actual.keys()
    )
    unexpected = sorted(
        actual[parameter_id]
        for parameter_id in actual.keys() - expected.keys()
    )
    if missing or unexpected:
        raise RuntimeError(
            "optimizer/FSDP trainable parameter inventory mismatch: "
            f"missing_roots={missing}, unexpected_entries={unexpected}"
        )
    if not expected:
        raise RuntimeError("optimizer parameter inventory is empty")
    return len(expected)


def _checkpoint_period_steps(saveckp_freq, official_epoch_length):
    saveckp_freq = int(saveckp_freq)
    official_epoch_length = int(official_epoch_length)
    if saveckp_freq < 1 or official_epoch_length < 1:
        raise ValueError("saveckp_freq and OFFICIAL_EPOCH_LENGTH must be positive")
    return saveckp_freq * official_epoch_length


def _validated_stop_after_steps(value, max_iter):
    stop_after_steps = int(value)
    max_iter = int(max_iter)
    if stop_after_steps < 0:
        raise ValueError("TACDINO_STOP_AFTER_STEPS must be non-negative")
    if stop_after_steps > max_iter:
        raise ValueError(
            "TACDINO_STOP_AFTER_STEPS cannot exceed the configured training schedule "
            f"({stop_after_steps} > {max_iter})"
        )
    return stop_after_steps


def _should_export_eval_checkpoint(
    iteration,
    max_iter,
    eval_period_iterations,
    stop_after_steps=0,
):
    """Export every configured milestone and always export the actual run end."""
    completed_steps = int(iteration) + 1
    run_end = int(stop_after_steps) or int(max_iter)
    period = int(eval_period_iterations)
    return completed_steps == run_end or (
        period > 0 and completed_steps % period == 0
    )


def _should_export_initial_checkpoint(value, start_iter):
    if str(value) not in {"0", "1"}:
        raise ValueError("TACDINO_EXPORT_INITIAL_CHECKPOINT must be 0 or 1")
    return str(value) == "1" and int(start_iter) == 0


def _unpack_forward_backward_result(result):
    """Accept the objective-aware result and the legacy loss-dict API."""
    if isinstance(result, tuple):
        if len(result) != 2:
            raise TypeError("forward_backward tuple must be (loss_dict, objective)")
        loss_dict, optimization_objective = result
    else:
        loss_dict = result
        # Backward compatibility for alternate architectures that still return
        # only components.  SSLMetaArch uses the objective-aware tuple below.
        optimization_objective = sum(loss_dict.values())
    if not isinstance(loss_dict, dict):
        raise TypeError("forward_backward loss components must be a dict")
    return loss_dict, optimization_objective


def _loss_log_values(optimization_objective, loss_dict):
    """Return the true backward objective under both legacy log names.

    ``loss_dict`` intentionally contains diagnostics such as retrieval accuracy,
    chance level, and representation variance in addition to objective terms.
    Summing that dictionary therefore has no mathematical meaning and changes
    when a metric is merely added or removed. Keep ``component_sum`` as a
    compatibility key, but make it an alias of the optimization objective.
    """
    del loss_dict
    objective = float(optimization_objective)
    return objective, objective


def _clip_grad_norm_globally(student_modules, max_norm):
    """Clip disjoint student FSDP modules with one shared coefficient.

    Each top-level student submodule is wrapped independently by FSDP. Calling
    ``clip_grad_norm_`` with the configured threshold on every submodule would
    rescale them independently and destroy the relative magnitudes of their
    gradients. Instead, first ask every trainable submodule for its distributed
    pre-clip norm, combine those disjoint norms, and then scale every local
    gradient shard by the same coefficient.

    Frozen modules are excluded completely. The returned per-module norms are
    pre-clip values and are suitable for the existing diagnostics.
    """
    max_norm = float(max_norm)
    if max_norm < 0:
        raise ValueError("max_norm must be non-negative")

    preclip_norms = {}
    trainable_modules = []
    for module_name, module in student_modules.items():
        trainable_parameters = tuple(
            parameter for parameter in module.parameters() if parameter.requires_grad
        )
        if not trainable_parameters:
            continue
        norm_value = module.clip_grad_norm_(float("inf"))
        norm = (
            float(norm_value.detach().float().item())
            if isinstance(norm_value, torch.Tensor)
            else float(norm_value)
        )
        preclip_norms[module_name] = norm
        trainable_modules.append(trainable_parameters)

    global_norm = math.sqrt(sum(norm * norm for norm in preclip_norms.values()))
    if math.isfinite(global_norm):
        clip_coefficient = min(1.0, max_norm / (global_norm + 1.0e-6))
    else:
        # Never hide a non-finite gradient by multiplying it by zero. The
        # existing GradScaler/BF16 finite checks will reject or skip the step.
        clip_coefficient = 1.0

    if clip_coefficient < 1.0:
        with torch.no_grad():
            for trainable_parameters in trainable_modules:
                for parameter in trainable_parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(clip_coefficient)

    return preclip_norms, global_norm, clip_coefficient


def _sha256_file(path):
    if not path or not os.path.isfile(path):
        return "unavailable"
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer_checkpointables(optimizer, scaler):
    """Keep AMP adaptation across stage/checkpoint boundaries, like Adam state."""
    state = {"optimizer": optimizer}
    if scaler is not None:
        state["grad_scaler"] = scaler
    return state


def _checkpoint_provenance(cfg, checkpoint_iteration):
    output_dir = os.path.abspath(str(cfg.train.output_dir))
    config_path = os.path.abspath(
        os.environ.get(
            "TACDINO_RESOLVED_CONFIG_PATH",
            os.path.join(output_dir, "config.yaml"),
        )
    )
    split_path = os.environ.get("TACDINO_TRAIN_SPLIT_FILE") or os.environ.get(
        "TRAIN_SPLIT_FILE"
    )
    if not split_path and os.environ.get("META_ROOT"):
        split_path = os.path.join(
            os.environ["META_ROOT"], "split0111", "train_split.json"
        )
    split_path = os.path.abspath(split_path) if split_path else "unavailable"
    git_sha = os.environ.get("TACDINO_GIT_SHA")
    if not git_sha:
        git_description = get_sha()
        git_sha = git_description.split("sha: ", 1)[-1].split(",", 1)[0]
    input_contract = "geotac_contact_consistent"
    # ``setup`` already validates this contract. Keep provenance construction
    # self-contained so offline checkpoint-inspection tools do not need an
    # initialized process group.
    accumulation_steps = int(
        getattr(cfg.train, "gradient_accumulation_steps", 1)
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    microbatch_per_gpu = int(getattr(cfg.train, "batch_size_per_gpu", 0))
    provenance = {
        "schema_version": "tacdino.checkpoint-provenance.v1",
        "run_id": os.environ.get("TACDINO_RUN_ID", os.path.basename(output_dir)),
        "iteration": int(checkpoint_iteration),
        "git_sha": git_sha,
        "resolved_config_path": config_path,
        "resolved_config_sha256": _sha256_file(config_path),
        "train_split_path": split_path,
        "train_split_sha256": os.environ.get(
            "TACDINO_TRAIN_SPLIT_SHA256",
            _sha256_file(split_path),
        ),
        "input_contract": input_contract,
        "optimizer_step_contract": {
            "implementation": str(
                getattr(
                    cfg.train,
                    "gradient_accumulation_implementation",
                    "native_fsdp_no_sync_v1",
                )
            ),
            "iteration_unit": "completed_optimizer_steps",
            "loss_batch_scope": str(
                getattr(cfg.train, "loss_batch_scope", "microbatch_global")
            ),
            "gradient_accumulation_steps": accumulation_steps,
            "microbatch_per_gpu": microbatch_per_gpu,
            "world_size": world_size,
            "loss_global_batch": microbatch_per_gpu * world_size,
            "effective_optimizer_batch": (
                microbatch_per_gpu * world_size * accumulation_steps
            ),
        },
    }
    return provenance


def _atomic_torch_save(payload, destination):
    temporary = f"{destination}.tmp.{os.getpid()}"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--retrive_loss_type",
        default="none",
        type=str,
        help="Retrive loss type",
    )
    parser.add_argument(
        "--resume-lr-scale",
        default=1.0,
        type=float,
        help="Scale factor for learning rate when resuming training (e.g., 0.1 to reduce LR by 10x)",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def _sample_time_momentum_exponent(current_global_batch, reference_global_batch):
    current_global_batch = int(current_global_batch)
    reference_global_batch = int(reference_global_batch)
    if current_global_batch <= 0 or reference_global_batch <= 0:
        raise ValueError("EMA current/reference global batches must be positive")
    return current_global_batch / reference_global_batch


def schedule_steps(cfg):
    """Explicit update counts take precedence; legacy recipes are unchanged."""
    epoch = int(cfg.train.OFFICIAL_EPOCH_LENGTH)
    result = dict(
        total=int(cfg.optim.get("total_steps", cfg.optim.epochs * epoch)),
        warmup=int(cfg.optim.get("warmup_steps", cfg.optim.warmup_epochs * epoch)),
        freeze=int(cfg.optim.get("freeze_last_layer_steps", cfg.optim.freeze_last_layer_epochs * epoch)),
        temperature=int(cfg.teacher.get("warmup_teacher_temp_steps", cfg.teacher.warmup_teacher_temp_epochs * epoch)),
    )
    if result["total"] <= 0 or any(result[k] < 0 for k in ("warmup", "freeze", "temperature")) or result["warmup"] > result["total"]:
        raise ValueError(f"invalid absolute training schedule: {result}")
    return result


def build_schedulers(cfg, lr_scale=1.0):
    steps = schedule_steps(cfg)
    lr = dict(
        base_value=cfg.optim["lr"] * lr_scale,
        final_value=cfg.optim["min_lr"] * lr_scale,
        total_iters=steps["total"],
        warmup_iters=steps["warmup"],
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=steps["total"],
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=steps["total"],
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=steps["temperature"],
        warmup_iters=steps["temperature"],
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    momentum_reference_global_batch = int(
        getattr(cfg.teacher, "momentum_reference_global_batch", 0)
    )
    if momentum_reference_global_batch > 0:
        current_optimizer_batch = (
            int(cfg.train.batch_size_per_gpu)
            * int(distributed.get_global_size())
            * resolve_gradient_accumulation_steps(cfg)
        )
        momentum_exponent = _sample_time_momentum_exponent(
            current_optimizer_batch,
            momentum_reference_global_batch,
        )
        # EMA is updated once per optimizer step. Raising each reference-batch
        # coefficient to B_current/B_reference preserves its decay per sample
        # when a capacity run changes the global batch.
        momentum_schedule.schedule = momentum_schedule.schedule ** momentum_exponent
        momentum_schedule.final_value = (
            float(momentum_schedule.final_value) ** momentum_exponent
        )
        logger.info(
            "Sample-time EMA scaling: effective_optimizer_batch=%d "
            "reference=%d exponent=%.6f "
            "initial_momentum=%.8f",
            current_optimizer_batch,
            momentum_reference_global_batch,
            momentum_exponent,
            float(momentum_schedule.schedule[0]),
        )
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : steps["freeze"]
    ] = 0  # mimicking the original schedules

    if lr_scale != 1.0:
        logger.info(f"LR schedule scaled by {lr_scale}: base_lr={cfg.optim['lr'] * lr_scale:.2e}, min_lr={cfg.optim['min_lr'] * lr_scale:.2e}")

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration, checkpoint_iteration=None):
    model.synchronize_auxiliary_buffers()
    new_state_dict = model.teacher.state_dict()
    new_student_state_dict = model.student.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        provenance = _checkpoint_provenance(
            cfg,
            checkpoint_iteration if checkpoint_iteration is not None else iteration,
        )
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        teacher_payload = {
            "teacher": new_state_dict,
            "tacdino_provenance": provenance,
        }
        student_payload = {
            "student": new_student_state_dict,
            "tacdino_provenance": provenance,
        }
        _atomic_torch_save(teacher_payload, teacher_ckp_path)
        student_ckp_path = os.path.join(eval_dir, "student_checkpoint.pth")
        _atomic_torch_save(student_payload, student_ckp_path)


def do_train(cfg, model, resume=False, resume_lr_scale=1.0):
    model.train()
    trainable_fsdp_roots = model.get_trainable_fsdp_roots()
    gradient_accumulation_steps = resolve_gradient_accumulation_steps(cfg)
    gradient_accumulation_implementation = str(
        getattr(
            cfg.train,
            "gradient_accumulation_implementation",
            "native_fsdp_no_sync_v1",
        )
    )
    loss_batch_scope = str(
        getattr(cfg.train, "loss_batch_scope", "microbatch_global")
    )
    world_size = int(distributed.get_global_size())
    forward_per_rank = int(cfg.train.batch_size_per_gpu)
    forward_global_batch = forward_per_rank * world_size
    loss_global_batch = forward_global_batch
    effective_optimizer_batch = (
        forward_global_batch * gradient_accumulation_steps
    )
    input_dtype_name = str(
        cfg.compute_precision.student.backbone.mixed_precision.param_dtype
    )
    input_dtype_by_name = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if input_dtype_name not in input_dtype_by_name:
        raise ValueError(f"unsupported training input dtype: {input_dtype_name}")
    inputs_dtype = input_dtype_by_name[input_dtype_name]
    fp16_scaler = model.fp16_scaler  # for mixed precision training
    max_consecutive_skipped_steps = int(
        os.environ.get("TACDINO_MAX_CONSECUTIVE_SKIPPED_STEPS", "8")
    )
    if max_consecutive_skipped_steps < 1:
        raise ValueError("TACDINO_MAX_CONSECUTIVE_SKIPPED_STEPS must be positive")
    consecutive_skipped_steps = 0
    total_skipped_steps = 0
    wandb_log_period = int(os.environ.get("WANDB_LOG_PERIOD", "10"))
    if wandb_log_period < 1:
        raise ValueError("WANDB_LOG_PERIOD must be positive")
    logger.info(
        "Optimizer-step audit enabled: input_dtype=%s grad_scaler=%s "
        "max_consecutive_skips=%d wandb_log_period=%d "
        "gradient_accumulation=%s:%d loss_batch_scope=%s "
        "loss_global_batch=%d effective_optimizer_batch=%d",
        input_dtype_name,
        fp16_scaler is not None,
        max_consecutive_skipped_steps,
        wandb_log_period,
        gradient_accumulation_implementation,
        gradient_accumulation_steps,
        loss_batch_scope,
        loss_global_batch,
        effective_optimizer_batch,
    )

    # setup optimizer

    args = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "job_type": "train",
    }
    if distributed.is_main_process():
        wandb.init(project=os.environ.get("WANDB_PROJECT", "tacdino"), config=args, name=os.path.basename(cfg.train.output_dir), dir=cfg.train.output_dir)

    optimizer = build_optimizer(cfg, model.get_params_groups())
    optimizer_parameter_count = _validate_optimizer_parameter_inventory(
        optimizer,
        trainable_fsdp_roots,
    )
    logger.info(
        "Optimizer parameter inventory validated: unique_trainable_parameters=%d",
        optimizer_parameter_count,
    )

    # 判断是否需要缩放学习率：仅在 resume 且 scale != 1.0 时生效
    effective_lr_scale = resume_lr_scale if (resume and resume_lr_scale != 1.0) else 1.0
    if effective_lr_scale != 1.0:
        logger.info(f"RESUME -- Scaling learning rate schedule by factor: {effective_lr_scale}")

    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg, lr_scale=effective_lr_scale)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, save_to_disk=True,
                                  **_optimizer_checkpointables(optimizer, fp16_scaler))

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    if fp16_scaler is not None:
        logger.info("AMP scale after checkpoint restore: %g", fp16_scaler.get_scale())

    # resume 后重置优化器动量状态，防止旧动量导致梯度爆炸
    if resume and start_iter > 0 and effective_lr_scale != 1.0:
        logger.info("RESUME -- Resetting optimizer state (clearing momentum buffers) to stabilize training")
        optimizer.state.clear()
        # 同时重置 fp16 scaler，防止旧 scale 值过大
        if fp16_scaler is not None:
            logger.info("RESUME -- Resetting FP16 GradScaler scale")
            fp16_scaler._scale = torch.tensor(2.0 ** 16, dtype=torch.float32).cuda()
            fp16_scaler._growth_tracker = torch.tensor(0, dtype=torch.int32).cuda()

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = schedule_steps(cfg)["total"]
    stop_after_steps = _validated_stop_after_steps(
        os.environ.get("TACDINO_STOP_AFTER_STEPS", "0"),
        max_iter,
    )
    run_end_iter = stop_after_steps or max_iter
    if start_iter >= run_end_iter:
        logger.info(
            "Training target already reached: start_iter=%d run_end_iter=%d",
            start_iter,
            run_end_iter,
        )
        return {}

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=_checkpoint_period_steps(
            cfg.train.saveckp_freq,
            OFFICIAL_EPOCH_LENGTH,
        ),
        max_iter=max_iter,
        max_to_keep=3,
    )

    export_initial_checkpoint = os.environ.get(
        "TACDINO_EXPORT_INITIAL_CHECKPOINT", "0"
    )
    if _should_export_initial_checkpoint(export_initial_checkpoint, start_iter):
        do_test(
            cfg,
            model,
            "training_-1",
            checkpoint_iteration=start_iter - 1,
        )
        torch.cuda.synchronize()
    elif export_initial_checkpoint == "1":
        logger.info(
            "Skipping training_-1 export on resume at iteration %d",
            start_iter,
        )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    data_transform = TacDataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        local_data=cfg.train.local_data,
        touch_config=cfg.touch,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
        local_data=cfg.train.local_data,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    sampler_seed, sampler_advance = _sampler_seed_and_advance(
        cfg.train.seed,
        cfg.train.batch_size_per_gpu,
        start_iter,
        gradient_accumulation_steps,
    )
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        prefetch_factor=getattr(cfg.train, "prefetch_factor", None),
        shuffle=True,
        seed=sampler_seed,
        sampler_type=sampler_type,
        sampler_advance=sampler_advance,
        drop_last=True,
        collate_fn=collate_fn,
    )
    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"
    diagnostic_period = int(os.environ.get("TACDINO_DIAGNOSTIC_PERIOD", "84"))

    optimizer_step_batches = _optimizer_step_microbatches(
        data_loader,
        gradient_accumulation_steps,
    )
    for microbatches in metric_logger.log_every(
        optimizer_step_batches,
        10,
        header,
        run_end_iter,
        start_iter,
    ):
        step_compute_started_at = time.perf_counter()
        if iteration >= run_end_iter:
            break

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # One optimizer step consumes exactly N microbatches. Scheduler state,
        # zero_grad, clipping, scaler/optimizer stepping, teacher EMA,
        # checkpointing and evaluation are deliberately outside this loop.
        optimizer.zero_grad(set_to_none=True)
        diagnostic_step = diagnostic_due(iteration, diagnostic_period)
        loss_keys_for_reduction = None
        loss_values_sum = None
        expected_batch_size = None
        for microstep_index, data in enumerate(microbatches):
            # DDP broadcasts BatchNorm buffers at the beginning of every
            # wrapper forward, including no_sync forwards. Reproduce that
            # lifecycle for the Point-BERT backbone.
            model.synchronize_auxiliary_buffers()
            current_batch_size = int(data["rank_local_batch_size"])
            if expected_batch_size is None:
                expected_batch_size = current_batch_size
            elif current_batch_size != expected_batch_size:
                raise RuntimeError(
                    "all microbatches in an optimizer step must have the same "
                    "rank-local size"
                )

            if microstep_index == 0 and distributed.is_main_process():
                if diagnostic_step:
                    # An optional diagnostic I/O failure must not abort training.
                    try:
                        capture_temporal_batch(data, cfg.train.output_dir, iteration)
                    except Exception:
                        logger.exception("Diagnostic batch capture failed; training unchanged")

            synchronize_gradients = (
                microstep_index + 1 == gradient_accumulation_steps
            )
            with _student_no_sync_context(
                trainable_fsdp_roots,
                enabled=not synchronize_gradients,
            ):
                forward_backward_result = model.forward_backward(
                    data,
                    teacher_temp=teacher_temp,
                    backward_loss_scale=(
                        1.0
                        if gradient_accumulation_steps == 1
                        else 1.0 / gradient_accumulation_steps
                    ),
                    return_objective=True,
                )
            loss_dict, optimization_objective = _unpack_forward_backward_result(
                forward_backward_result
            )
            loss_dict.update(objective_ledger(loss_dict, cfg, optimization_objective))

            # Preserve stable raw (unscaled) FP32 metrics before FSDP
            # resharding and optimizer mutation. Metrics are averaged across
            # microsteps; only the backward objective is divided before
            # backward. Never all-reduce autograd-connected loss tensors.
            microstep_loss_keys = tuple(loss_dict)
            if loss_keys_for_reduction is None:
                loss_keys_for_reduction = microstep_loss_keys
            elif microstep_loss_keys != loss_keys_for_reduction:
                raise RuntimeError(
                    "loss metric keys changed inside an optimizer step"
                )
            microstep_loss_values = torch.stack(
                [optimization_objective.detach().float().reshape(())]
                + [
                    loss_dict[key].detach().float().reshape(())
                    for key in loss_keys_for_reduction
                ]
            )

            # Fail immediately after every microstep on every rank. Waiting
            # until the optimizer boundary could let later backward passes
            # obscure the first invalid objective.
            local_losses_finite = torch.isfinite(microstep_loss_values).all()
            all_ranks_finite = local_losses_finite.to(dtype=torch.int32)
            if distributed.get_global_size() > 1:
                torch.distributed.all_reduce(
                    all_ranks_finite,
                    op=torch.distributed.ReduceOp.MIN,
                )
            if not bool(all_ranks_finite.item()):
                local_loss_values = dict(
                    zip(
                        ("optimization_objective",) + loss_keys_for_reduction,
                        microstep_loss_values.cpu().tolist(),
                    )
                )
                logger.error(
                    "Non-finite loss before optimizer step on rank %d at "
                    "optimizer_iteration=%d microstep=%d/%d: %s",
                    distributed.get_global_rank(),
                    iteration,
                    microstep_index + 1,
                    gradient_accumulation_steps,
                    local_loss_values,
                )
                raise AssertionError(
                    "Non-finite loss detected inside gradient accumulation"
                )

            if loss_values_sum is None:
                loss_values_sum = microstep_loss_values
            else:
                loss_values_sum = loss_values_sum + microstep_loss_values

        if expected_batch_size is None or loss_values_sum is None:
            raise RuntimeError("optimizer step received no microbatches")
        current_batch_size = expected_batch_size
        loss_values_for_reduction = (
            loss_values_sum
            if gradient_accumulation_steps == 1
            else loss_values_sum / gradient_accumulation_steps
        )

        # clip gradients

        grad_norms = {}
        update_sample = update_snapshot(trainable_fsdp_roots, model.teacher) if diagnostic_step else None
        if fp16_scaler is not None:
            scale_before = float(fp16_scaler.get_scale())
            # Always unscale explicitly. Besides making clipping correct, this
            # materializes found_inf_per_device so skipped updates are auditable.
            fp16_scaler.unscale_(optimizer)
            grad_norms, grad_norm, grad_clip_coefficient = (
                _clip_grad_norm_globally(
                    trainable_fsdp_roots,
                    cfg.optim.clip_grad if cfg.optim.clip_grad else float("inf"),
                )
            )
            fp16_scaler.step(optimizer)
            optimizer_state = getattr(fp16_scaler, "_per_optimizer_states", {}).get(
                id(optimizer), {}
            )
            found_inf_values = optimizer_state.get("found_inf_per_device", {}).values()
            found_inf = sum(float(value.detach().item()) for value in found_inf_values)
            fp16_scaler.update()
            scale_after = float(fp16_scaler.get_scale())
            # ShardedGradScaler synchronizes found_inf across ranks. The scale
            # comparison remains a conservative fallback for API changes.
            optimizer_step_applied = found_inf == 0.0 and scale_after >= scale_before
        else:
            scale_before = scale_after = 1.0
            found_inf = 0.0
            grad_norms, grad_norm, grad_clip_coefficient = (
                _clip_grad_norm_globally(
                    trainable_fsdp_roots,
                    cfg.optim.clip_grad if cfg.optim.clip_grad else float("inf"),
                )
            )
            finite_gradients = torch.tensor(
                int(
                    math.isfinite(grad_norm)
                    and math.isfinite(grad_clip_coefficient)
                    and all(math.isfinite(value) for value in grad_norms.values())
                ),
                device="cuda",
                dtype=torch.int32,
            )
            if distributed.get_global_size() > 1:
                torch.distributed.all_reduce(
                    finite_gradients, op=torch.distributed.ReduceOp.MIN
                )
            if not bool(finite_gradients.item()):
                logger.error("Non-finite BF16/FP32 update roots at iteration=%d: %s", iteration, grad_norms)
                raise RuntimeError(
                    "non-finite BF16/FP32 gradient norm detected before optimizer step; "
                    "the run was stopped before corrupting weights"
                )
            optimizer.step()
            optimizer_step_applied = True

        applied_on_all_ranks = torch.tensor(
            int(optimizer_step_applied), device="cuda", dtype=torch.int32
        )
        if distributed.get_global_size() > 1:
            torch.distributed.all_reduce(
                applied_on_all_ranks, op=torch.distributed.ReduceOp.MIN
            )
        optimizer_step_applied = bool(applied_on_all_ranks.item())

        if optimizer_step_applied:
            consecutive_skipped_steps = 0
        else:
            consecutive_skipped_steps += 1
            total_skipped_steps += 1
            logger.error("Non-finite update roots at iteration=%d: %s", iteration, grad_norms)
            logger.error(
                "OPTIMIZER STEP SKIPPED at iteration=%d scale_before=%g scale_after=%g "
                "found_inf=%g grad_norm=%g consecutive=%d total=%d",
                iteration,
                scale_before,
                scale_after,
                found_inf,
                grad_norm,
                consecutive_skipped_steps,
                total_skipped_steps,
            )
            if consecutive_skipped_steps >= max_consecutive_skipped_steps:
                raise RuntimeError(
                    f"training aborted at configured AMP skip limit: "
                    f"consecutive={consecutive_skipped_steps}, total={total_skipped_steps}, "
                    f"limit={max_consecutive_skipped_steps}; last update was not applied"
                )

        # perform teacher EMA update
        # Never update the EMA teacher from a student whose optimizer update
        # was skipped. That would make a failed FP16 run look superficially alive.
        if optimizer_step_applied:
            model.update_teacher(mom)
        update_metrics = (update_statistics(update_sample, mom if optimizer_step_applied else 1.)
                          if update_sample is not None else {})

        # logging

        # One vector collective replaces one tiny all-reduce per loss. This is
        # numerically identical and removes substantial fixed overhead when the
        # contact-field objective reports many 1/2/4/8-touch components.
        if distributed.get_global_size() > 1:
            torch.distributed.all_reduce(loss_values_for_reduction)
        reduced_loss_values = (
            loss_values_for_reduction / distributed.get_global_size()
        ).cpu().tolist()
        losses_reduced = reduced_loss_values[0]
        loss_dict_reduced = dict(
            zip(loss_keys_for_reduction, reduced_loss_values[1:])
        )
        losses_reduced, component_sum_reduced = _loss_log_values(
            losses_reduced,
            loss_dict_reduced,
        )

        if not math.isfinite(losses_reduced) or not math.isfinite(
            component_sum_reduced
        ):
            logger.info("Non-finite loss detected")
            logger.info(f"Loss dict components: {loss_dict_reduced}")
            logger.info(f"Current iteration: {iteration}")
            logger.info(f"LR: {lr}, WD: {wd}, Mom: {mom}")
            if distributed.is_main_process():
                 # Force a sync to ensure all processes stop or log
                 pass
            raise AssertionError("Non-finite loss detected")
        
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(**schedule_metrics(cfg, iteration, total_skipped_steps,
            start_iter, effective_optimizer_batch, teacher_temp))
        metric_logger.update(diagnostic_update_sample_measured=float(diagnostic_step),
            diagnostic_update_sample_step=float(iteration+1) if diagnostic_step else -1.)
        if update_metrics:
            metric_logger.update(**update_metrics)
        metric_logger.update(
            gradient_accumulation_steps=float(gradient_accumulation_steps),
            loss_batch_scope_microbatch_global=float(loss_batch_scope == "microbatch_global"),
            forward_global_batch=float(forward_global_batch),
            effective_rank_batch_size=float(
                current_batch_size * gradient_accumulation_steps
            ),
            loss_global_batch=float(loss_global_batch),
            effective_optimizer_batch=float(effective_optimizer_batch),
            microsteps_completed=float(gradient_accumulation_steps),
            optimizer_step_applied=float(optimizer_step_applied),
            optimizer_step_skipped=float(not optimizer_step_applied),
            optimizer_steps_skipped_total=float(total_skipped_steps),
            consecutive_optimizer_steps_skipped=float(consecutive_skipped_steps),
            grad_scale=float(scale_after),
            grad_found_inf=float(found_inf),
            grad_norm=float(grad_norm),
            grad_clip_coefficient=float(grad_clip_coefficient),
            **{
                f"grad_norm_{name}": float(value)
                for name, value in grad_norms.items()
            },
        )
        resource_log_due = (
            iteration % wandb_log_period == 0 or iteration + 1 == run_end_iter
        )
        resource_metrics = {}
        metric_logger.update(resource_metrics_measured=float(resource_log_due))
        if resource_log_due:
            resource_values = torch.tensor(
                [
                    time.perf_counter() - step_compute_started_at,
                    torch.cuda.max_memory_allocated() / (1024.0 ** 3),
                    torch.cuda.max_memory_reserved() / (1024.0 ** 3),
                ],
                device="cuda",
                dtype=torch.float64,
            )
            if distributed.get_global_size() > 1:
                torch.distributed.all_reduce(
                    resource_values,
                    op=torch.distributed.ReduceOp.MAX,
                )
            slowest_step_seconds, max_allocated_gib, max_reserved_gib = (
                resource_values.cpu().tolist()
            )
            resource_metrics = {
                "max_memory_allocated_gib": max_allocated_gib,
                "max_memory_reserved_gib": max_reserved_gib,
                "global_samples_per_sec": (
                    current_batch_size
                    * gradient_accumulation_steps
                    * distributed.get_global_size()
                    / max(slowest_step_seconds, 1.0e-12)
                ),
            }
            metric_logger.update(**resource_metrics)
        metric_logger.update(
            total_loss=losses_reduced,
            component_sum=component_sum_reduced,
            **loss_dict_reduced,
        )
        
        if distributed.is_main_process() and (
            iteration % wandb_log_period == 0 or iteration + 1 == run_end_iter
        ):
            log_dict = {
                "train/lr": lr,
                "train/wd": wd,
                "train/mom": mom,
                "train/last_layer_lr": last_layer_lr,
                "train/total_loss": losses_reduced,
                "train/component_sum": component_sum_reduced,
                **{"train/"+k: v for k, v in schedule_metrics(cfg, iteration,
                    total_skipped_steps, start_iter, effective_optimizer_batch, teacher_temp).items()},
                **{"train/"+k: v for k, v in update_metrics.items()},
                "train/gradient_accumulation_steps": float(
                    gradient_accumulation_steps
                ),
                "train/loss_batch_scope_microbatch_global": float(loss_batch_scope == "microbatch_global"),
                "train/forward_global_batch": float(forward_global_batch),
                "train/loss_global_batch": float(loss_global_batch),
                "train/effective_optimizer_batch": float(
                    effective_optimizer_batch
                ),
                "train/optimizer_step_applied": float(optimizer_step_applied),
                "train/optimizer_steps_skipped_total": float(total_skipped_steps),
                "train/grad_scale": float(scale_after),
                "train/grad_found_inf": float(found_inf),
                "train/grad_norm": float(grad_norm),
                "train/grad_clip_coefficient": float(grad_clip_coefficient),
                **{
                    f"train/grad_norm_{name}": float(value)
                    for name, value in grad_norms.items()
                },
                **{f"train/{key}": value for key, value in resource_metrics.items()},
            }
            for k, v in loss_dict_reduced.items():
                log_dict[f"train/{k}"] = v
            wandb.log(log_dict, step=iteration)

        # checkpointing and testing
        
        if _should_export_eval_checkpoint(
            iteration,
            max_iter,
            cfg.evaluation.eval_period_iterations,
            stop_after_steps,
        ):
            do_test(
                cfg,
                model,
                f"training_{iteration}",
                checkpoint_iteration=iteration,
            )
            torch.cuda.synchronize()

        if stop_after_steps and iteration + 1 == stop_after_steps:
            # PeriodicCheckpointer only treats the configured full schedule as
            # final.  A bounded pilot therefore explicitly writes the same
            # resumable per-rank FSDP checkpoint before every rank exits.
            checkpointer.save(f"model_{iteration:07d}", iteration=iteration)
            sharded_checkpoint_path = os.path.join(
                cfg.train.output_dir,
                f"model_{iteration:07d}.rank_{distributed.get_global_rank()}.pth",
            )
            checkpoint_written = torch.tensor(
                int(
                    os.path.isfile(sharded_checkpoint_path)
                    and os.path.getsize(sharded_checkpoint_path) > 0
                ),
                device="cuda",
                dtype=torch.int32,
            )
            if distributed.get_global_size() > 1:
                torch.distributed.all_reduce(
                    checkpoint_written,
                    op=torch.distributed.ReduceOp.MIN,
                )
            if not bool(checkpoint_written.item()):
                raise RuntimeError(
                    "bounded pilot ended without a complete sharded resume checkpoint"
                )
            if distributed.get_global_size() > 1:
                torch.distributed.barrier()
        else:
            periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(
            cfg,
            model,
            f"manual_{iteration}",
            checkpoint_iteration=iteration - 1,
        )

    do_train(
        cfg,
        model,
        resume=not args.no_resume,
        resume_lr_scale=args.resume_lr_scale,
    )


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
