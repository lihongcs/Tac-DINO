# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified for Tac-DINO.

import math
import logging
import os

from omegaconf import OmegaConf

import dinov2.distributed as distributed
from dinov2.logging import setup_logging
from dinov2.utils import utils
from dinov2.configs import dinov2_default_config


logger = logging.getLogger("dinov2")

GRADIENT_ACCUMULATION_IMPLEMENTATION = "native_fsdp_no_sync_v1"
LOSS_BATCH_SCOPE = "microbatch_global"


def resolve_gradient_accumulation_steps(cfg, environ=None):
    """Validate the recorded config and optional environment assertions.

    The config is authoritative so the resolved YAML/checkpoint provenance
    always describes the real optimizer-step semantics. Environment variables
    are assertions, not hidden overrides.
    """

    configured = getattr(cfg.train, "gradient_accumulation_steps", 1)
    if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
        raise ValueError("train.gradient_accumulation_steps must be a positive integer")
    implementation = str(
        getattr(
            cfg.train,
            "gradient_accumulation_implementation",
            GRADIENT_ACCUMULATION_IMPLEMENTATION,
        )
    )
    if implementation != GRADIENT_ACCUMULATION_IMPLEMENTATION:
        raise ValueError(
            "unsupported train.gradient_accumulation_implementation: "
            f"{implementation!r}"
        )
    loss_batch_scope = str(
        getattr(cfg.train, "loss_batch_scope", LOSS_BATCH_SCOPE)
    )
    if loss_batch_scope != LOSS_BATCH_SCOPE:
        raise ValueError(f"Unsupported loss batch scope: {loss_batch_scope}")

    environment = os.environ if environ is None else environ
    expected_steps = environment.get("TACDINO_GRADIENT_ACCUMULATION_STEPS")
    if expected_steps is not None:
        if not str(expected_steps).isdigit() or int(expected_steps) < 1:
            raise ValueError(
                "TACDINO_GRADIENT_ACCUMULATION_STEPS must be a positive integer"
            )
        if int(expected_steps) != configured:
            raise ValueError(
                "TACDINO_GRADIENT_ACCUMULATION_STEPS does not match the resolved config"
            )
    expected_implementation = environment.get(
        "TACDINO_GRADIENT_ACCUMULATION_IMPLEMENTATION"
    )
    if (
        expected_implementation is not None
        and expected_implementation != implementation
    ):
        raise ValueError(
            "TACDINO_GRADIENT_ACCUMULATION_IMPLEMENTATION does not match the "
            "resolved config"
        )
    expected_loss_batch_scope = environment.get("TACDINO_LOSS_BATCH_SCOPE")
    if (
        expected_loss_batch_scope is not None
        and expected_loss_batch_scope != loss_batch_scope
    ):
        raise ValueError(
            "TACDINO_LOSS_BATCH_SCOPE does not match the resolved config"
        )
    return configured


def apply_scaling_rules_to_cfg(cfg):  # to fix
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        accumulation_steps = resolve_gradient_accumulation_steps(cfg)
        forward_per_rank = int(cfg.train.batch_size_per_gpu)
        world_size = int(distributed.get_global_size())
        forward_global_batch = forward_per_rank * world_size
        loss_global_batch = forward_global_batch
        effective_optimizer_batch = forward_global_batch * accumulation_steps
        if loss_global_batch < 1 or effective_optimizer_batch < 1:
            raise ValueError("loss and optimizer batch sizes must be positive")
        batch_assertions = {
            "TACDINO_MICROBATCH_PER_GPU": int(cfg.train.batch_size_per_gpu),
            "TACDINO_LOSS_GLOBAL_BATCH": loss_global_batch,
            "TACDINO_EFFECTIVE_OPTIMIZER_BATCH": effective_optimizer_batch,
            "TACDINO_FORWARD_BATCH_PER_GPU": forward_per_rank,
        }
        for variable, actual in batch_assertions.items():
            expected = os.environ.get(variable)
            if expected is None:
                continue
            if not str(expected).isdigit() or int(expected) < 1:
                raise ValueError(f"{variable} must be a positive integer")
            if int(expected) != actual:
                raise ValueError(
                    f"{variable} does not match the resolved distributed config"
                )
        cfg.train.forward_global_batch = forward_global_batch
        cfg.train.loss_global_batch = loss_global_batch
        cfg.train.effective_optimizer_batch = effective_optimizer_batch
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(effective_optimizer_batch / 1024.0)
        logger.info(
            "sqrt scaling learning rate; base: %s, loss_batch_scope=%s, "
            "loss_global_batch=%d, effective_optimizer_batch=%d, "
            "accumulation_steps=%d, new: %s",
            base_lr,
            str(getattr(cfg.train, "loss_batch_scope", LOSS_BATCH_SCOPE)),
            loss_global_batch,
            effective_optimizer_batch,
            accumulation_steps,
            cfg.optim.lr,
        )
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def default_setup(args, seed=0):
    distributed.enable(overwrite=True)
    seed = int(seed)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    # The resolved training config is the single seed source.  In particular,
    # CLI overrides such as ``train.seed=123`` must affect Python, NumPy,
    # PyTorch, CUDA and the training sampler in the same run.
    default_setup(args, seed=cfg.train.seed)
    # Scaling also validates the explicit config/environment accumulation
    # contract before the resolved config is persisted.
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg
