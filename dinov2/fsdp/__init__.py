# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified for Tac-DINO.

import os
from typing import Any

import torch
import dinov2.distributed as distributed
from functools import partial
from fvcore.common.checkpoint import Checkpointer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp._runtime_utils import _reshard

from torch.distributed.fsdp import StateDictType, FullStateDictConfig


def get_fsdp_wrapper(model_cfg, modules_to_wrap=set()):
    sharding_strategy_dict = {
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    }

    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    mixed_precision_config = MixedPrecision(
        param_dtype=dtype_dict[model_cfg.mixed_precision.param_dtype],
        reduce_dtype=dtype_dict[model_cfg.mixed_precision.reduce_dtype],
        buffer_dtype=dtype_dict[model_cfg.mixed_precision.buffer_dtype],
    )

    sharding_strategy_config = sharding_strategy_dict[model_cfg.sharding_strategy]

    local_rank = distributed.get_local_rank()

    fsdp_wrapper = partial(
        FSDP,
        sharding_strategy=sharding_strategy_config,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=ModuleWrapPolicy(modules_to_wrap),
    )
    return fsdp_wrapper


def is_fsdp(x):
    return isinstance(x, FSDP)


def is_sharded_fsdp(x):
    return is_fsdp(x) and x.sharding_strategy is not ShardingStrategy.NO_SHARD


def free_if_fsdp(x):
    if not is_sharded_fsdp(x):
        return
    # PyTorch 2.1+ replaced the list API with a single optional handle.
    # Silently skipping it leaves SHARD_GRAD_OP's full-precision / mixed-
    # precision all-gather cache alive after a no-grad teacher forward. EMA
    # then updates the local master shard but the next forward can reuse the
    # OLD unsharded weights, until checkpoint export/restart refreshes them.
    if hasattr(x, "_handle"):
        handle = x._handle
        if handle is not None and hasattr(handle.flat_param, "_local_shard"):
            _reshard(x, handle, True)
    elif hasattr(x, "_handles"):
        handles = [h for h in x._handles if hasattr(h.flat_param, "_local_shard")]
        if handles:
            _reshard(x, handles, [True for _ in handles])
    else:
        raise RuntimeError("unsupported FSDP handle API; refusing stale teacher weights")


def get_fsdp_modules(x):
    return FSDP.fsdp_modules(x)


def reshard_fsdp_model(x):
    for m in get_fsdp_modules(x):
        free_if_fsdp(m)


def rankstr():
    return f"rank_{distributed.get_global_rank()}"


def _all_ranks_succeeded(local_succeeded: bool) -> bool:
    """Return true only when every distributed rank reports success."""

    if not distributed.is_enabled():
        return bool(local_succeeded)
    backend_name = str(torch.distributed.get_backend()).lower()
    device = (
        torch.device("cuda", distributed.get_local_rank())
        if "nccl" in backend_name
        else torch.device("cpu")
    )
    succeeded = torch.tensor(int(local_succeeded), device=device, dtype=torch.int32)
    torch.distributed.all_reduce(succeeded, op=torch.distributed.ReduceOp.MIN)
    return bool(succeeded.item())


def _remove_if_present(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        # Preserve the original checkpoint failure. A stale uniquely named
        # temporary file is never considered by resume_or_load.
        pass


class FSDPCheckpointer(Checkpointer):
    
    def _load_file(self, f: str) -> object:
        """
        Load a checkpoint file.
        Args:
            f (str): a local path or a path handled by :class:`PathManager`.
        """
        with self.path_manager.open(f, "rb") as file:
            try:
                return torch.load(file, map_location=torch.device("cpu"), weights_only=False)
            except TypeError:
                return torch.load(file, map_location=torch.device("cpu"))
    
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return
        data = {}
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.{rankstr()}.pth"
        save_file = os.path.join(self.save_dir, basename)
        temporary_file = f"{save_file}.tmp.{os.getpid()}"
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        write_error = None
        try:
            with self.path_manager.open(temporary_file, "wb") as f:
                torch.save(data, f)
        except Exception as error:
            write_error = error

        if not _all_ranks_succeeded(write_error is None):
            _remove_if_present(temporary_file)
            if write_error is not None:
                raise write_error
            raise RuntimeError(
                "another distributed rank failed to write its checkpoint shard"
            )

        replace_error = None
        try:
            os.replace(temporary_file, save_file)
        except Exception as error:
            replace_error = error

        if not _all_ranks_succeeded(replace_error is None):
            _remove_if_present(temporary_file)
            if replace_error is not None:
                raise replace_error
            raise RuntimeError(
                "another distributed rank failed to commit its checkpoint shard"
            )

        # Every rank has now atomically committed its distinct LOCAL_STATE_DICT
        # shard. Only this globally successful generation may advance the tags.
        self.tag_last_checkpoint(basename)

    def load(self, *args, **kwargs):
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        # pyre-fixme[6]: For 2nd param expected `Union[PathLike[str], str]` but got
        #  `Union[bytes, str]`.
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        temporary_file = f"{save_file}.tmp.{os.getpid()}"
        rollback_file = f"{save_file}.rollback.{os.getpid()}"
        previous_exists = os.path.exists(save_file)
        previous_value = None
        stage_error = None
        try:
            if previous_exists:
                with self.path_manager.open(save_file, "r") as f:
                    previous_value = f.read()
            with self.path_manager.open(temporary_file, "w") as f:
                f.write(last_filename_basename)  # pyre-ignore
        except Exception as error:
            stage_error = error

        if not _all_ranks_succeeded(stage_error is None):
            _remove_if_present(temporary_file)
            if stage_error is not None:
                raise stage_error
            raise RuntimeError(
                "another distributed rank failed to stage its last-checkpoint tag"
            )

        replace_error = None
        try:
            os.replace(temporary_file, save_file)
        except Exception as error:
            replace_error = error

        if _all_ranks_succeeded(replace_error is None):
            return

        # A tag-generation commit spans one file per rank. Roll back every rank
        # to its exact prior tag when any atomic replace fails, so a failed save
        # cannot leave a mixed generation visible to resume_or_load.
        rollback_error = None
        try:
            _remove_if_present(temporary_file)
            if previous_exists:
                with self.path_manager.open(rollback_file, "w") as f:
                    f.write(previous_value)  # pyre-ignore
                os.replace(rollback_file, save_file)
            else:
                _remove_if_present(save_file)
        except Exception as error:
            rollback_error = error
        rollback_succeeded = _all_ranks_succeeded(rollback_error is None)
        _remove_if_present(rollback_file)
        if not rollback_succeeded:
            raise RuntimeError(
                "distributed last-checkpoint tag rollback failed"
            ) from rollback_error
        if replace_error is not None:
            raise replace_error
        raise RuntimeError(
            "another distributed rank failed to commit its last-checkpoint tag"
        )


ShardedGradScaler = ShardedGradScaler
