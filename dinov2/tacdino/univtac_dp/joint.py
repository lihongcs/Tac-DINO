"""End-to-end GeoTAC-initialized Diffusion Policy training on raw observations."""
import contextlib
import copy
import datetime
import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler

from .data import (LEGACY_INPUT_PROTOCOL, causal_indices, episode_split,
                   hdf5_image_chw, identity, input_protocol, sha_file,
                   tactile_image_field)
from .model import FrozenFeatureDP


def temporal_segments(steps):
    """Some official demos rewind simulation time during an expert retry.

    Preserve samples but never supervise or observe across those discontinuities.
    Single-frame segments cannot define an action target and have no windows.
    """
    cuts = [0] + (np.flatnonzero(np.diff(steps) <= 0) + 1).tolist() + [len(steps)]
    return [(start, stop) for start, stop in zip(cuts, cuts[1:]) if stop-start >= 2]


class RawDataset(Dataset):
    """Load compressed RGB streams only, never full depth/marker arrays.

    Workers decode sampled windows; no fixed DINO feature cache is used.
    """
    def __init__(self, cfg, task, split, normalizer=None, episode_ids=None):
        self.cfg, self.task, self.split = cfg, task, split
        self.input_protocol = input_protocol(cfg)
        self.episodes, self.index = [], []
        root = Path(cfg["data_root"])
        ids = episode_split(cfg)[split] if episode_ids is None else episode_ids
        self.source_manifest = []
        self.temporal_resets = []
        for episode in ids:
            rel = f"isaac45/{task}/hdf5/{episode}.hdf5"
            trajectory = root / rel
            receipt_path = root / "_download_receipts" / (rel + ".json")
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                if not receipt.get("verified") or trajectory.stat().st_size != receipt["bytes"]:
                    raise ValueError(f"Missing/invalid verified trajectory: {rel}")
            else:
                receipt = dict(path=rel, bytes=trajectory.stat().st_size, sha256=sha_file(trajectory))
            self.source_manifest.append(receipt)
            with h5py.File(root / rel, "r") as source:
                ep = {"qpos": source["embodiment/joint"][:, :cfg["action_dim"]].astype(np.float32)}
                steps = source["step"][:]
                if not np.isfinite(ep["qpos"]).all() or len(steps) != len(ep["qpos"]) or not np.isfinite(steps).all():
                    raise ValueError(f"Invalid state/time sequence: {rel}")
                ep["vision"] = [source[f"observation/{cam}/rgb"][:]
                                for cam in cfg["cameras"][task]]
                tactile_field = tactile_image_field(cfg)
                ep["tactile"] = [source[f"tactile/{sensor}/{tactile_field}"][:]
                                 for sensor in cfg["tactile_sensors"]]
                for modality in ("vision", "tactile"):
                    if modality in ep and any(len(x) != len(steps) for x in ep[modality]):
                        raise ValueError("Sensor/state frame counts disagree")
                boundaries = (np.flatnonzero(np.diff(steps) <= 0) + 1).tolist()
                if boundaries:
                    self.temporal_resets.append(dict(episode=episode, boundaries=boundaries))
                for start, stop in temporal_segments(steps):
                    segment = {"qpos": ep["qpos"][start:stop]}
                    for modality in ("vision", "tactile"):
                        if modality in ep:
                            segment[modality] = [stream[start:stop] for stream in ep[modality]]
                    number = len(self.episodes)
                    self.index.extend((number, t) for t in range(stop-start-1))
                    self.episodes.append(segment)
        if not self.index:
            raise ValueError("No causal windows")
        if normalizer is None:
            if split != "train":
                raise ValueError("VAL must receive TRAIN-only normalization")
            self.normalizer = {}
            for name, sl in (("state", slice(None, -1)), ("action", slice(1, None))):
                x = np.concatenate([ep["qpos"][sl] for ep in self.episodes])
                self.normalizer[name] = dict(mean=x.mean(0).tolist(), std=np.maximum(x.std(0), .01).tolist())
        else:
            self.normalizer = normalizer

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, t = self.index[int(index)]
        ep = self.episodes[episode]
        obs, future, valid = causal_indices(t, len(ep["qpos"]),
            self.cfg["observation_steps"], self.cfg["prediction_horizon"])
        result = {"valid": torch.from_numpy(valid.astype(np.float32))}
        for name, indices in (("state", obs), ("action", future)):
            stats = self.normalizer[name]
            x = (ep["qpos"][indices] - np.array(stats["mean"], np.float32)) / np.array(stats["std"], np.float32)
            result[name] = torch.from_numpy(x)
        for modality in ("vision", "tactile"):
            if modality not in ep:
                continue
            frames = np.stack([np.stack([hdf5_image_chw(
                                        stream[int(i)], self.cfg["image_size"], self.input_protocol)
                                        for stream in ep[modality]]) for i in obs])
            result[modality] = torch.from_numpy(frames)
        return result


class StepSampler(Sampler):
    """Replayable batches independent of prefetch/worker state and restarts."""
    def __init__(self, length, cfg, rank, accumulation, first=0):
        self.length, self.cfg, self.rank = length, cfg, rank
        self.accumulation, self.first = accumulation, first

    def __iter__(self):
        for step in range(self.first + 1, self.cfg["training_steps"] + 1):
            for micro in range(self.accumulation):
                rng = np.random.default_rng(np.random.SeedSequence([self.cfg["seed"], step, micro, self.rank]))
                yield rng.integers(self.length, size=self.cfg["micro_batch_size"]).tolist()

    def __len__(self):
        return (self.cfg["training_steps"] - self.first) * self.accumulation


class ImageDP(nn.Module):
    def __init__(self, cfg, device, encoder_factory=None, checkpoint_state=None):
        super().__init__()
        self.cfg = cfg
        if cfg["freeze_backbones"]:
            raise ValueError("ImageDP is the explicitly unfrozen experiment")
        if encoder_factory is None:
            from .geotac import load_pair
            self.vision, self.tactile, self.encoder_provenance = load_pair(
                cfg, device, initialize=checkpoint_state is None)
        else:
            self.vision, self.tactile = encoder_factory(), encoder_factory()
        mean, std = (.485, .456, .406), (.229, .224, .225)
        # Same DP-head initialization despite one vs two backbone constructions.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg["seed"])
            self.policy = FrozenFeatureDP(cfg)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        if checkpoint_state is not None:
            self.load_state_dict(checkpoint_state, strict=True, assign=True)
        self.to(device)
        # Check storage only after checkpoint assignment and materialization.
        # Meta tensors all report data_ptr()==0, so checking them earlier falsely
        # rejects a valid two-backbone checkpoint during rollout loading.
        if self.tactile is not None:
            if next(self.vision.parameters()).data_ptr() == next(self.tactile.parameters()).data_ptr():
                raise ValueError("Modality encoders must have independent parameter storage")

    def encode_batch(self, batch):
        features = []
        b, t = batch["vision"].shape[:2]
        for modality in ("vision", "tactile"):
            encoder = getattr(self, modality)
            frames = batch[modality]
            views = frames.shape[2]
            x = frames.flatten(0, 2).float().div(255.)
            x = (x - self.mean) / self.std
            z = encoder.forward_features(x)["x_norm_clstoken"].float()
            if z.shape[-1] != self.cfg["feature_dim"]:
                raise ValueError("encoder output width changed from the registered DP configuration")
            z = z.reshape(b, t, views, -1)
            if views == 1:
                z = torch.cat([z, torch.zeros_like(z)], dim=2)
            features.append(z)
        return dict(batch, features=torch.cat(features, dim=2))

    def forward(self, batch, generator):
        return self.policy.loss(self.encode_batch(batch), generator)

    @torch.no_grad()
    def predict(self, batch, generator):
        return self.policy.predict(self.encode_batch(batch), generator)


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def setup():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    if world > 1:
        torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(minutes=15))
    return rank, world, torch.device(f"cuda:{local}")


def grad_report(model):
    result = {}
    for name in ("vision", "tactile", "policy"):
        module = getattr(model, name)
        if module is None:
            result[name] = None
            continue
        grads = [p.grad.detach().float().norm().square() for p in module.parameters() if p.grad is not None]
        result[name] = float(torch.stack(grads).sum().sqrt()) if grads else 0.
    return result


def witnesses(model):
    return {name: next(p for p in module.parameters() if p.requires_grad)
            for name in ("vision", "tactile", "policy")
            if (module := getattr(model, name)) is not None}


def update_ema(ema, model, decay):
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.lerp_(source, 1 - decay)


@torch.no_grad()
def validate(model, data, cfg, device, rank, world):
    model.eval()
    ids = np.random.default_rng(cfg["seed"] + 1000).choice(len(data),
        min(cfg["validation_windows"], len(data)), replace=False)[rank::world]
    loader = DataLoader(torch.utils.data.Subset(data, ids.tolist()),
        batch_size=8, num_workers=0)  # Keep validation noise grouping unchanged across batch migration.
    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 2000 + rank)
    sums = torch.zeros(2, device=device, dtype=torch.float64)
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch, gen)
        weight = batch["valid"].sum()
        sums += torch.stack([loss.double() * weight, weight.double()])
    if world > 1:
        torch.distributed.all_reduce(sums)
    return float(sums[0] / sums[1])


def normalized_copy_current(batch, normalizer, device):
    """Express the last observed qpos in the action normalizer's space."""
    state_mean = torch.as_tensor(normalizer["state"]["mean"], device=device)
    state_std = torch.as_tensor(normalizer["state"]["std"], device=device)
    action_mean = torch.as_tensor(normalizer["action"]["mean"], device=device)
    action_std = torch.as_tensor(normalizer["action"]["std"], device=device)
    current = batch["state"][:, -1] * state_std + state_mean
    return (current - action_mean) / action_std


@torch.no_grad()
def validate_sampled_actions(model, data, cfg, device, rank, world):
    """Measure executable sampled actions, not only diffusion noise prediction."""
    model.eval()
    ids = np.random.default_rng(cfg["seed"] + 3000).choice(
        len(data), min(32, len(data)), replace=False,
    )[rank::world]
    loader = DataLoader(torch.utils.data.Subset(data, ids.tolist()),
                        batch_size=4, num_workers=0)
    # sampled SSE, copy-current SSE, valid scalar count
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 4000 + rank)
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction = model.predict(batch, generator).float()
        mask = batch["valid"].unsqueeze(-1)
        target = batch["action"]
        copied = normalized_copy_current(batch, data.normalizer, device)
        copied = copied[:, None].expand_as(target)
        count = mask.sum() * cfg["action_dim"]
        sums[0] += (((prediction - target).square() * mask).sum()).double()
        sums[1] += (((copied - target).square() * mask).sum()).double()
        sums[2] += count.double()
    if world > 1:
        torch.distributed.all_reduce(sums)
    sampled = float(sums[0] / sums[2].clamp_min(1))
    copied = float(sums[1] / sums[2].clamp_min(1))
    return {
        "val_sampled_action_mse": sampled,
        "val_copy_current_mse": copied,
        "val_sampled_over_copy_ratio": sampled / max(copied, 1e-12),
        "val_sampled_windows": min(32, len(data)),
    }


def preflight(args, cfg):
    from .cli import atomic_json, code_sha, require_download_complete, seed_all
    rank, world, device = setup()
    if world != 1:
        raise ValueError("Run standalone preflight per experiment")
    require_download_complete(cfg, args.task)
    seed_all(cfg["seed"])
    data = RawDataset(cfg, args.task, "train", episode_ids=[0])
    batch = move_batch(next(iter(DataLoader(data, batch_size=cfg["micro_batch_size"]))), device)
    model = ImageDP(cfg, device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()  # Include formal EMA memory in capacity check.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"], betas=tuple(cfg["optimizer_betas"]), weight_decay=cfg["weight_decay"])
    initial = {k: p.detach().clone() for k, p in witnesses(model).items()}
    losses, gradients = [], []
    for step in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        gen = torch.Generator(device=device).manual_seed(42 + step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch, gen)
        loss.backward()
        grads = grad_report(model)
        if not torch.isfinite(loss) or any(v is not None and (not math.isfinite(v) or v <= 0) for v in grads.values()):
            raise FloatingPointError(f"Invalid encoder/DP gradients: {grads}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss))
        gradients.append(grads)
    deltas = {k: float((p.detach()-initial[k]).abs().max()) for k, p in witnesses(model).items()}
    if not all(v > 0 for v in deltas.values()):
        raise ValueError("An encoder/DP parameter did not update")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.predict(batch, torch.Generator(device=device).manual_seed(42))
        original = model.policy.condition(model.encode_batch(batch))
        changed = dict(batch)
        changed["tactile"] = torch.zeros_like(batch.get("tactile", batch["vision"]))
        perturbed = model.policy.condition(model.encode_batch(changed))
        sensitivity = float((original - perturbed).abs().max())
    if not torch.isfinite(prediction).all() or tuple(prediction.shape) != (cfg["micro_batch_size"], cfg["prediction_horizon"], cfg["action_dim"]):
        raise ValueError("Non-finite or incorrectly shaped sampled actions")
    if sensitivity <= 0:
        raise ValueError("Modality input ablation failed")
    report = dict(commit=code_sha(), task=args.task, experiment=cfg["experiment"],
        checkpoint_sha256=sha_file(cfg["checkpoint"]), freeze_backbones=False,
        encoder_family=cfg.get("encoder_family", "dinov2_vitl14"),
        encoder_provenance=model.encoder_provenance,
        losses=losses, gradients=gradients, parameter_max_updates=deltas,
        micro_batch_size=cfg["micro_batch_size"], includes_ema_memory=True,
        tactile_sensitivity=sensitivity, prediction_shape=list(prediction.shape),
        gpu_peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
        passed=True, formal_training_started=False, rollout_evaluated=False)
    atomic_json(Path(cfg["output_root"])/"preflight"/f"{args.task}_{cfg['experiment']}_joint.json", report)
    print(json.dumps(report), flush=True)


def check_batch_migration(old, new):
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    if not changed or not changed <= {"batch_size", "micro_batch_size"}:
        raise ValueError(f"Only explicit batch changes are allowed: {sorted(changed)}")
    return sorted(changed)


def check_budget_extension(old, new):
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    compatibility = set()
    if ("input_protocol" in changed and old.get("input_protocol") is None
            and new.get("input_protocol") == LEGACY_INPUT_PROTOCOL):
        compatibility.add("input_protocol")
    if ("live_tactile_sensors" in changed
            and old.get("live_tactile_sensors") is None
            and new.get("live_tactile_sensors") == ["left_tactile", "right_tactile"]):
        compatibility.add("live_tactile_sensors")
    if (changed - compatibility != {"training_steps"}
            or new["training_steps"] <= old["training_steps"]):
        raise ValueError("Training extension must only increase training_steps")
    return sorted(changed)


def check_budget_revision(old, new, completed):
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    if changed != {"training_steps"} or new["training_steps"] <= completed:
        raise ValueError("Budget revision must only change training_steps beyond the saved step")
    return sorted(changed)


def train(args, cfg):
    from .cli import atomic_json, code_sha, emit, require_download_complete, seed_all
    rank, world, device = setup()
    require_download_complete(cfg, args.task)
    commit = code_sha()
    seed_all(cfg["seed"])
    denominator = world * cfg["micro_batch_size"]
    if cfg["batch_size"] % denominator:
        raise ValueError("Effective batch must equal micro batch * world size * accumulation")
    accumulation = cfg["batch_size"] // denominator
    out = Path(cfg["output_root"])/args.task/cfg["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    # Hold a lock for the full run to prevent accidental duplicate launch/resume.
    lock = None
    if rank == 0:
        import fcntl
        lock = open(out/"train.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/"latest.pt").exists() and not args.resume:
        raise ValueError("Existing checkpoint: explicit --resume required")
    if args.resume and not (out/"latest.pt").is_file():
        raise FileNotFoundError(f"No checkpoint to resume at {out / 'latest.pt'}")
    if rank == 0:
        print(json.dumps(dict(stage="loading_data", task=args.task, experiment=cfg["experiment"])), flush=True)
    train_data = RawDataset(cfg, args.task, "train")
    val_data = RawDataset(cfg, args.task, "val", normalizer=train_data.normalizer)
    checkpoint_sha = sha_file(cfg["checkpoint"])
    run_identity = identity(dict(config=cfg, task=args.task, split=episode_split(cfg),
        normalizer=train_data.normalizer, checkpoint_sha=checkpoint_sha,
        sources=train_data.source_manifest + val_data.source_manifest))
    model = ImageDP(cfg, device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"], betas=tuple(cfg["optimizer_betas"]), weight_decay=cfg["weight_decay"])
    first, best = 0, float("inf")
    resume_lineage = None
    if args.resume:
        saved = torch.load(out/"latest.pt", map_location="cpu", weights_only=False, mmap=True)
        if saved["run_identity"] != run_identity:
            if args.revise_training_budget:
                changed = check_budget_revision(saved["config"], cfg, saved["step"])
            elif args.extend_training:
                changed = check_budget_extension(saved["config"], cfg)
            elif args.allow_batch_change:
                changed = check_batch_migration(saved["config"], cfg)
            else:
                raise ValueError("Resume data/configuration identity mismatch")
            old_identity = identity(dict(config=saved["config"], task=args.task,
                split=episode_split(cfg), normalizer=train_data.normalizer,
                checkpoint_sha=checkpoint_sha,
                sources=train_data.source_manifest + val_data.source_manifest))
            if old_identity != saved["run_identity"]:
                raise ValueError("Migration cannot change data or initialization identity")
            resume_lineage = dict(step=saved["step"], parent_identity=old_identity,
                parent_commit=saved["commit"], changed_keys=changed,
                old_config=saved["config"], new_config=cfg)
            if rank == 0:
                kind = "budget_revision" if args.revise_training_budget else "budget_extension" if args.extend_training else "batch_migration"
                atomic_json(out/f"{kind}_{saved['step']}.json", resume_lineage)
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        first, best = saved["step"], saved["best_val"]
        del saved
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False,
                  gradient_as_bucket_view=True) if world > 1 else model
    sampler = StepSampler(len(train_data), cfg, rank, accumulation, first)
    loader = DataLoader(train_data, batch_sampler=sampler, num_workers=cfg["data_workers"],
        pin_memory=True, persistent_workers=cfg["data_workers"] > 0,
        **({"multiprocessing_context": "spawn", "prefetch_factor": 2} if cfg["data_workers"] else {}))
    batches = iter(loader)
    if rank == 0:
        atomic_json(out/"run_config.json", dict(config=cfg, task=args.task, commit=commit,
            run_identity=run_identity, checkpoint_sha256=checkpoint_sha,
            resumed_from_step=first, resume_lineage=resume_lineage,
            split=episode_split(cfg), normalizer=train_data.normalizer,
            world_size=world, gradient_accumulation=accumulation, effective_batch=cfg["batch_size"],
            train_windows=len(train_data), val_windows=len(val_data), freeze_encoder=False,
            encoder_family=cfg.get("encoder_family", "dinov2_vitl14"),
            encoder_provenance=model.encoder_provenance,
            temporal_resets={"train": train_data.temporal_resets, "val": val_data.temporal_resets},
            trainable_parameters={name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                for name in ("vision", "tactile", "policy") if (module := getattr(model, name)) is not None},
            auxiliary_losses=[], rollout_evaluated=False))
    start = time.time()
    for step in range(first + 1, cfg["training_steps"] + 1):
        model.train()
        phase = max(0., (step-cfg["warmup_steps"])/max(1, cfg["training_steps"]-cfg["warmup_steps"]))
        lr = cfg["learning_rate"] * min(step/cfg["warmup_steps"], 1.) * .5 * (1+math.cos(math.pi*phase))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=device)
        before = {k: p.detach().clone() for k, p in witnesses(model).items()} if step == first+1 else None
        # Future padding varies across windows. Weight each micro-batch by its
        # actual valid target count, so DDP accumulation equals a global masked MSE.
        chunks = [next(batches) for _ in range(accumulation)]
        global_valid = torch.tensor(sum(float(x["valid"].sum()) for x in chunks), device=device)
        if world > 1:
            torch.distributed.all_reduce(global_valid)
        for micro, cpu_batch in enumerate(chunks):
            batch = move_batch(cpu_batch, device)
            sync = wrapped.no_sync() if world > 1 and micro < accumulation-1 else contextlib.nullcontext()
            gen = torch.Generator(device=device).manual_seed(cfg["seed"] + step*10000 + rank*100 + micro)
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = wrapped(batch, gen)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {step}")
                    weight = batch["valid"].sum() * world / global_valid
                    scaled = loss * weight
                scaled.backward()
            loss_total += scaled.detach()
        grads = grad_report(model)
        if any(v is not None and not math.isfinite(v) for v in grads.values()):
            raise FloatingPointError(f"Non-finite gradient at step {step}: {grads}")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        deltas = {k: float((p.detach()-before[k]).abs().max()) for k, p in witnesses(model).items()} if before else None
        if lr > 0 and deltas and not all(v > 0 for v in deltas.values()):
            raise ValueError(f"Encoder/DP failed to update: {deltas}")
        # Early EMA warmup avoids keeping the initial network for 1000 updates.
        decay = min(cfg["ema_decay"], 1 - (1 + step)**(-.75))
        update_ema(ema, model, decay)
        if world > 1:
            torch.distributed.all_reduce(loss_total)
            loss_total /= world
        if rank == 0 and (step == first + 1 or step <= 3 or step % 10 == 0):
            emit(out/"metrics.jsonl", dict(step=step, total_steps=cfg["training_steps"],
                train_noise_mse=float(loss_total), grad_norm=float(norm), module_grad_norms=grads,
                first_step_parameter_updates=deltas, lr=lr, ema_decay=decay,
                elapsed_seconds=time.time()-start, seconds_per_update=(time.time()-start)/(step-first),
                gpu_peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
                freeze_encoder=False, skipped_optimizer_steps=0))
        if step == cfg["initial_validation_step"] or step % cfg["validation_every"] == 0 or step == cfg["training_steps"]:
            val = validate(ema, val_data, cfg, device, rank, world)
            if not math.isfinite(val):
                raise FloatingPointError("Non-finite validation")
            action_metrics = {}
            if (step == cfg["initial_validation_step"] or step % 1000 == 0
                    or step == cfg["training_steps"]):
                action_metrics = validate_sampled_actions(
                    ema, val_data, cfg, device, rank, world,
                )
                if not all(math.isfinite(value) for key, value in action_metrics.items()
                           if key != "val_sampled_windows"):
                    raise FloatingPointError("Non-finite sampled-action validation")
            improved = val < best
            best = min(best, val)
            if rank == 0:
                emit(out/"metrics.jsonl", dict(step=step, val_noise_mse=val, best_val_noise_mse=best,
                    rollout_evaluated=False, **action_metrics))
                state = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, best_val=best, config=cfg, task=args.task, normalizer=train_data.normalizer,
                    run_identity=run_identity, commit=commit, checkpoint_sha256=checkpoint_sha,
                    freeze_encoder=False)
                for path in ([out/"best.pt", out/"latest.pt"] if improved else [out/"latest.pt"]):
                    tmp = path.with_suffix(".pt.tmp")
                    torch.save(state, tmp)
                    tmp.replace(path)
                atomic_json(out/"checkpoint_status.json", dict(step=step, best_val=best,
                    latest="latest.pt", best="best.pt", can_resume=True, rollout_evaluated=False))
            if world > 1:
                torch.distributed.barrier()
    if rank == 0:
        atomic_json(out/"complete.json", dict(step=cfg["training_steps"], best_val_noise_mse=best,
            run_identity=run_identity, rollout_evaluated=False))
    if world > 1:
        torch.distributed.destroy_process_group()
