"""Observability only: detached objective accounting and sampled update checks.

No new objective, randomness, optimizer operation, checkpoint, or stop gate.
Sampled parameter statistics are explicitly NOT full-parameter norm estimates.
"""
import json
import math
from pathlib import Path

import torch

SCHEMA = "tacdino.training-diagnostics.v1"


def objective_ledger(logs, cfg, objective):
    """Reconstruct the weighted GeoTAC objective from its recorded terms."""
    total = objective.detach().float().reshape(())
    entries = {}

    def add(name, raw, weight):
        raw = raw.detach().float().reshape(())
        entries[name] = (raw, float(weight), raw * float(weight))
    for name in ("dino_local_crops_loss", "dino_global_crops_loss"):
        if name in logs:
            add(name, logs[name], float(cfg.dino.loss_weight))
    if "ibot_loss" in logs:
        add("ibot_loss", logs["ibot_loss"] * 2, float(cfg.ibot.loss_weight))
    if "koleo_loss" in logs:
        weight = float(cfg.dino.koleo_loss_weight)
        add("koleo_loss", logs["koleo_loss"] * (2 / weight) if weight else logs["koleo_loss"] * 0,
            weight)
    if "multiresource_alignment_loss" in logs:
        from dinov2.layers.local_four_modality import PAIRS
        names = [f"align_{left}_{right}" for left, right in PAIRS]
        present = [name for name in names if name in logs]
        for name in present:
            add(name, logs[name], float(cfg.touch.multiresource_weight) / len(present))
    if "extra_cls_vt_loss" in logs:
        add("extra_cls_vt_loss", logs["extra_cls_vt_loss"], 1.)
    if "relation_distillation_loss" in logs:
        add("relation_distillation_loss", logs["relation_distillation_loss"],
            float(cfg.touch.get("relation_distillation_weight", 0.)))
    reconstructed = sum((v[2] for v in entries.values()), torch.zeros_like(total))
    supported = str(cfg.train.local_data) == "tac_patchalign"
    result = {}
    for name, (raw, weight, contribution) in entries.items():
        result.update({"loss_raw/"+name: raw, "loss_weight/"+name: total.new_tensor(weight),
                       "loss_objective/"+name: contribution})
    result.update(loss_objective_sum=reconstructed,
        loss_accounting_residual=total-reconstructed,
        loss_accounting_relative_error=(total-reconstructed).abs()/total.abs().clamp_min(1.),
        loss_accounting_supported=total.new_tensor(float(supported)),
        loss_accounting_pass=((total-reconstructed).abs() <= 1e-5*(1+total.abs())).float())
    return result


def schedule_metrics(cfg, iteration, skipped_in_process, start_iteration, batch, teacher_temp):
    epoch = int(cfg.train.OFFICIAL_EPOCH_LENGTH)
    warmup = int(cfg.optim.get("warmup_steps", cfg.optim.warmup_epochs * epoch))
    total = int(cfg.optim.get("total_steps", cfg.optim.epochs * epoch))
    return dict(optimizer_attempts_completed=iteration+1,
        successful_updates_this_process=iteration+1-start_iteration-skipped_in_process,
        samples_seen_attempted=(iteration+1)*batch,
        successful_update_count_scope_process_start=start_iteration,
        teacher_temperature=float(teacher_temp), schedule_total_steps=total,
        schedule_warmup_steps=warmup, schedule_in_warmup=float(iteration < warmup),
        schedule_warmup_fraction=min((iteration+1)/max(warmup, 1), 1.))


def diagnostic_due(iteration, period=84):
    return period > 0 and (iteration == 0 or (iteration+1) % period == 0)


def _shard(parameter):
    # Reading the local shard does not materialize or free an FSDP full cache.
    return getattr(parameter, "_local_shard", parameter).detach().reshape(-1)


def _parameter_entries(module):
    # With use_orig_params=True, teacher original-parameter views can still be
    # unsharded after a no-grad forward. Reading the master's local flat shard
    # avoids changing cache lifetime just to observe an EMA update.
    flat = {name+"._flat_param": m._handle.flat_param for name, m in module.named_modules()
            if getattr(m, "_handle", None) is not None and hasattr(m._handle, "flat_param")}
    return flat if flat else dict(module.named_parameters())


def parameter_sample_indices(count, samples=16, device="cpu"):
    """Integer-only endpoints: FP32 linspace can round count-1 UP to count."""
    if count <= 0 or samples <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    n = min(count, samples)
    return torch.arange(n, device=device, dtype=torch.long) * (count-1) // max(n-1, 1)


@torch.no_grad()
def update_snapshot(roots, teacher=None, samples=16):
    result = {}
    for name, module in roots.items():
        target = None
        if teacher is not None and name.startswith("student.") and name[8:] in teacher:
            target = _parameter_entries(teacher[name[8:]])
        rows = []
        for key, p in _parameter_entries(module).items():
            value = _shard(p)
            count = value.numel()-int(getattr(p, "_shard_numel_padded", 0))
            if count <= 0:
                continue
            ix = parameter_sample_indices(count, samples, value.device)
            q = target.get(key) if target is not None else None
            if q is not None and _shard(q).numel() != value.numel():
                q = None  # Sharding mismatch is missing coverage, not zero drift.
            rows.append((p, q, ix, value[ix].float().clone(),
                         None if q is None else _shard(q)[ix].float().clone()))
        result[name] = rows
    return result


@torch.no_grad()
def update_statistics(snapshot, momentum, reduce=True):
    output = {}
    for name, rows in snapshot.items():
        device = rows[0][0].device if rows else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sums = torch.zeros(7, device=device, dtype=torch.float64)
        for p, q, ix, old, old_teacher in rows:
            current = _shard(p)[ix].float()
            delta = current-old
            sums[:4] += torch.stack((old.square().sum(), delta.square().sum(),
                (delta != 0).sum(), delta.new_tensor(delta.numel()))).double()
            if q is not None:
                t = _shard(q)[ix].float()
                sums[4:] += torch.stack(((current-t).square().sum(),
                    (t-(old_teacher*momentum+current*(1-momentum))).square().sum(),
                    current.new_tensor(current.numel()))).double()
        if reduce and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
        weight, delta, changed, count, drift, ema_error, ema_count = sums.cpu().tolist()
        prefix = "update_sample/"+name+"/"
        output.update({prefix+"count": count, prefix+"delta_norm": math.sqrt(delta),
            prefix+"relative_delta": math.sqrt(delta/max(weight, 1e-30)),
            prefix+"changed_fraction": changed/max(count, 1),
            prefix+"EMA_count": ema_count, prefix+"student_EMA_rms": math.sqrt(drift/max(ema_count, 1)),
            prefix+"EMA_equation_rms": math.sqrt(ema_error/max(ema_count, 1))})
    return output


def capture_temporal_batch(data, output, iteration, examples=2):
    """Small PNG + metadata of ACTUAL tensors; no extra decoding or RNG calls."""
    if "resource_frames" in data:
        return capture_resource_batch(data, output, iteration, examples)
    if "paper_context_frame_ids" not in data:
        return None
    from PIL import Image
    import numpy as np
    output = Path(output)/"diagnostics"/"actual_batches"/f"step_{iteration+1}"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(min(examples, len(data["paper_context"]))):
        past = data["paper_context_frame_ids"][i].tolist()
        future = int(data["paper_target_frame_id"][i])
        if not all(a < b for a, b in zip(past, past[1:])) or past[-1] >= future:
            raise ValueError("actual history is not strictly before the target")
        parts = [data["paper_query_rgb"][i], *data["paper_context"][i], data["paper_teacher_target"][i]]
        arrays = [(x.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()*255).astype(np.uint8) for x in parts]
        name = f"sample_{i}.png"
        Image.fromarray(np.concatenate(arrays, axis=1)).save(output/name)
        records.append(dict(sample=i, history_frame_ids=past, target_frame_id=future,
            fps=float(data["paper_video_fps"][i]), frame_count=int(data["paper_video_frame_count"][i]),
            time_offsets=data["paper_context_time_offsets"][i].tolist(),
            source=data.get("paper_source_metadata", [{}]*len(data["paper_context"]))[i],
            image=name, panels="RGB | four observed tactile frames | target (teacher only)"))
    path = output/"manifest.json"
    path.write_text(json.dumps(dict(schema=SCHEMA, iteration=iteration, rank=0,
        tensor_range="raw_0_1", diagnostic_only=True, samples=records), indent=2)+"\n")
    return str(path)


def capture_resource_batch(data, output, iteration, examples=2):
    from PIL import Image
    import numpy as np
    output = Path(output)/"diagnostics"/"actual_batches"/f"step_{iteration+1}"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(min(examples, len(data["resource_tactile"]))):
        frames = data["resource_frames"][i].tolist()
        unique = int(data.get("resource_tactile_unique_frames",
            torch.full((len(data["resource_tactile"]),), len(set(frames)),
                device=data["resource_tactile"].device))[i].item())
        if unique == 1:
            if frames != [10] * len(frames):
                raise ValueError("single-image tactile input must be canonical frame 10")
        elif not all(a < b for a, b in zip(frames, frames[1:])):
            raise ValueError("resource clip frames must be ordered and distinct")
        parts = [data["resource_rgb"][i], *data["resource_tactile"][i]]
        arrays = [(x.detach().float().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8) for x in parts]
        name = f"sample_{i}.png"
        Image.fromarray(np.concatenate(arrays, 1)).save(output/name)
        records.append(dict(image=name,
            panels=("local_RGB | canonical_tactile_frame_repeated_in_head"
                    if unique == 1 else "local_RGB | full_ordered_contact_clip"),
            frame_indices=frames, frame_count=int(data["resource_frame_count"][i]),
            unique_tactile_frames=unique,
            fps=float(data["resource_fps"][i]), contact_key=int(data["resource_id"][i]),
            geometry_coverage=float(data["resource_coverage"][i])))
    path = output/"manifest.json"
    path.write_text(json.dumps(dict(schema="direct_contact_actual_batch.v1", samples=records,
        future_prediction=False), indent=2)+"\n")
    return str(path)
