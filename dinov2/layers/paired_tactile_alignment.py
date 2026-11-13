"""Match external geometry to same-contact sensor geometry, not RGB tactile."""
import torch
from torch.nn import functional as F
from .contact_regions import region_scores, pooled_scores

PAIRED_TARGETS = 'paired_tactile_modalities'
SHARED_TARGET = 'shared_image_tactile'
RELATION_TEACHERS = ('depth_only', 'pointmap_only', 'depth_point_equal')
RELATION_MODES = ('full_distribution', 'noncontact_conditional', 'agreement_rank')


def dual_teacher_agreement_rank(depth_logits, point_logits, student_logits,
                                positive, temperature=.07, max_pairs=32):
    """Distil only reliable non-contact orderings shared by both teachers.

    Depth and pointmap scores are standardized per query before comparing their
    order, because the two frozen encoders have unrelated logit scales.  A
    candidate pair supervises the student only when both teachers rank it in
    the same direction.  The weight is the smaller teacher margin, attenuated
    when the two standardized teachers disagree on either endpoint.  Keeping
    only the strongest pairs prevents a nearly uniform teacher distribution
    from contributing a large number of low-confidence constraints.
    """
    if not (depth_logits.shape == point_logits.shape == student_logits.shape == positive.shape):
        raise ValueError('agreement-rank shape mismatch')
    if positive.dtype != torch.bool:
        raise ValueError('positive mask must be boolean')
    if not 0 < temperature <= 1 or max_pairs < 1:
        raise ValueError('invalid agreement-rank hyperparameters')
    candidates = ~positive
    if student_logits.shape[0] == 0 or student_logits.shape[1] == 0:
        zero = student_logits.sum(1) * 0.
        diagnostics = dict(pair_count=zero.detach(), mean_confidence=zero.detach(),
                           candidate_agreement_fraction=zero.detach())
        return zero, torch.zeros_like(zero, dtype=torch.bool), diagnostics

    def standardize(value):
        count = candidates.sum(1, keepdim=True).clamp_min(1)
        mean = (value.detach().float() * candidates).sum(1, keepdim=True) / count
        centered = (value.detach().float() - mean) * candidates
        scale = (centered.square().sum(1, keepdim=True) / count).sqrt().clamp_min(1e-6)
        return centered / scale

    depth_z, point_z = standardize(depth_logits), standardize(point_logits)
    depth_delta = depth_z[:, :, None] - depth_z[:, None, :]
    point_delta = point_z[:, :, None] - point_z[:, None, :]
    pair_mask = (candidates[:, :, None] & candidates[:, None, :]
                 & (depth_delta > 0) & (point_delta > 0))
    endpoint_agreement = torch.exp(-(depth_z-point_z).abs())
    confidence = torch.minimum(depth_delta, point_delta).clamp_min(0)
    confidence = confidence * torch.sqrt(
        endpoint_agreement[:, :, None] * endpoint_agreement[:, None, :])
    confidence = confidence.masked_fill(~pair_mask, 0.)
    flat = confidence.flatten(1)
    k = min(max_pairs, flat.shape[1])
    weights, indices = flat.topk(k, dim=1)
    selected = weights > 0
    left = indices // student_logits.shape[1]
    right = indices % student_logits.shape[1]
    student_delta = (student_logits.gather(1, left)
                     - student_logits.gather(1, right)) / temperature
    pair_loss = F.softplus(-student_delta)
    weight_sum = weights.sum(1)
    eligible = selected.any(1)
    per_query = (pair_loss * weights).sum(1) / weight_sum.clamp_min(1e-8)
    per_query = torch.where(eligible, per_query, per_query*0.)
    diagnostics = dict(
        pair_count=selected.sum(1).float(),
        mean_confidence=(weights.sum(1) / selected.sum(1).clamp_min(1)),
        candidate_agreement_fraction=(pair_mask.sum((1, 2)).float()
            / (candidates.sum(1).float().square()-candidates.sum(1).float()).clamp_min(1)))
    return per_query, eligible, diagnostics


def conditional_relation_kl(teacher_logits, student_logits, positive):
    """KL among non-matching contacts only; true-pair logits have zero gradient.

    Exclude *all* copies of a contact, not only the diagonal. Empty candidate
    sets contribute differentiable zero and are excluded from the row mean by
    the caller. This is a relation-distillation ablation, not gradient surgery.
    """
    if teacher_logits.shape != student_logits.shape or positive.shape != student_logits.shape:
        raise ValueError('conditional KL shape mismatch')
    if positive.dtype != torch.bool:
        raise ValueError('positive mask must be boolean')
    eligible = (~positive).any(1)
    # Rows without negatives get finite dummy logits before softmax. Their
    # contribution and gradient are then zero; never softmax an all--inf row.
    excluded = positive & eligible[:, None]
    t = teacher_logits.detach().float().masked_fill(excluded, -torch.inf)
    s = student_logits.float().masked_fill(excluded, -torch.inf)
    tlog, slog = F.log_softmax(t, 1), F.log_softmax(s, 1)
    probability = tlog.exp()
    delta = torch.where(excluded, torch.zeros_like(slog), tlog-slog)
    per_query = (probability * delta).sum(1)
    return torch.where(eligible, per_query, per_query*0.), eligible


def isolated_gradient_view(feature, diagnostics, name):
    """Observe one loss branch without changing its value or upstream gradient.

    Pass the returned identity view ONLY to the loss being measured. A hook on
    the original shared tensor would also include unrelated losses. The norm is
    the weighted (and, under AMP, scaled) gradient at this representation, not a
    separate backbone-parameter gradient measurement.
    """
    branch = feature * 1.
    if branch.requires_grad:
        branch.register_hook(lambda grad: diagnostics.update(
            {name: grad.detach().float().norm()}))
    return branch


def paired_geometry_contact_loss(projected, ids, valid, mask=None, distributed=True):
    """Input order is [external observations, sensor reconstructions].

    A shared modality encoder/projection has processed both halves. Neither
    absolute coordinates nor equal patch indices define correspondence. Missing
    reconstructions remove the entire geometry pair, never create self-pairs.
    """
    b = len(ids)
    if len(projected) != 2*b or valid.shape != ids.shape:
        raise ValueError('expected paired external/sensor geometry and contact validity')
    if mask is not None and mask.shape != projected.shape[:2]:
        raise ValueError('geometry region mask shape mismatch')
    if valid.dtype != torch.bool:
        raise ValueError('sensor geometry validity must be boolean')
    if mask is None:
        mask = torch.ones(projected.shape[:2], dtype=torch.bool, device=projected.device)
    left, right = projected[:b], projected[b:]
    lm, rm = mask[:b], mask[b:]
    import torch.distributed as dist
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    a, c, am, cm, all_ids, all_valid = left, right, lm, rm, ids, valid
    if world > 1:
        from torch.distributed.nn.functional import all_gather
        a, c = torch.cat(all_gather(left)), torch.cat(all_gather(right))
        def gather(value):
            values = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(values, value)
            return torch.cat(values)
        am, cm, all_ids, all_valid = [gather(v) for v in (lm, rm, ids, valid)]
    if not all_valid.any():
        raise ValueError('no certified tactile geometry pairs globally')
    positive = (ids[:, None] == all_ids[None]) & all_valid[None]
    if not positive[valid].any(1).all():
        raise ValueError('missing same-contact geometry positive')
    logits = [region_scores(left, c, lm, cm)/.07,
              region_scores(right, a, rm, am)/.07]
    loss = projected.new_zeros((), dtype=torch.float32)
    # Empty local query sets contribute differentiable zeros and still join
    # all-gather backward. Invalid pairs are never negatives in either direction.
    for score in logits:
        s = score[valid].masked_fill(~all_valid[None], -torch.inf)
        p = positive[valid]
        per_query = torch.logsumexp(s, 1)-torch.logsumexp(s.masked_fill(~p, -torch.inf), 1)
        loss = loss + .5*per_query.sum()*world/all_valid.sum()
    with torch.no_grad():
        s = logits[0][valid].masked_fill(~all_valid[None], -torch.inf)
        p = positive[valid]
        negative = ~p & all_valid[None]
        zero = s.new_zeros(())
        pos = s[p].mean() if p.any() else zero
        neg = s[negative].mean() if negative.any() else zero
        diag = dict(positive_logit=pos, negative_logit=neg, logit_margin=pos-neg,
            positive_probability=(s.softmax(1)*p).sum(1).mean() if len(s) else zero,
            unique_contacts=s.new_tensor(all_ids[all_valid].unique().numel()),
            positive_count=p.sum(1).float().mean() if len(s) else zero,
            reverse_positive_count=p.sum(1).float().mean() if len(s) else zero,
            resource_observations=all_valid.float().sum(), tactile_observations=all_valid.float().sum())
    return loss, diag


def geometry_consensus_distillation(rgb, tactile, depth, point, valid,
                                    depth_mask=None, temperature=.07,
                                    teacher='depth_point_equal',
                                    distributed=True, detach_rgb=True,
                                    relation_mode='full_distribution', contact_ids=None,
                                    score_mode='regional'):
    """Distil paired geometry relations into RGB--tactile-image matching.

    ``depth`` and ``point`` are ordered as ``[external, tactile-sensor]``.
    Their consensus distribution is a stop-gradient teacher.  RGB is also
    detached in the student score, so this objective can update only the
    tactile image representation (and its head), never move the external
    resource encoders to satisfy their own target. ``detach_rgb=False`` allows
    a student RGB projection to learn on top of an independently frozen RGB
    backbone; teacher geometry remains detached in both modes.
    """
    b = len(rgb)
    if len(tactile) != b or len(depth) != 2*b or len(point) != 2*b:
        raise ValueError('relation distillation requires one aligned six-view contact tuple')
    if valid.shape != (b,) or valid.dtype != torch.bool:
        raise ValueError('invalid tactile geometry availability mask')
    if not 0 < temperature <= 1:
        raise ValueError('invalid relation distillation temperature')
    if teacher not in RELATION_TEACHERS:
        raise ValueError('unknown geometry relation teacher')
    if relation_mode not in RELATION_MODES:
        raise ValueError('unknown relation distillation mode')
    if score_mode not in ('regional', 'contact_pooled'):
        raise ValueError('unknown relation score mode')
    if relation_mode in ('noncontact_conditional', 'agreement_rank') and contact_ids is None:
        raise ValueError('noncontact distillation requires contact IDs')
    if contact_ids is not None and contact_ids.shape != (b,):
        raise ValueError('contact IDs must match local contacts')
    if depth_mask is None:
        depth_mask = torch.ones(depth.shape[:2], dtype=torch.bool, device=depth.device)
    if depth_mask.shape != depth.shape[:2]:
        raise ValueError('depth relation mask shape mismatch')
    de, dt = depth[:b].detach(), depth[b:].detach()
    pe, pt = point[:b].detach(), point[b:].detach()
    dm_e, dm_t = depth_mask[:b], depth_mask[b:]

    import torch.distributed as dist
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1

    def gather_feature(value):
        if world == 1:
            return value
        from torch.distributed.nn.functional import all_gather
        return torch.cat(all_gather(value))

    def gather_plain(value):
        if world == 1:
            return value
        values = [torch.empty_like(value) for _ in range(world)]
        dist.all_gather(values, value)
        return torch.cat(values)

    # Geometry is already detached, while tactile candidates retain their
    # autograd all-gather path across ranks.
    student_rgb = rgb.detach() if detach_rgb else rgb
    all_rgb = gather_feature(student_rgb)
    all_tactile = gather_feature(tactile)
    all_de, all_dt = gather_feature(de), gather_feature(dt)
    all_pe, all_pt = gather_feature(pe), gather_feature(pt)
    all_dm_e, all_dm_t, all_valid = [gather_plain(x) for x in (dm_e, dm_t, valid)]
    all_ids = gather_plain(contact_ids) if contact_ids is not None else None
    if not all_valid.any():
        raise ValueError('no certified tactile geometry pairs globally')

    scorer = pooled_scores if score_mode == 'contact_pooled' else region_scores
    depth_forward = scorer(de, all_dt, dm_e, all_dm_t)
    depth_reverse = scorer(dt, all_de, dm_t, all_dm_e)
    point_forward = scorer(pe, all_pt)
    point_reverse = scorer(pt, all_pe)
    if relation_mode == 'agreement_rank' and teacher != 'depth_point_equal':
        raise ValueError('agreement-rank distillation requires both geometry teachers')
    if teacher == 'depth_only':
        teacher_forward, teacher_reverse = depth_forward, depth_reverse
    elif teacher == 'pointmap_only':
        teacher_forward, teacher_reverse = point_forward, point_reverse
    else:
        # Fixed equal weighting is part of the controlled default, not a
        # learned gate that could add another varying factor to the ablation.
        teacher_forward = .5 * (depth_forward + point_forward)
        teacher_reverse = .5 * (depth_reverse + point_reverse)
    teacher_forward = teacher_forward / temperature
    teacher_reverse = teacher_reverse / temperature
    student_forward = scorer(student_rgb, all_tactile) / temperature
    student_reverse = scorer(tactile, all_rgb) / temperature

    losses, entropies = [], []
    quality = []
    positives = ((contact_ids[valid, None] == all_ids[None, all_valid])
                 if contact_ids is not None else None)
    eligible_count = all_valid.sum()
    if relation_mode == 'noncontact_conditional':
        global_positive = all_ids[all_valid, None] == all_ids[None, all_valid]
        eligible_count = (~global_positive).any(1).sum()
    rank_diagnostics = []
    for teacher_logits, student_logits, depth_logits, point_logits in (
            (teacher_forward, student_forward, depth_forward, point_forward),
            (teacher_reverse, student_reverse, depth_reverse, point_reverse)):
        # Remove unavailable candidates before KL. Keeping them as -inf would
        # create the undefined product 0 * inf in the explicit KL expression.
        teacher_logits = teacher_logits[valid][:, all_valid]
        student_logits = student_logits[valid][:, all_valid]
        teacher_probability = F.softmax(teacher_logits, dim=1).detach()
        teacher_log_probability = torch.where(
            teacher_probability > 0, teacher_probability.log(),
            torch.zeros_like(teacher_probability))
        student_log_probability = F.log_softmax(student_logits, dim=1)
        if relation_mode == 'agreement_rank':
            depth_logits = depth_logits[valid][:, all_valid]
            point_logits = point_logits[valid][:, all_valid]
            per_query, rank_eligible, rank_diag = dual_teacher_agreement_rank(
                depth_logits, point_logits, student_logits, positives,
                temperature=temperature)
            rank_diagnostics.append(rank_diag)
        elif relation_mode == 'noncontact_conditional':
            per_query, _ = conditional_relation_kl(teacher_logits, student_logits, positives)
        else:
            per_query = (teacher_probability * (teacher_log_probability - student_log_probability)).sum(1)
        losses.append(per_query.sum() * world / eligible_count.clamp_min(1))
        entropy = -(teacher_probability * teacher_log_probability).sum(1)
        entropies.append(entropy.sum() * world / all_valid.sum())
        if positives is not None:
            with torch.no_grad():
                # Raw/full-distribution quality remains comparable in both arms.
                student_probability = student_log_probability.exp()
                normalizer = world / all_valid.sum()
                quality.append(dict(
                    teacher_positive_mass=(teacher_probability*positives).sum()*normalizer,
                    student_positive_mass=(student_probability*positives).sum()*normalizer,
                    positive_chance=(positives.float().sum(1)/all_valid.sum()).sum()*normalizer,
                    teacher_positive_top1=positives.gather(1, teacher_logits.argmax(1, keepdim=True)).float().sum()*normalizer,
                    teacher_entropy_normalized=entropy.sum()*normalizer/all_valid.sum().float().log().clamp_min(1e-8)))
    loss = .5 * (losses[0] + losses[1])
    diagnostics = dict(
        teacher_entropy=.5 * (entropies[0] + entropies[1]),
        valid_contacts=all_valid.float().sum(),
        forward_kl=losses[0].detach(), reverse_kl=losses[1].detach())
    diagnostics['conditional_noncontact'] = loss.new_tensor(float(relation_mode == 'noncontact_conditional'))
    diagnostics['agreement_rank'] = loss.new_tensor(float(relation_mode == 'agreement_rank'))
    diagnostics['contact_pooled_scores'] = loss.new_tensor(float(score_mode == 'contact_pooled'))
    diagnostics['eligible_contacts'] = eligible_count.detach().float()
    if quality:
        diagnostics.update({k: .5*(quality[0][k]+quality[1][k]) for k in quality[0]})
    if rank_diagnostics:
        for key in rank_diagnostics[0]:
            diagnostics['rank_'+key] = .5 * (
                rank_diagnostics[0][key].mean() + rank_diagnostics[1][key].mean())
    return loss, diagnostics
