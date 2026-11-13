"""Region soft matching inside contact InfoNCE; no fourth auxiliary loss.

All candidate pairs use exactly the same scorer. No absolute geometry, paired
patch indices, object IDs, or positive-only cross attention enter the score.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _mask(tokens, mask):
    if tokens.ndim != 3 or not torch.isfinite(tokens).all():
        raise ValueError('finite B x regions x width tokens required')
    if mask is None:
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool or not mask.any(1).all():
        raise ValueError('every contact requires at least one valid region')
    return mask


def region_scores(left, right, left_mask=None, right_mask=None, softness=.1, chunk_size=8):
    """Symmetric masked log-mean-exp over region similarities, then mean.

    Softness is fixed, not a trainable gate that can drop a modality. A region
    may match any valid region in the other contact. Normalizing by valid
    counts avoids larger point/patch sets receiving a cardinality bonus.
    Recompute chunk activations in backward to bound pair-token memory.
    """
    if not 0 < softness <= 1 or chunk_size < 1 or left.shape[-1] != right.shape[-1]:
        raise ValueError('invalid soft region scorer configuration')
    lm, rm = _mask(left, left_mask), _mask(right, right_mask)
    a, b = F.normalize(left.float(), dim=-1), F.normalize(right.float(), dim=-1)

    def block(x, y, xm, ym):
        sim = torch.einsum('ard,bsd->abrs', x, y)/softness
        l2r = (torch.logsumexp(sim.masked_fill(~ym[None, :, None, :], -torch.inf), -1)
            - ym.sum(-1).float().log()[None, :, None])*softness
        r2l = (torch.logsumexp(sim.masked_fill(~xm[:, None, :, None], -torch.inf), -2)
            - xm.sum(-1).float().log()[:, None, None])*softness
        return .5*((l2r*xm[:, None]).sum(-1)/xm.sum(-1)[:, None]
            + (r2l*ym[None]).sum(-1)/ym.sum(-1)[None])

    rows = []
    for i in range(0, len(a), chunk_size):
        columns = []
        for j in range(0, len(b), chunk_size):
            args = (a[i:i+chunk_size], b[j:j+chunk_size], lm[i:i+chunk_size], rm[j:j+chunk_size])
            columns.append(checkpoint(block, *args, use_reentrant=False) if torch.is_grad_enabled()
                and (a.requires_grad or b.requires_grad) else block(*args))
        rows.append(torch.cat(columns, 1))
    return torch.cat(rows, 0)


def pool_regions(tokens, mask=None):
    """Return one mask-aware descriptor for the registered contact crop.

    Unlike ``region_scores``, this representation cannot select a different
    high-similarity patch for every candidate.  It is therefore the same
    contact-level representation that can be exported for exact retrieval.
    """
    valid = _mask(tokens, mask)
    return ((tokens.float() * valid[..., None]).sum(1)
            / valid.sum(1, keepdim=True).clamp_min(1))


def pooled_scores(left, right, left_mask=None, right_mask=None):
    """Cosine scores between contact-pooled regional representations."""
    left = F.normalize(pool_regions(left, left_mask), dim=-1)
    right = F.normalize(pool_regions(right, right_mask), dim=-1)
    return left @ right.T


def symmetric_region_contact_loss(left, right, ids, left_mask=None, right_mask=None,
                                  temperature=.07, distributed=True, right_ids=None,
                                  left_valid=None, right_valid=None, metadata=None):
    """Same all-rank, multi-positive candidate pool as original contact CL."""
    if not 0 < temperature <= 1:
        raise ValueError('invalid InfoNCE temperature')
    lm, rm = _mask(left, left_mask), _mask(right, right_mask)
    ids = ids.reshape(-1)
    right_ids = ids if right_ids is None else right_ids.reshape(-1)
    if len(ids) != len(left) or len(right_ids) != len(right):
        raise ValueError('paired contact count mismatch')
    lv = torch.ones_like(ids, dtype=torch.bool) if left_valid is None else left_valid
    rv = torch.ones_like(right_ids, dtype=torch.bool) if right_valid is None else right_valid
    if lv.shape != ids.shape or rv.shape != right_ids.shape or lv.dtype != torch.bool or rv.dtype != torch.bool:
        raise ValueError('invalid observation validity mask')
    if not lv.any() or not rv.any(): raise ValueError('no valid modality observations')
    a, b, am, bm, ai, bi, av, bv = left, right, lm, rm, ids, right_ids, lv, rv
    import torch.distributed as dist
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    if world > 1:
        from torch.distributed.nn.functional import all_gather
        a, b = torch.cat(all_gather(left)), torch.cat(all_gather(right))
        def gather(value):
            values = [torch.empty_like(value) for _ in range(dist.get_world_size())]
            dist.all_gather(values, value)
            return torch.cat(values)
        am, bm, ai, bi, av, bv = [gather(v) for v in (lm, rm, ids, right_ids, lv, rv)]
    positives = [(ids[:, None] == bi[None]) & bv[None],
                 (right_ids[:, None] == ai[None]) & av[None]]
    for positive, valid in zip(positives, (lv, rv)):
        if not positive[valid].any(1).all(): raise ValueError('missing positive contact')
    logits = [region_scores(left, b, lm, bm)/temperature,
              region_scores(right, a, rm, am)/temperature]
    allowed = [torch.ones_like(x, dtype=torch.bool) for x in logits]
    if metadata is not None:
        from .contact_negative_mask import admissible_contacts
        if right_ids.shape != ids.shape or not torch.equal(ids, right_ids):
            raise ValueError('masked paired loss needs matching left/right contacts')
        objects, materials = metadata
        all_objects, all_materials = (gather(objects), gather(materials)) if world > 1 else metadata
        allowed = [admissible_contacts(ids, objects, materials, bi, all_objects, all_materials),
                   admissible_contacts(right_ids, objects, materials, ai, all_objects, all_materials)]
        for s, keep in zip(logits, allowed):
            s.masked_fill_(~keep, -torch.inf)
    loss = left.new_zeros((), dtype=torch.float32)
    for s, positive, query_valid, candidate_valid, all_query_valid in zip(
            logits, positives, (lv, rv), (bv, av), (av, bv)):
        # Remove unavailable reconstructed observations from both the query
        # mean and candidate pool, not just their positive label. Never turn
        # an unavailable tactile reconstruction into a fabricated zero view.
        selected = s[query_valid].masked_fill(~candidate_valid[None], -torch.inf)
        p = positive[query_valid]
        values = torch.logsumexp(selected, 1)-torch.logsumexp(selected.masked_fill(~p, -torch.inf), 1)
        loss = loss + .5*values.sum()*world/all_query_valid.sum()
    with torch.no_grad():
        s = logits[0][lv].masked_fill(~bv[None], -torch.inf)
        positive = positives[0][lv]
        positive_mean = s.masked_select(positive).mean()
        negative = ~positive & bv[None] & allowed[0][lv]
        negative_mean = s.masked_select(negative).mean() if negative.any() else s.new_zeros(())
        diagnostics = dict(positive_logit=positive_mean, negative_logit=negative_mean,
            logit_margin=positive_mean-negative_mean,
            positive_probability=(s.softmax(1)*positive).sum(1).mean(),
            unique_contacts=s.new_tensor(bi[bv].unique().numel()),
            positive_count=positive.sum(1).float().mean(),
            reverse_positive_count=positives[1][rv].sum(1).float().mean(),
            resource_observations=av.float().sum(), tactile_observations=bv.float().sum())
        diagnostics['ambiguous_candidates_excluded'] = (~allowed[0]).float().sum(1).mean()
        diagnostics['admissible_candidates'] = allowed[0].float().sum(1).mean()
    return loss, diagnostics


def symmetric_pooled_contact_loss(left, right, ids, left_mask=None, right_mask=None,
                                  temperature=.07, distributed=True, right_ids=None,
                                  left_valid=None, right_valid=None, metadata=None):
    """Contact InfoNCE on the same pooled descriptor used by evaluation."""
    left = pool_regions(left, left_mask)[:, None]
    right = pool_regions(right, right_mask)[:, None]
    return symmetric_region_contact_loss(
        left, right, ids, temperature=temperature, distributed=distributed,
        right_ids=right_ids, left_valid=left_valid, right_valid=right_valid, metadata=metadata)
