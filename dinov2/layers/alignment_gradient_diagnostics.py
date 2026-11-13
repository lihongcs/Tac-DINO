"""Read-only comparison of weighted objective gradients at the same features.

No extra backward, RNG calls, optimization, or gradient surgery. Hooks return
None. Distributed statistics concatenate the rank-local tactile feature spaces;
this is NOT a cosine between full backbone-parameter gradients.
"""
import torch


class _AnchorPreservingMerge(torch.autograd.Function):
    """Merge two gradients while treating direct alignment as the anchor."""
    @staticmethod
    def forward(ctx, feature, owner):
        ctx.owner = owner
        return feature.clone(), feature.clone()

    @staticmethod
    def backward(ctx, alignment_gradient, distillation_gradient):
        if alignment_gradient is None or distillation_gradient is None:
            raise RuntimeError('anchor-preserving merge requires both objectives')
        a = alignment_gradient.float()
        d = distillation_gradient.float()
        shape = (len(a),) + (1,) * (a.ndim - 1)
        flat_a, flat_d = a.flatten(1), d.flatten(1)
        aa = flat_a.square().sum(1)
        dot = (flat_a * flat_d).sum(1)
        # Remove only the auxiliary component that reverses the anchor.
        projected = d - (dot.clamp_max(0) / aa.clamp_min(1e-30)).view(shape) * a
        projected_norm = projected.flatten(1).norm(dim=1)
        anchor_norm = aa.sqrt()
        scale = torch.minimum(torch.ones_like(anchor_norm),
            anchor_norm / projected_norm.clamp_min(1e-30))
        safe_distillation = projected * scale.view(shape)
        ctx.owner.record(a, d, safe_distillation, scale)
        return (a + safe_distillation).to(alignment_gradient.dtype), None


class AnchorPreservingGradientMerge:
    """Asymmetric GradNorm/PCGrad-style merge for an auxiliary teacher."""
    def __init__(self):
        self.records = []

    def pair(self, feature):
        if not feature.requires_grad:
            raise ValueError('gradient merge requires trainable tactile features')
        return _AnchorPreservingMerge.apply(feature, self)

    def record(self, alignment, distillation, safe_distillation, scale):
        self.records.append(tuple(x.detach().double() for x in
            (alignment, distillation, safe_distillation, scale)))

    @torch.no_grad()
    def statistics(self, distributed=True):
        if len(self.records) != 1:
            raise RuntimeError('expected exactly one merged tactile feature gradient')
        a, d, safe, scale = self.records.pop()
        a, d, safe = (x.flatten(1) for x in (a, d, safe))
        aa, dd = a.square().sum(1), d.square().sum(1)
        ss, ad = safe.square().sum(1), (a*d).sum(1)
        valid = (aa > 0) & (dd > 0)
        cosine = ad / (aa*dd).sqrt().clamp_min(1e-30)
        values = torch.stack([aa.sum(), dd.sum(), ss.sum(), ad.sum(),
            cosine[valid].sum(), (cosine[valid] < 0).sum(), valid.sum(),
            aa.new_tensor(len(aa)), scale.sum(), (scale < 1).sum()])
        if distributed and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
        aa, dd, ss, ad, cs, negative, valid_count, count, scale_sum, capped = values
        measured = (aa > 0) & (dd > 0)
        return dict(
            measured=measured.float(),
            global_cosine=(ad/(aa*dd).sqrt().clamp_min(1e-30)).clamp(-1, 1).float(),
            mean_contact_cosine=(cs/valid_count.clamp_min(1)).float(),
            negative_contact_fraction=(negative/valid_count.clamp_min(1)).float(),
            valid_contact_count=valid_count.float(), total_contact_count=count.float(),
            alignment_norm=aa.sqrt().float(), distillation_norm=dd.sqrt().float(),
            norm_ratio_distillation_to_alignment=(dd/aa.clamp_min(1e-30)).sqrt().float(),
            safe_distillation_norm=ss.sqrt().float(),
            safe_norm_ratio_to_alignment=(ss/aa.clamp_min(1e-30)).sqrt().float(),
            mean_distillation_scale=(scale_sum/count.clamp_min(1)).float(),
            capped_contact_fraction=(capped/count.clamp_min(1)).float(),
            anchor_preserved=aa.new_tensor(1.).float())


class AlignmentGradientComparison:
    def __init__(self):
        self.gradients = {}

    def view(self, feature, name):
        if name not in ('alignment', 'distillation'):
            raise ValueError('unknown objective gradient')
        branch = feature * 1.
        if not branch.requires_grad:
            raise ValueError('gradient diagnostics require trainable tactile features')
        def capture(gradient):
            value = gradient.detach().float().clone()
            # Accumulate if an autograd engine invokes a hook more than once.
            self.gradients[name] = value if name not in self.gradients else self.gradients[name] + value
        branch.register_hook(capture)
        return branch

    @torch.no_grad()
    def statistics(self, distributed=True):
        if set(self.gradients) != {'alignment', 'distillation'}:
            raise RuntimeError('both isolated objective gradients must be measured')
        a, b = [self.gradients[k].flatten(1).double() for k in ('alignment', 'distillation')]
        if a.shape != b.shape:
            raise ValueError('gradient feature spaces differ')
        aa, bb, ab = (a*a).sum(1), (b*b).sum(1), (a*b).sum(1)
        valid = (aa > 0) & (bb > 0)
        cosine = ab / (aa*bb).sqrt().clamp_min(1e-30)
        values = torch.stack([aa.sum(), bb.sum(), ab.sum(),
            cosine[valid].sum(), (cosine[valid] < 0).sum(), valid.sum(),
            aa.new_tensor(len(aa))])
        if distributed and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
        aa, bb, ab, cs, negative, valid_count, count = values
        measured = (aa > 0) & (bb > 0)
        denom = (aa*bb).sqrt().clamp_min(1e-30)
        # A zero gradient is recorded as invalid, never silently as evidence
        # of orthogonal/nonconflicting gradients.
        result = dict(
            measured=measured.float(), global_cosine=(ab/denom).clamp(-1, 1),
            mean_contact_cosine=cs/valid_count.clamp_min(1),
            negative_contact_fraction=negative/valid_count.clamp_min(1),
            valid_contact_count=valid_count, total_contact_count=count,
            alignment_norm=aa.sqrt(), distillation_norm=bb.sqrt(),
            norm_ratio_distillation_to_alignment=(bb/aa.clamp_min(1e-30)).sqrt(),
            cancellation_ratio=(aa+bb+2*ab).clamp_min(0).sqrt()/(aa.sqrt()+bb.sqrt()).clamp_min(1e-30))
        self.gradients.clear()
        return {k: v.float() for k, v in result.items()}
