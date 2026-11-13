"""Controlled local four-modality alignment / sensor-relation ablations.

The two switches alter objectives, never contact selection, initialization,
teacher targets, or the original DINO branch. Sensor geometry is TRAIN-only.
"""
import torch
import math
from .contact_regions import symmetric_pooled_contact_loss
from .paired_tactile_alignment import (
    RELATION_MODES, geometry_consensus_distillation, isolated_gradient_view)

ARMS = (
    'full', 'alignment_only', 'geometry_only',
    'rgb_only', 'depth_only', 'pointmap_only',
)
# Tactile is the only alignment hub.  Resource-resource terms would optimize
# frozen-anchor adapters without providing additional tactile supervision and
# are not part of the proposed objective.
PAIRS = (('rgb', 'tactile'), ('depth', 'tactile'), ('pc', 'tactile'))


def active_pairs(arm):
    """Return the exact tactile-centred alignment terms for one ablation arm."""
    if arm not in ARMS:
        raise ValueError('unknown controlled loss ablation')
    if arm in ('full', 'alignment_only'):
        return PAIRS
    if arm == 'rgb_only':
        return (PAIRS[0],)
    if arm == 'depth_only':
        return (PAIRS[1],)
    if arm == 'pointmap_only':
        return (PAIRS[2],)
    return ()


def weights(arm, alignment_scale=1.):
    pairs = active_pairs(arm)
    if not math.isfinite(alignment_scale) or not 0 < alignment_scale <= 1:
        raise ValueError('alignment scale must be finite in (0, 1]')
    relation = arm in ('full', 'geometry_only')
    return float(bool(pairs)) * alignment_scale, float(relation)


def contract(arm, alignment_scale=1., relation_mode=None, extra_cls_vt_alignment=False):
    align, relation = weights(arm, alignment_scale)
    pairs = active_pairs(arm)
    relation_mode = relation_mode or ('noncontact_conditional' if relation else 'full_distribution')
    if relation_mode not in RELATION_MODES or (relation_mode != 'full_distribution' and not relation):
        raise ValueError('invalid relation mode for loss arm')
    result = dict(arm=arm, alignment_weight=align, relation_weight=relation,
        alignment_pairs=[list(pair) for pair in pairs],
        alignment_scope=('same_canonical_local_contact_tactile_centered_'
                         + str(len(pairs)) + '_pairs'),
        resource_resource_alignment=False,
        teacher=('fixed_native_frozen_Depth_and_PointBERT_concordant_rank'
                 if relation_mode == 'agreement_rank' else
                 'fixed_native_frozen_Depth_and_PointBERT_contact_pooled_equal_consensus'),
        teacher_projection='none', teacher_gradient='stopped',
        sensor_geometry='frame10_uploaded_PLY_depth_PC_TRAIN_auxiliary_only',
        tactile_input='single_canonical_frame_repeated_inside_temporal_head',
        contact_representation='mask_aware_region_mean_256D',
        student='local_RGB_tactile_clip_matching_distribution',
        trainable='tactile_backbone_temporal_head_and_active_modality_projection_heads',
        external_backbones='frozen', global_to_local_geometry_distillation=False,
        relation_mode=relation_mode,
        old_sensor_pair_InfoNCE=False, generation=False, original_DINO_unchanged=True)
    if extra_cls_vt_alignment:
        if arm not in ('full', 'alignment_only', 'geometry_only', 'rgb_only'):
            raise ValueError('unsupported Local VT ablation')
        result.update(additional_alignment='backbone_CLS_cosine_InfoNCE',
            rgb_alignment_replaces_existing_term=False, additional_alignment_terms=1,
            extra_cls_vt_weight=1., original_three_alignment_terms_unchanged=len(pairs) == 3,
            rgb_readout='single_frame_tactile_CLS_and_frozen_local_RGB_CLS',
            geometry_readout='unchanged_mask_aware_region_mean_256D',
            gradient_merge_scope='unchanged_contact_head_RGB_Depth_PC_alignment_vs_distillation;_CLS_bypasses_head')
    return result


def objective(projected, tactile, ids, native_depth, native_point, valid,
              depth_mask, arm, diagnostics, temperature=.07, distributed=True,
              alignment_scale=1., gradient_comparison=None,
              relation_mode='full_distribution', metadata=None, rgb_cls=None,
              include_relation=True):
    align_weight, relation_weight = weights(arm, alignment_scale)
    if not include_relation:
        # Diagnostic replay only: measure the alignment terms actually enabled
        # for this arm, without introducing disabled CMA terms in CL + GD.
        relation_weight = 0.
    b = len(tactile)
    logs = {}
    total = tactile.sum() * 0.
    if rgb_cls is not None:
        if arm not in ('full', 'alignment_only', 'geometry_only', 'rgb_only'):
            raise ValueError('unsupported Local VT ablation')
        if (len(rgb_cls) != 2 or rgb_cls[0].ndim != 2
                or rgb_cls[0].shape != rgb_cls[1].shape or len(rgb_cls[0]) != b):
            raise ValueError('paired B x width RGB and tactile CLS descriptors required')
    merge = gradient_comparison is not None and hasattr(gradient_comparison, 'pair')
    if merge:
        alignment_tactile, relation_source = gradient_comparison.pair(tactile)
    else:
        alignment_tactile = (gradient_comparison.view(tactile, 'alignment')
                            if gradient_comparison is not None else tactile)
        relation_source = (gradient_comparison.view(tactile, 'distillation')
                           if gradient_comparison is not None else tactile)
    if align_weight:
        pairs = active_pairs(arm)
        values = dict(projected, tactile=alignment_tactile)
        masks = {'depth': depth_mask[:b]}
        terms = []
        for left, right in pairs:
            a, c = values[left], values[right]
            if right == 'tactile':
                c = isolated_gradient_view(c, diagnostics, left+'_to_tactile_image')
            loss, diag = symmetric_pooled_contact_loss(a, c, ids,
                left_mask=masks.get(left), right_mask=masks.get(right),
                temperature=temperature, distributed=distributed, metadata=metadata)
            name = left+'_'+right
            logs['align_'+name] = loss.detach()
            logs.update({'region_'+name+'_'+k: v for k, v in diag.items()})
            terms.append(loss)
        alignment = torch.stack(terms).mean() * align_weight
        logs['multiresource_alignment_loss'] = alignment.detach()
        logs['four_modality_pair_count'] = alignment.new_tensor(len(pairs))
        total = total + alignment
    else:
        logs['four_modality_pair_count'] = tactile.new_tensor(0.)
    if rgb_cls is not None:
        # User-requested additive experiment. Preserve ALL original regional
        # RGB/Depth/PC terms. This separate CLS term reaches the backbone
        # directly and is not part of the contact-head gradient merge.
        a, c = (value[:, None] for value in rgb_cls)
        c = isolated_gradient_view(c, diagnostics, 'extra_cls_vt_to_tactile_image')
        extra, diag = symmetric_pooled_contact_loss(a, c, ids,
            temperature=temperature, distributed=distributed, metadata=metadata)
        total = total + extra
        logs['extra_cls_vt_loss'] = extra.detach()
        logs['extra_cls_vt_weight'] = extra.new_tensor(1.)
        logs['extra_cls_vt_alignment_enabled'] = extra.new_tensor(1.)
        logs.update({'extra_cls_vt_'+k: v for k, v in diag.items()})
    if relation_weight:
        relation_tactile = isolated_gradient_view(relation_source, diagnostics,
            'geometry_relation_to_tactile_image')
        # Native within-modality similarities need no common feature width.
        # Do NOT use random or CL-trained heads: the teacher is identical in
        # full and geometry-only, including when CL is disabled.
        relation, diag = geometry_consensus_distillation(projected['rgb'],
            relation_tactile, native_depth, native_point, valid, depth_mask,
            temperature=temperature, distributed=distributed, detach_rgb=False,
            relation_mode=relation_mode, contact_ids=ids,
            score_mode='contact_pooled')
        total = total + relation_weight * relation
        logs['relation_distillation_loss'] = relation.detach()
        logs['geometry_relation_gradient_isolated'] = relation.new_tensor(1.)
        logs['relation_teacher_native_frozen'] = relation.new_tensor(1.)
        logs.update({'relation_'+k: v for k, v in diag.items()})
    if merge:
        logs['anchor_preserving_gradient_merge'] = total.new_tensor(1.)
    logs['four_modality_alignment_enabled'] = total.new_tensor(float(align_weight > 0))
    logs['geometry_relation_enabled'] = total.new_tensor(relation_weight)
    return total, logs
