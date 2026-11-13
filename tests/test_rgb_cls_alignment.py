"""The additive CLS trial preserves every original GeoTAC objective."""
import torch
from torch.nn import functional as F
from dinov2.layers.local_four_modality import objective, contract
from dinov2.layers.contact_regions import symmetric_pooled_contact_loss
from dinov2.layers.alignment_gradient_diagnostics import AnchorPreservingGradientMerge


def fixture():
    torch.manual_seed(17)
    b = 6
    projected = {k: torch.randn(b, 3, 8, requires_grad=True) for k in ('rgb', 'depth', 'pc')}
    tactile = torch.randn(b, 3, 8, requires_grad=True)
    ids = torch.arange(b)
    depth, point = torch.randn(2*b, 3, 12), torch.randn(2*b, 4, 10)
    valid, mask = torch.ones(b, dtype=torch.bool), torch.ones(2*b, 3, dtype=torch.bool)
    cls = (torch.randn(b, 16), torch.randn(b, 16, requires_grad=True))
    return (projected, tactile, ids, depth, point, valid, mask), cls


def test_three_original_terms_plus_cls_matches_feature_level_infonce():
    args, cls = fixture()
    diagnostics = {}
    total, logs = objective(*args, 'alignment_only', diagnostics, distributed=False, rgb_cls=cls)
    logits = F.normalize(cls[0], dim=-1) @ F.normalize(cls[1], dim=-1).T / .07
    target = torch.arange(len(logits))
    rgb = .5*(F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    depth, _ = symmetric_pooled_contact_loss(args[0]['depth'], args[1], args[2], distributed=False)
    point, _ = symmetric_pooled_contact_loss(args[0]['pc'], args[1], args[2], distributed=False)
    original_rgb, _ = symmetric_pooled_contact_loss(args[0]['rgb'], args[1], args[2], distributed=False)
    torch.testing.assert_close(total, (original_rgb+depth+point)/3 + rgb)
    torch.testing.assert_close(logs['extra_cls_vt_loss'], rgb)
    torch.testing.assert_close(logs['align_rgb_tactile'], original_rgb)
    assert logs['four_modality_pair_count'] == 3
    total.backward()
    assert cls[1].grad.norm() > 0
    assert args[0]['rgb'].grad.norm() > 0
    assert args[0]['depth'].grad.norm() > 0 and args[0]['pc'].grad.norm() > 0
    assert diagnostics['rgb_to_tactile_image'] > 0
    assert diagnostics['extra_cls_vt_to_tactile_image'] > 0


def test_geometry_terms_and_distillation_are_identical():
    args, cls = fixture()
    _, old = objective(*args, 'full', {}, distributed=False, relation_mode='agreement_rank')
    _, new = objective(*args, 'full', {}, distributed=False, relation_mode='agreement_rank', rgb_cls=cls)
    for name in ('align_rgb_tactile', 'align_depth_tactile', 'align_pc_tactile', 'relation_distillation_loss'):
        torch.testing.assert_close(old[name], new[name])
    c = contract('full', relation_mode='agreement_rank', extra_cls_vt_alignment=True)
    assert c['additional_alignment_terms'] == 1 and len(c['alignment_pairs']) == 3
    assert not c['rgb_alignment_replaces_existing_term']
    gd = contract('geometry_only', extra_cls_vt_alignment=True)
    assert gd['alignment_pairs'] == [] and gd['extra_cls_vt_weight'] == 1.


def test_cls_addition_preserves_contact_head_merge_and_cls_gradient():
    args, cls = fixture()
    merge = AnchorPreservingGradientMerge()
    total, _ = objective(*args, 'full', {}, distributed=False, rgb_cls=cls,
                         gradient_comparison=merge, relation_mode='agreement_rank')
    total.backward()
    stats = merge.statistics(distributed=False)
    assert stats['anchor_preserved'] == 1
    assert stats['safe_norm_ratio_to_alignment'] <= 1.00001
    assert cls[1].grad.norm() > 0 and args[1].grad.norm() > 0


def test_additive_objective_ledger_accounts_for_extra_term_once():
    from omegaconf import OmegaConf
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('isolated_ledger',
        Path(__file__).resolve().parents[1]/'dinov2/train/training_diagnostics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    objective_ledger = module.objective_ledger
    args, cls = fixture()
    total, logs = objective(*args, 'full', {}, distributed=False,
        relation_mode='agreement_rank', rgb_cls=cls)
    cfg = OmegaConf.create(dict(touch=dict(local_four_modality_ablation='full',
        multiresource_weight=1., relation_distillation_weight=1.), train=dict(local_data='tac_patchalign')))
    ledger = objective_ledger(logs, cfg, total)
    assert ledger['loss_accounting_pass'] == 1
    torch.testing.assert_close(ledger['loss_objective/extra_cls_vt_loss'], logs['extra_cls_vt_loss'])
