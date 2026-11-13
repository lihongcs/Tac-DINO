import copy
import unittest
from types import SimpleNamespace
import torch
from dinov2.layers.local_four_modality import ARMS, PAIRS, active_pairs, contract, objective, weights
from dinov2.layers.paired_tactile_alignment import geometry_consensus_distillation
from dinov2.train.training_diagnostics import objective_ledger


class Settings(dict):
    __getattr__ = dict.__getitem__


class LocalFourModalityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.b = 4
        self.projected = {k: torch.randn(4, 3, 8, requires_grad=True) for k in ('rgb', 'depth', 'pc')}
        self.tactile = torch.randn(4, 3, 8, requires_grad=True)
        self.depth = torch.randn(8, 3, 12, requires_grad=True)
        self.point = torch.randn(8, 5, 6, requires_grad=True)
        self.mask = torch.ones(8, 3, dtype=torch.bool)
        self.valid = torch.ones(4, dtype=torch.bool)
        self.ids = torch.arange(4)

    def call(self, arm, diagnostics=None):
        return objective(self.projected, self.tactile, self.ids, self.depth, self.point,
            self.valid, self.mask, arm, diagnostics if diagnostics is not None else {}, distributed=False)

    def test_exact_two_factor_sum_same_teacher(self):
        full, f = self.call('full')
        alignment, a = self.call('alignment_only')
        geometry, g = self.call('geometry_only')
        torch.testing.assert_close(full, alignment+geometry)
        torch.testing.assert_close(f['relation_teacher_entropy'], g['relation_teacher_entropy'])
        self.assertNotIn('relation_distillation_loss', a)
        self.assertNotIn('multiresource_alignment_loss', g)
        self.assertEqual(PAIRS, (('rgb', 'tactile'), ('depth', 'tactile'), ('pc', 'tactile')))
        self.assertEqual(f['four_modality_pair_count'].item(), 3)
        self.assertFalse(contract('full')['old_sensor_pair_InfoNCE'])
        self.assertFalse(contract('full')['resource_resource_alignment'])

    def test_every_tactile_alignment_branch_has_independent_gradient(self):
        diag = {}
        loss, _ = self.call('full', diag)
        loss.backward()
        for name in ('rgb', 'depth', 'pc', 'geometry_relation'):
            self.assertGreater(diag[name+'_to_tactile_image'].item(), 0.)
        self.assertIsNone(self.depth.grad)
        self.assertIsNone(self.point.grad)
        for value in self.projected.values(): self.assertGreater(value.grad.norm().item(), 0.)

    def test_geometry_only_not_hidden_alignment_or_random_teacher_heads(self):
        diag = {}
        a, _ = self.call('geometry_only', diag)
        with torch.no_grad():
            self.projected['depth'].mul_(13)
            self.projected['pc'].normal_()
        b, _ = self.call('geometry_only')
        torch.testing.assert_close(a, b)
        b.backward()
        self.assertIsNone(self.projected['depth'].grad)
        self.assertIsNone(self.projected['pc'].grad)
        self.assertGreater(self.projected['rgb'].grad.norm().item(), 0.)
        self.assertGreater(self.tactile.grad.norm().item(), 0.)
        self.assertIsNone(self.depth.grad)
        self.assertIsNone(self.point.grad)

    def test_tactile_centered_terms_and_ledger(self):
        for arm in ARMS:
            loss, logs = self.call(arm)
            a, g = weights(arm)
            cfg = SimpleNamespace(touch=Settings(local_four_modality_ablation=arm,
                multiresource_weight=a, relation_distillation_weight=g),
                train=SimpleNamespace(local_data='tac_patchalign'))
            ledger = objective_ledger(logs, cfg, loss)
            self.assertEqual(ledger['loss_accounting_pass'].item(), 1.)
            if a:
                terms = [logs['align_'+x+'_'+y] for x, y in active_pairs(arm)]
                torch.testing.assert_close(torch.stack(terms).mean(), logs['multiresource_alignment_loss'])
                self.assertFalse(any(key in logs for key in (
                    'align_rgb_depth', 'align_rgb_pc', 'align_depth_pc')))

    def test_single_source_alignment_arms_are_exact(self):
        expected = {
            'rgb_only': (('rgb', 'tactile'),),
            'depth_only': (('depth', 'tactile'),),
            'pointmap_only': (('pc', 'tactile'),),
        }
        for arm, pairs in expected.items():
            loss, logs = self.call(arm)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(active_pairs(arm), pairs)
            self.assertEqual(contract(arm)['alignment_pairs'], [list(pair) for pair in pairs])
            self.assertEqual(logs['four_modality_pair_count'].item(), 1)
            self.assertNotIn('relation_distillation_loss', logs)
            present = {key for key in logs if key.startswith('align_')}
            self.assertEqual(present, {'align_'+pairs[0][0]+'_tactile'})

    def test_duplicate_contact_is_not_false_negative(self):
        self.ids.zero_()
        loss, _ = self.call('alignment_only')
        torch.testing.assert_close(loss, torch.zeros_like(loss))

    def test_depth_validity_and_region_permutation(self):
        self.mask[:, -1] = False
        a, _ = self.call('geometry_only')
        with torch.no_grad(): self.depth[:, -1].normal_(0, 100)
        b, _ = self.call('geometry_only')
        torch.testing.assert_close(a, b)
        self.depth = self.depth.flip(1)
        self.mask = self.mask.flip(1)
        c, _ = self.call('geometry_only')
        torch.testing.assert_close(a, c)

    def test_no_sensor_geometry_fails_only_when_needed(self):
        self.valid.zero_()
        self.assertTrue(torch.isfinite(self.call('alignment_only')[0]))
        with self.assertRaisesRegex(ValueError, 'no certified'):
            self.call('geometry_only')


if __name__ == '__main__': unittest.main()
