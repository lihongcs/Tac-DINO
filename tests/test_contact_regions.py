import tempfile
import unittest
import numpy as np
import torch
from dinov2.layers.contact_regions import (
    pool_regions, pooled_scores, region_scores,
    symmetric_pooled_contact_loss, symmetric_region_contact_loss,
)
from dinov2.layers.tactile_temporal import build_clip_head


class RegionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_soft_matching_not_same_index(self):
        x = torch.randn(3, 7, 16)
        y = torch.randn(4, 5, 16)
        score = region_scores(x, y)
        self.assertTrue(torch.allclose(score, region_scores(x[:, [4, 0, 6, 1, 2, 3, 5]], y.flip(1)), atol=1e-6))
        self.assertTrue(torch.allclose(score, region_scores(y, x).T, atol=1e-6))

    def test_masks_and_count_normalization(self):
        x, y = torch.randn(2, 3, 8), torch.randn(4, 5, 8)
        base = region_scores(x, y)
        # Uniform replication adds no artificial advantage to a modality.
        self.assertTrue(torch.allclose(base, region_scores(x.repeat_interleave(2, 1), y), atol=1e-6))
        padded = torch.cat([y, torch.randn(4, 9, 8)*100], 1)
        mask = torch.arange(14)[None].expand(4, -1) < 5
        self.assertTrue(torch.allclose(base, region_scores(x, padded, right_mask=mask), atol=1e-6))
        with self.assertRaises(ValueError): region_scores(x, padded, right_mask=torch.zeros_like(mask))

    def test_contact_pool_is_the_exported_descriptor(self):
        x = torch.tensor([[[1., 0.], [0., 1.], [100., 100.]]])
        mask = torch.tensor([[True, True, False]])
        torch.testing.assert_close(pool_regions(x, mask), torch.tensor([[.5, .5]]))
        score = pooled_scores(x, x, mask, mask)
        torch.testing.assert_close(score, torch.ones_like(score))

    def test_pooled_loss_cannot_use_candidate_specific_region_shortcuts(self):
        ids = torch.arange(3)
        left = torch.eye(3).repeat_interleave(2, 0).reshape(3, 2, 3)
        right = left.flip(1).clone().requires_grad_()
        loss, diagnostics = symmetric_pooled_contact_loss(
            left, right, ids, distributed=False)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(diagnostics['logit_margin']))
        loss.backward()
        self.assertTrue(torch.isfinite(right.grad).all())

    def test_duplicate_contacts_are_positives(self):
        x = torch.randn(3, 5, 8, requires_grad=True)
        loss, diag = symmetric_region_contact_loss(x, x, torch.ones(3, dtype=torch.long), distributed=False)
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(diag['positive_count'].item(), 3.)
        self.assertEqual(diag['unique_contacts'].item(), 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_three_losses_reach_same_encoder_and_every_frame(self):
        encoder = torch.nn.Linear(10, 12)
        head = build_clip_head('temporal_regions', 12, 16)
        images = torch.randn(5, 4, 7, 10, requires_grad=True)
        temporal = head(encoder(images))
        losses = []
        for regions in (9, 7, 11):
            loss, _ = symmetric_region_contact_loss(torch.randn(5, regions, 16), temporal,
                torch.arange(5), distributed=False)
            grad = torch.autograd.grad(loss, encoder.weight, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm().item(), 0.)
            losses.append(loss)
        torch.stack(losses).mean().backward()
        self.assertTrue((images.grad.abs().sum((0, 2, 3)) > 0).all())
        self.assertGreater(head.temporal.layers[0].self_attn.in_proj_weight.grad.norm().item(), 0.)

    def test_geometry_views_are_same_contact_positives(self):
        left, right = torch.randn(8, 5, 12, requires_grad=True), torch.randn(4, 7, 12, requires_grad=True)
        ids = torch.arange(4)
        loss, diag = symmetric_region_contact_loss(left, right, ids.repeat(2), right_ids=ids, distributed=False)
        self.assertEqual(diag['unique_contacts'].item(), 4)
        self.assertEqual(diag['reverse_positive_count'].item(), 2)
        loss.backward()
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue((left.grad.flatten(1).norm(dim=1) > 0).all())

    def test_invalid_geometry_is_not_a_candidate_or_gradient(self):
        left, right = torch.randn(4, 5, 12), torch.randn(4, 7, 12)
        ids = torch.arange(4)
        base, _ = symmetric_region_contact_loss(left, right, ids, distributed=False)
        views = torch.cat([left, torch.randn_like(left)*100]).requires_grad_()
        loss, _ = symmetric_region_contact_loss(views, right, ids.repeat(2), right_ids=ids,
            left_valid=torch.tensor([True]*4+[False]*4), distributed=False)
        torch.testing.assert_close(loss, base)
        loss.backward()
        self.assertEqual(views.grad[4:].abs().sum().item(), 0.)

    def test_region_head_strict_roundtrip(self):
        head = build_clip_head('temporal_regions', width=12).eval()
        restored = build_clip_head('temporal_regions', width=12).eval()
        restored.load_state_dict(head.state_dict(), strict=True)
        x = torch.randn(2, 4, 49, 12)
        torch.testing.assert_close(restored(x), head(x))

    def test_tiny_contacts_can_be_learned(self):
        encoder = torch.nn.Linear(10, 12)
        head = build_clip_head('temporal_regions', 12, 16)
        inputs = torch.randn(8, 4, 5, 10)
        target = torch.randn(8, 6, 16)
        opt = torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()), lr=.004)
        values = []
        for _ in range(90):
            opt.zero_grad()
            loss, _ = symmetric_region_contact_loss(target, head(encoder(inputs)), torch.arange(8), distributed=False)
            loss.backward(); opt.step(); values.append(loss.item())
        self.assertLess(values[-1], values[0]*.15)
        with torch.no_grad():
            score = region_scores(target, head(encoder(inputs)))
        self.assertTrue(torch.equal(score.argmax(1), torch.arange(8)))


if __name__ == '__main__': unittest.main()
