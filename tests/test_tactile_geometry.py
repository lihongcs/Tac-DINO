import json
import os
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from dinov2.tacdino.tactile_geometry import (read_tactile_ply, organized_grid, geometry_arrays,
    training_geometry, UnavailableGeometry, training_views, preprocessing_certificate, geometry_index)


class GeometryTests(unittest.TestCase):
    def arrays(self):
        yy, xx = np.mgrid[:24, :32]
        xyz = np.stack([xx, yy, np.sin(xx)*np.cos(yy)], -1).reshape(-1, 3).astype(np.float64)
        rgb = np.tile(np.array([[100, 100, 100]], np.uint8), (len(xyz), 1))
        return xyz, rgb

    def test_ply_reader_preserves_raw_xyz_rgb(self):
        xyz, rgb = self.arrays()
        dtype = np.dtype([(k, '<f8') for k in ('x', 'y', 'z')]+[(k, 'u1') for k in ('red', 'green', 'blue')])
        array = np.empty(len(xyz), dtype=dtype)
        for i, key in enumerate(('x', 'y', 'z')): array[key] = xyz[:, i]
        for i, key in enumerate(('red', 'green', 'blue')): array[key] = rgb[:, i]
        header = ('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(xyz))+'\n'+
            ''.join('property double '+k+'\n' for k in ('x', 'y', 'z'))+
            ''.join('property uchar '+k+'\n' for k in ('red', 'green', 'blue'))+'end_header\n').encode()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'test.ply'; original = header+array.tobytes(); path.write_bytes(original)
            a, b = read_tactile_ply(path)
            np.testing.assert_array_equal(a, xyz); np.testing.assert_array_equal(b, rgb)
            self.assertEqual(path.read_bytes(), original)

    def test_grid_uses_coordinates_not_file_order(self):
        xyz, rgb = self.arrays()
        grid, colors = organized_grid(xyz, rgb)
        order = np.random.default_rng(2).permutation(len(xyz))
        shuffled, _ = organized_grid(xyz[order], rgb[order])
        np.testing.assert_array_equal(grid, shuffled)
        perturbed = xyz.copy(); perturbed[17, 1] += 1e-7
        fixed, _ = organized_grid(perturbed, rgb)
        np.testing.assert_array_equal(grid, fixed)
        xyz[17, 1] += .1
        with self.assertRaises(ValueError): organized_grid(xyz, rgb)

    def test_relative_depth_unit_invariant_and_finite(self):
        xyz, rgb = self.arrays()
        a = geometry_arrays(xyz, rgb)
        xyz[:, 2] = xyz[:, 2]*1000+25
        b = geometry_arrays(xyz, rgb)
        np.testing.assert_allclose(a['height'], b['height'], atol=1e-6)
        self.assertEqual(a['xyz'].shape, (98, 98, 3))
        self.assertTrue(a['valid'].all())

    def test_duplicate_and_nonfinite_points_become_invalid_not_fake_geometry(self):
        xyz, rgb = self.arrays()
        xyz[17] = xyz[0]
        xyz[45, 2] = np.nan
        points, _, mask = organized_grid(xyz, rgb, return_mask=True)
        self.assertFalse(mask[0, 0]); self.assertFalse(mask[0, 17]); self.assertFalse(mask[1, 13])
        self.assertTrue(np.isfinite(points).all())
        converted = geometry_arrays(xyz, rgb)
        self.assertGreater(converted['valid'].mean(), .9)
        self.assertTrue(np.isfinite(converted['height']).all())
        self.assertTrue((converted['height'][~converted['valid']] == 0).all())

    def test_unavailable_geometry_is_explicit_not_wrongly_renamed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); manifest = root/'index.json'
            video = root/'bowl/obj_0000/Tactile/tactile_3.mp4'
            key = str(video.relative_to(root))
            manifest.write_text(json.dumps(dict(complete=True, frame=10, data_root=str(root),
                preprocessing_validation=preprocessing_certificate(0, 0),
                entries={key: dict(usable=False, reason='ambiguous')})))
            with patch.dict(os.environ, TACDINO_TACTILE_GEOMETRY_MANIFEST=str(manifest)):
                self.assertEqual(training_geometry(video), (None, None, False))
                with self.assertRaises(ValueError): training_geometry(root/'unknown.mp4')

    def test_failed_reconstruction_is_not_fabricated(self):
        xyz, rgb = self.arrays()
        xyz[:, 2] = np.nan
        with self.assertRaises(UnavailableGeometry): geometry_arrays(xyz, rgb)
        xyz[:, 2] = 0
        xyz[17, 0] += .1
        # Invalid raster conventions remain unexpected errors, not unavailable observations.
        with self.assertRaises(ValueError) as error: geometry_arrays(xyz, rgb)
        self.assertNotIsInstance(error.exception, UnavailableGeometry)

    def test_exact_model_preprocessing_unchanged_for_valid_geometry(self):
        from dinov2.tacdino.pointbert_runtime import prepare_local_pool
        xyz, rgb = self.arrays()
        arrays = geometry_arrays(xyz, rgb)
        depth, pc = training_views(arrays)
        np.testing.assert_array_equal(pc, prepare_local_pool(arrays['xyz'][arrays['valid']]))
        np.testing.assert_array_equal(depth[0], arrays['height'])
        np.testing.assert_array_equal(depth[3], arrays['valid'])

    def test_finite_exploding_heights_rejected_before_training(self):
        xyz, rgb = self.arrays()
        arrays = geometry_arrays(xyz, rgb)
        for scale in (5.7e36, 2.3e37):
            arrays['xyz'][..., 2] = np.linspace(-scale, scale, 98*98).reshape(98, 98)
            self.assertTrue(np.isfinite(arrays['xyz']).all())
            with self.assertRaisesRegex(UnavailableGeometry, 'Point-BERT'):
                training_views(arrays)

    def test_uncertified_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'index.json'
            path.write_text(json.dumps(dict(complete=True, frame=10, entries={})))
            with self.assertRaisesRegex(ValueError, 'certification'): geometry_index(path)

    def test_certifier_masks_invalid_view_but_preserves_contact(self):
        from scripts.certify_tactile_geometry import certify_contact
        xyz, rgb = self.arrays()
        good = geometry_arrays(xyz, rgb)
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)/'contact.npz'
            np.savez_compressed(cache, **good)
            original_bytes = cache.read_bytes()
            key, row, unexpected = certify_contact(('contact.mp4', dict(usable=True, cache=str(cache))))
            self.assertTrue(row['usable']); self.assertFalse(unexpected)
            self.assertEqual(row['cache_sha256'], hashlib.sha256(original_bytes).hexdigest())
            good['xyz'][..., 2] = np.linspace(-2e37, 2e37, 98*98).reshape(98, 98)
            np.savez_compressed(cache, **good)
            original_bytes = cache.read_bytes()
            key, row, unexpected = certify_contact(('contact.mp4', dict(usable=True, cache=str(cache))))
            self.assertEqual(key, 'contact.mp4'); self.assertFalse(row['usable']); self.assertFalse(unexpected)
            self.assertEqual(row['reason'], 'pointbert_preprocessing_rejected')
            self.assertEqual(cache.read_bytes(), original_bytes)
            # Parser corruption must block certification, not silently skip.
            cache.write_bytes(b'not a numpy archive')
            _, _, unexpected = certify_contact(('contact.mp4', dict(usable=True, cache=str(cache))))
            self.assertTrue(unexpected)

    def test_loader_checks_certified_bytes_and_reports_contact(self):
        xyz, rgb = self.arrays()
        arrays = geometry_arrays(xyz, rgb)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); cache = root/'contact.npz'; manifest = root/'index.json'
            np.savez_compressed(cache, **arrays)
            key = 'bowl/obj_0000/Tactile/tactile_3.mp4'
            manifest.write_text(json.dumps(dict(complete=True, frame=10, data_root=str(root),
                preprocessing_validation=preprocessing_certificate(1, 1),
                entries={key: dict(usable=True, cache=str(cache),
                    cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())})))
            with patch.dict(os.environ, TACDINO_TACTILE_GEOMETRY_MANIFEST=str(manifest)):
                depth, pc, available = training_geometry(root/key)
                self.assertTrue(available)
                expected_depth, expected_pc = training_views(arrays)
                np.testing.assert_array_equal(depth, expected_depth); np.testing.assert_array_equal(pc, expected_pc)
                cache.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'tactile_3.mp4.*checksum changed'):
                    training_geometry(root/key)

    def test_full_certification_publishes_new_index_without_source_mutation(self):
        from scripts.certify_tactile_geometry import main
        xyz, rgb = self.arrays()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'original.json'; output = root/'checked.json'
            good = root/'good.npz'; bad = root/'bad.npz'
            arrays = geometry_arrays(xyz, rgb); np.savez_compressed(good, **arrays)
            arrays['xyz'][..., 2] = np.linspace(-2e37, 2e37, 98*98).reshape(98, 98)
            np.savez_compressed(bad, **arrays)
            source.write_text(json.dumps(dict(complete=True, frame=10, entries={
                'good.mp4': dict(usable=True, cache=str(good)),
                'bad.mp4': dict(usable=True, cache=str(bad)),
                'ambiguous.mp4': dict(usable=False, reason='ambiguous')})))
            before = {p: p.read_bytes() for p in (source, good, bad)}
            with patch('sys.argv', ['certify', '--manifest', str(source), '--output', str(output),
                    '--report', str(root/'report'), '--workers', '2']): main()
            result = geometry_index(output)
            self.assertTrue(result['complete']); self.assertEqual(result['total'], 3)
            self.assertEqual(result['usable'], 1); self.assertEqual(result['preprocessing_validation']['checked'], 2)
            self.assertEqual(set(result['entries']), {'good.mp4', 'bad.mp4', 'ambiguous.mp4'})
            for p, payload in before.items(): self.assertEqual(p.read_bytes(), payload)


if __name__ == '__main__': unittest.main()
