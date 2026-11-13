"""Preserve original DINO crops; replace local CL with full-clip resource CL."""
import os
import torch
from dinov2.data.contact_consistency import matched_contact_box, contact_metadata
from dinov2.data.contact_inputs import preserve_data_rng, validate_contact_input_alignment
from dinov2.data.tacdino_augmentations import TacDataAugmentationDINO
from dinov2.data.legacy_fast_input import LazyCanonicalFrames, canonical_frame
from dinov2.data.collate import collate_data_and_cast
from dinov2.tacdino.contact_clip import recording, contact_key, geometry_inputs
from dinov2.tacdino.local_pointmap import load_local_pointmap_box
from dinov2.tacdino.pointbert_runtime import prepare_local_pool
from dinov2.tacdino.tactile_geometry import training_geometry


class MultiResourceAugmentation(TacDataAugmentationDINO):
    needs_rgb_source_path = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = kwargs['touch_config'].tactile_geometry_manifest
        if manifest != os.environ.get('TACDINO_TACTILE_GEOMETRY_MANIFEST'):
            raise ValueError('model/data tactile geometry binding differs')
        self.requires_3d = False  # Never load whole-shape or tactile-derived PC into original branch.
        self.defer_tactile_loading = True
        self.contact_metadata_path = kwargs.get('touch_config', {}).get('contact_metadata_path')
        if self.local_data != "tac_patchalign":
            raise ValueError("requires original local tactile DINO route")

    def __call__(self, index, image, tactile_data, materials, correspondence,
                 tactile_location, global_pc, tactile_pc, inst_idx):
        paths = tactile_data.paths if isinstance(tactile_data, LazyCanonicalFrames) else tactile_data
        validate_contact_input_alignment(paths, correspondence, materials)
        result = super().__call__(index, image, LazyCanonicalFrames(paths), materials, correspondence,
                                  None, None, None, inst_idx)
        with preserve_data_rng():
            # Reuse the canonical Patch Tac decision.  Do not independently
            # sample a contact for the multimodal branch.
            if not result.get("local_contact_indices") or not result.get("local_crop_boxes"):
                raise ValueError("canonical local-contact metadata is missing")
            geometry = None
            failures = []
            # Geometry can be sparse in a valid random RGB crop. Choose among
            # the already sampled canonical local slots; never draw a new
            # contact/crop and never fabricate missing geometry.
            for slot, (candidate, candidate_box, candidate_flip) in enumerate(zip(
                    result["local_contact_indices"], result["local_crop_boxes"],
                    result["local_crop_flips"])):
                try:
                    candidate_box = matched_contact_box(list(correspondence.values())[candidate],
                        image.size, self.local_crops_size)
                    candidate_flip = False
                    object_id, material_set = contact_metadata(self.contact_metadata_path, paths[candidate])
                    candidate_geometry = load_local_pointmap_box(
                        image.info["tacdino_rgb_path"], candidate_box,
                        size=self.local_crops_size, flip=bool(candidate_flip),
                        return_camera=True)
                    candidate_xyz, candidate_mask, _, candidate_camera = candidate_geometry
                    # Validate the exact tensors consumed by both geometry
                    # encoders before accepting this already-sampled local
                    # slot.  In particular, Point-BERT requires at least 32
                    # post-z-buffer camera points.  Do not accept a weaker
                    # raster-only threshold and fail later in a worker.
                    _, candidate_depth = geometry_inputs(
                        candidate_xyz, candidate_mask, candidate_camera)
                    candidate_pc = prepare_local_pool(candidate_camera[candidate_mask])
                    geometry = candidate_geometry
                    depth = candidate_depth
                    pc = candidate_pc
                    selected = int(candidate)
                    crop_box = candidate_box
                    flip = bool(candidate_flip)
                    break
                except ValueError as error:
                    failures.append(f"slot{slot}:{error}")
            if geometry is None:
                raise ValueError("no canonical local slot has valid geometry: " + "; ".join(failures))
            video = paths[selected]
            frames, count, fps = recording(video)
            # The benchmark query is one canonical tactile RGB image. Repeat
            # that one observed image only inside the existing temporal head,
            # so training and evaluation consume exactly the same information.
            tactile_image = self.local_tactile_transfo(canonical_frame(video))
            clip = [tactile_image] * 4
            xyz, mask, audit, camera = geometry
            rgb = self.local_visual_transfo(image.crop(tuple(crop_box)))
            result.update(resource_object_id=object_id, resource_material_set=material_set, resource_rgb=rgb,
                resource_tactile=torch.stack(clip),
                resource_depth=torch.from_numpy(depth), resource_pc=torch.from_numpy(pc),
                resource_id=torch.tensor(contact_key(video)), resource_frames=torch.tensor([10] * 4),
                resource_frame_count=torch.tensor(count), resource_fps=torch.tensor(fps),
                resource_tactile_unique_frames=torch.tensor(1),
                resource_coverage=torch.tensor(float(mask.mean())),
                resource_local_slot=torch.tensor(slot),
                resource_geometry_attempts=torch.tensor(slot + 1),
                resource_contact_index=torch.tensor(selected),
                resource_crop_box=torch.tensor(crop_box),
                resource_crop_flip=torch.tensor(flip),
                resource_canonical_pair_id=torch.tensor(result["local_pair_ids"][slot]))
            tactile_depth, tactile_pc, available = training_geometry(video)
            result.update(resource_tactile_depth=torch.from_numpy(tactile_depth if available else depth),
                resource_tactile_pc=torch.from_numpy(tactile_pc if available else pc),
                resource_tactile_geometry_valid=torch.tensor(available),
                resource_tactile_geometry_frame=torch.tensor(10))
        return result


def collate_multiresource(samples_list, *args, **kwargs):
    result = collate_data_and_cast(samples_list, *args, **kwargs)
    for key in samples_list[0][0]:
        if key.startswith("resource_"):
            result[key] = torch.stack([sample[0][key] for sample in samples_list])
    return result
