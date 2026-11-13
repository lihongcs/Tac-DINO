# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified for Tac-DINO.

import torch
import random

# def get_patch_index(coords, raw_pixel, img_size=96, num_patches_side=16):
    
#     if not isinstance(coords, torch.Tensor):
#         coords = torch.tensor(coords)
#     patch_size = img_size // num_patches_side
    
#     coords = torch.clamp(coords, min=0, max=img_size - 1)
    
#     x = coords[..., 0]
#     y = coords[..., 1]
    
#     grid_x = x // patch_size
#     grid_y = y // patch_size
    
#     patch_indices = (grid_y * num_patches_side + grid_x).long()
    
#     return patch_indices


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, local_data, n_tokens=None, mask_generator=None, skip_global_crops=False):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])
    if any(len(sample[0]["global_crops"]) != n_global_crops for sample in samples_list):
        raise ValueError("batch mixes global-crop protocols")
    if skip_global_crops != (n_global_crops == 0):
        raise ValueError("global-crop generation and collation contracts disagree")

    if local_data != "tac_patchalign":
        raise ValueError("GeoTAC requires tactile patch alignment")
    n_tactile_crops = len(samples_list[0][0]["local_tactile_crops"])
    tactile_crops = torch.stack([
        sample[0]["local_tactile_crops"][i]
        for i in range(n_tactile_crops) for sample in samples_list
    ])
    pair_ids = torch.tensor([
        sample[0]["local_pair_ids"][i]
        for i in range(n_tactile_crops) for sample in samples_list
    ], dtype=torch.long)

    collated_global_crops = (
        torch.empty((0, 3, 0, 0), dtype=dtype)
        if skip_global_crops
        else torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    )
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    
    B = len(collated_global_crops)
    N = int(n_tokens or 0)
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = (
        torch.empty((0, N), dtype=torch.bool)
        if skip_global_crops
        else torch.stack(masks_list).flatten(1)
    )
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "rank_local_batch_size": len(samples_list),
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    
    
    out["collated_local_tactile_crops"] = tactile_crops.to(dtype)
    out["collated_local_pair_ids"] = pair_ids
    return out
