import hashlib
import json
import os
import torch
import random
import logging
import math
import numbers
from pathlib import Path
from torchvision import transforms
from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)
import torchvision.transforms.functional as F
from typing import Sequence

from .datasets.tacdino.contact_alignment import index_contact_mapping, is_visible_coordinate
from .tactile_video import (
    LEGACY_CANONICAL_FRAME_INTERPOLATION,
    load_tactile_clip_from_mp4,
)


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")


def correspondence_values_to_xy(tactile_corr_data):
    """Convert stored image coordinates from [row, column] to [x, y]."""
    coordinates = torch.tensor(list(tactile_corr_data.values()))
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"invalid tactile correspondence shape: {tuple(coordinates.shape)}")
    return coordinates[:, [1, 0]]


def get_patch_index(points_orig, crop_coords, is_flipped, img_size=96, num_patches_side=16):
    
    if not isinstance(points_orig, torch.Tensor):
        points_orig = torch.tensor(points_orig)
    
    points_orig = points_orig.float()
    
    x1, y1, x2, y2 = crop_coords
    crop_w = x2 - x1
    crop_h = y2 - y1
    
    points_rel_x = (points_orig[..., 0] - x1) / crop_w
    points_rel_y = (points_orig[..., 1] - y1) / crop_h
    
    if is_flipped:
        points_rel_x = 1.0 - points_rel_x
        
    points_target_x = points_rel_x * img_size
    points_target_y = points_rel_y * img_size
    
    points_target_x = torch.clamp(points_target_x, min=0, max=img_size - 1e-4)
    points_target_y = torch.clamp(points_target_y, min=0, max=img_size - 1e-4)
    
    patch_size = img_size // num_patches_side
    
    grid_x = (points_target_x / patch_size).long()
    grid_y = (points_target_y / patch_size).long()
    
    grid_x = torch.clamp(grid_x, min=0, max=num_patches_side - 1)
    grid_y = torch.clamp(grid_y, min=0, max=num_patches_side - 1)
    
    patch_indices = (grid_y * num_patches_side + grid_x).long()
    
    return patch_indices


def get_index_closest_to_center(points, bbox):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points).float()
        
    x_min, y_min, x_max, y_max = bbox
    
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    valid = ~((points[:, 0] == 0) & (points[:, 1] == 0))
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("no visible tactile correspondence is available")
    valid_points = points[valid_indices]
    dist_sq = (valid_points[:, 0] - center_x)**2 + (valid_points[:, 1] - center_y)**2

    return valid_indices[torch.argmin(dist_sq)]

def get_indices_in_region_torch(points, bbox):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    
    x = points[:, 0]
    y = points[:, 1]
    
    x_min, y_min, x_max, y_max = bbox
    
    mask = (
        (x >= x_min)
        & (x <= x_max)
        & (y >= y_min)
        & (y <= y_max)
        & ~((x == 0) & (y == 0))
    )
    
    indices = torch.nonzero(mask).squeeze()
    return indices


def get_deterministic_contact_crop(points, image_size, crop_size, sample_index):
    """Return a reproducible contact-centered crop and its contact indices.

    Evaluation previously retried random crops and could finally pair a tactile
    sample with a crop that did not contain that contact.  This helper selects a
    valid contact deterministically and searches nearby crop positions for the
    smallest number of contacts, while always keeping the selected contact in
    the crop.
    """
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    points = points.float()
    image_width, image_height = image_size
    crop_width = min(int(crop_size), image_width)
    crop_height = min(int(crop_size), image_height)

    valid = (
        (points[:, 0] >= 0)
        & (points[:, 0] < image_width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < image_height)
        & ~((points[:, 0] == 0) & (points[:, 1] == 0))
    )
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    if not valid_indices:
        raise ValueError("evaluation image has no valid tactile correspondence")

    selected_index = valid_indices[int(sample_index) % len(valid_indices)]
    selected_x = float(points[selected_index, 0])
    selected_y = float(points[selected_index, 1])

    centered_left = min(max(int(round(selected_x - crop_width / 2)), 0), image_width - crop_width)
    centered_top = min(max(int(round(selected_y - crop_height / 2)), 0), image_height - crop_height)
    left_candidates = {
        centered_left,
        min(max(int(selected_x) - crop_width + 1, 0), image_width - crop_width),
        min(max(int(selected_x), 0), image_width - crop_width),
    }
    top_candidates = {
        centered_top,
        min(max(int(selected_y) - crop_height + 1, 0), image_height - crop_height),
        min(max(int(selected_y), 0), image_height - crop_height),
    }

    candidates = []
    for left in sorted(left_candidates):
        for top in sorted(top_candidates):
            box = [left, top, left + crop_width, top + crop_height]
            inside = get_indices_in_region_torch(points, torch.tensor(box))
            if isinstance(inside, torch.Tensor):
                inside = inside.flatten().tolist()
            elif isinstance(inside, (int, float)):
                inside = [int(inside)]
            inside = [idx for idx in inside if idx in valid_indices]
            if selected_index not in inside:
                continue
            distance = abs((left + crop_width / 2) - selected_x) + abs((top + crop_height / 2) - selected_y)
            candidates.append((len(inside), distance, box, inside))

    if not candidates:
        raise RuntimeError("failed to construct a contact-containing evaluation crop")
    _, _, box, inside = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return box, inside, selected_index


def resolve_tactile_point_cloud_path(tactile_root, contact_index):
    base = os.path.join(
        tactile_root,
        f"tactile_3d_{int(contact_index)}",
        f"contact_{int(contact_index)}_10",
    )
    candidates = [base + extension for extension in (".ply", ".npy", ".npz")]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    # Keep the expected PLY path so the preflight can report a useful name.
    return candidates[0]


def sample_contact_constrained_crop(augmentation, image, points, require_single_contact, max_attempts=32):
    """Sample a crop that always contains the tactile contact used as its pair."""
    best = None
    for _ in range(max_attempts):
        local_image, coordinates = augmentation(image)
        indices = get_indices_in_region_torch(points, torch.tensor(coordinates))
        if isinstance(indices, torch.Tensor):
            indices = indices.flatten().tolist()
        elif isinstance(indices, (int, float)):
            indices = [int(indices)]
        if not indices:
            continue
        candidate = (len(indices), local_image, coordinates, indices)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if not require_single_contact or len(indices) == 1:
            return local_image, coordinates, indices

    if best is not None:
        _, local_image, coordinates, indices = best
        return local_image, coordinates, indices

    valid = ~((points[:, 0] == 0) & (points[:, 1] == 0))
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    if not valid_indices:
        raise ValueError("training image has no visible tactile correspondence")
    selected_index = random.choice(valid_indices)
    selected_x = float(points[selected_index, 0])
    selected_y = float(points[selected_index, 1])
    width, height = image.size
    source_side = max(1, int(round((width * height * augmentation.scale[0]) ** 0.5)))
    crop_width = min(source_side, width)
    crop_height = min(source_side, height)
    left = min(max(int(round(selected_x - crop_width / 2)), 0), width - crop_width)
    top = min(max(int(round(selected_y - crop_height / 2)), 0), height - crop_height)
    coordinates = [left, top, left + crop_width, top + crop_height]
    local_image = F.resized_crop(
        image,
        top,
        left,
        crop_height,
        crop_width,
        augmentation.size,
        augmentation.interpolation,
    )
    if torch.rand(1) < augmentation.flip_p:
        local_image = F.hflip(local_image)
        augmentation.flip_state = True
    else:
        augmentation.flip_state = False
    indices = get_indices_in_region_torch(points, torch.tensor(coordinates)).flatten().tolist()
    if selected_index not in indices:
        raise RuntimeError("contact-centered fallback crop lost its selected contact")
    return local_image, coordinates, indices

class TacDataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        local_data="tac_patchalign",
        touch_config=None,
        generate_global_crops=True,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.local_data = local_data
        self.touch_config = touch_config
        # The training loop derives this from the model's actual execution
        # graph. Frozen semantic-only branches must not spend CPU time on
        # 518px crops that are never forwarded. Local/contact transforms stay
        # unchanged; ordinary DINO training keeps the legacy default.
        self.generate_global_crops = bool(generate_global_crops)
        self.requires_3d = True
        if local_data != "tac_patchalign":
            raise ValueError("GeoTAC requires tactile patch alignment")
        self.defer_tactile_loading = True

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"local data: {local_data}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = RandomResizedCropAndFlipWithCoords(size = local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC, flip_p=0.5)
        
        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        # self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])
        self.totensor = transforms.ToTensor()
        # The ViT-L main method starts from ImageNet-pretrained DINOv2.  Keep
        # both modalities in the same normalized input space before feature
        # extraction/fusion.  Legacy modes retain their original raw tensors.
        self.local_visual_transfo = self.totensor
        self.local_tactile_transfo = self.totensor

    def __call__(self, index, image, tactile_data,tactile_mat_label, tactile_corr_data, tactile_3d_location_data, global_scaned_pc_path, local_tactile_pc_path, inst_idx):
        output = {}

        global_crops = []
        if self.generate_global_crops:
            im1_base = self.geometric_augmentation_global(image)
            global_crop_1 = self.global_transfo1(im1_base)
            im2_base = self.geometric_augmentation_global(image)
            global_crop_2 = self.global_transfo2(im2_base)
            global_crops = [global_crop_1, global_crop_2]
        output["global_crops"] = global_crops
        output["global_crops_teacher"] = global_crops
        local_crops = []
        local_corrps_tactiles = []
        local_pair_ids = []
        # Keep the exact local sampling decision available to additive
        # multimodal branches.  Those branches must not independently draw a
        # second contact or silently use a different crop.
        local_contact_indices = []
        local_crop_boxes = []
        local_crop_flips = []
        local_all_coords = []
        
        coor_list = correspondence_values_to_xy(tactile_corr_data)
        contact_ids = [int(key) for key in tactile_corr_data]
        
        for _ in range(self.local_crops_number):
            local_image, corrdinate, face_indices = sample_contact_constrained_crop(
                self.geometric_augmentation_local,
                image,
                coor_list,
                True,
            )
            local_image = self.local_visual_transfo(local_image)
            local_crops.append(local_image)
            local_crop_boxes.append([int(value) for value in corrdinate])
            local_crop_flips.append(bool(self.geometric_augmentation_local.flip_state))
        
        
            selected = random.choice(face_indices)
            frame = tactile_data[selected]
            if isinstance(frame, (str, os.PathLike)):
                frame = load_tactile_clip_from_mp4(
                    os.fspath(frame), [10], image_size=self.local_crops_size,
                    resize_interpolation=LEGACY_CANONICAL_FRAME_INTERPOLATION,
                )[0]
            local_corrps_tactiles.append(self.local_tactile_transfo(frame))
            local_pair_ids.append((int(index), contact_ids[selected]))
            local_contact_indices.append(int(selected))

        output.update(local_crops=local_crops, local_tactile_crops=local_corrps_tactiles,
                      local_pair_ids=local_pair_ids, local_contact_indices=local_contact_indices,
                      local_crop_boxes=local_crop_boxes, local_crop_flips=local_crop_flips, offsets=())
        return output


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


class RandomResizedCropAndFlipWithCoords:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), 
                 interpolation=transforms.InterpolationMode.BICUBIC, flip_p=0.5):
        """
        Args:
            size (int or sequence): Expected output size of the crop, for each edge.
            scale (tuple): Range of size of the origin size cropped.
            ratio (tuple): Range of aspect ratio of the origin aspect ratio cropped.
            interpolation: Interpolation mode.
            flip_p: Probability of horizontal flip.
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            self.size = size
        
        self.flip_state = False

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.flip_p = flip_p

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input Image.
        Returns:
            tuple: (transformed_img, coords)
            coords format: [x1, y1, x2, y2] relative to original image
        """
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)
        
        coords = [j, i, j + w, i + h]

        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if torch.rand(1) < self.flip_p:
            img = F.hflip(img)
            self.flip_state = True
        else:
            self.flip_state = False

        return img, coords

class RandomCropWithCoords:
    def __init__(self, crop_size, mean, std):
        """
        Args:
            crop_size (int or sequence): Expected output size of the crop (e.g. 98).
            mean (sequence): Normalization mean.
            std (sequence): Normalization std.
            flip_p: Probability of horizontal flip.
        """
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

        self.mean = mean
        self.std = std
        self.flip_state = False

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input Image (Original large image).

        Returns:
            tuple: (transformed_img, coords)
            coords format: [x1, y1, x2, y2] relative to original image
        """
        img = F.to_tensor(img)
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.crop_size)
        
        coords = [j, i, j + w, i + h]

        img = F.crop(img, i, j, h, w)

        img = F.normalize(img, self.mean, self.std)

        return img, coords
