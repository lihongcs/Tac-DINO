# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified for Tac-DINO.
import os
import torch
import random
import logging
import numbers
from typing import Sequence
from torchvision import transforms

logger = logging.getLogger("dinov2")

import torchvision.transforms.functional as F



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
class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


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


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)



def get_index_closest_to_center(points, bbox):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points).float()
        
    x_min, y_min, x_max, y_max = bbox
    
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    dist_sq = (points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2
    
    return torch.argmin(dist_sq)

# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44

def get_indices_in_region_torch(points, bbox):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    
    x = points[:, 0]
    y = points[:, 1]
    
    x_min, y_min, x_max, y_max = bbox
    
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    
    indices = torch.nonzero(mask).squeeze()
    return indices

class EvalLinearTacDataAugmentationDINO(object):
    def __init__(
        self,
        crop_size: int = 224,
        local_crop_size: int = 98,
        local_crops_number:int = 1,
        interpolation=transforms.InterpolationMode.BICUBIC,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
        local_data="tactile"
    ):  
        
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.Resize(crop_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                # transforms.RandomHorizontalFlip(hflip_prob),
                MaybeToTensor(),
                # make_normalize_transform(mean=mean, std=std),
            ]
        )
        
        self.totensor = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.local_transfo = transforms.Compose([self.totensor])
        
        self.geometric_augmentation_local = RandomCropWithCoords(crop_size = local_crop_size, mean = mean, std = std)
    
    
    def __call__(self, index, image, tactile_data, tactile_mat_label, tactile_corr_data, tactile_3d_location_data, global_scaned_pc_path, local_tactile_pc_path, inst_idx):
        output = {}
        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = im1_base

        # im2_base = self.geometric_augmentation_global(image)
        # global_crop_2 = im2_base
        output["global_crops"] = [global_crop_1,]
        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1,]
        local_crops = []
        # local_corrps_tactiles = []
        local_all_tactiles = []
        local_one_corrps_tactiles = []
        local_mat_labels = []
        local_one_mat_labels = []
        local_patch_indices = [] # tactile in local crops patch idx
        global_mat_labels = []
        
        coor_list = torch.tensor(list(tactile_corr_data.values()))
        
        # for _ in range(self.local_crops_number):
        # get local crops
        local_image, corrdinate = self.geometric_augmentation_local(image)
        # local_image = self.local_transfo(local_image)
        local_crops.append(local_image)
        # get local tactile corresponding indices
        
        face_indices = get_indices_in_region_torch(coor_list, torch.tensor(corrdinate))
        if isinstance(face_indices, torch.Tensor):
            if face_indices.numel() == 0:
                face_indices = []
            else:
                face_indices = face_indices.tolist()
        
        if isinstance(face_indices, (int, float)):
            face_indices = [face_indices,]
        
        idx = 0
        
        while len(face_indices) != 1:
            local_image, corrdinate = self.geometric_augmentation_local(image)
            # local_image = self.local_transfo(local_image)
            face_indices = get_indices_in_region_torch(coor_list, torch.tensor(corrdinate))
            idx += 1
            if idx == 5:
                face_indices = get_index_closest_to_center(coor_list, torch.tensor(corrdinate))
            
            
            if isinstance(face_indices, torch.Tensor):
                if face_indices.numel() == 0:
                    face_indices = []
                elif face_indices.ndim == 0:
                    face_indices = [face_indices.item()]
                else:
                    face_indices = face_indices.tolist()
            
            if isinstance(face_indices, (int, float)):
                face_indices = [face_indices,]
            
        sel_indices = random.choice(face_indices)
        
        local_one_tactile = tactile_data[sel_indices]
        local_one_corrps_tactiles.append(self.local_transfo(local_one_tactile))
        sel_name = f"tactile_{sel_indices}.mp4"
        if sel_name not in tactile_3d_location_data.keys():
            print(tactile_3d_location_data, sel_name, local_tactile_pc_path)
            exit()
        sel_tactile_3d_location_data = tactile_3d_location_data[sel_name]
        sel_tactile_3d_location_data = torch.tensor(sel_tactile_3d_location_data)
        local_tactile_pc_path = os.path.join(local_tactile_pc_path, f"tactile_3d_{int(sel_indices)}", f"contact_{int(sel_indices)}_10.ply")
        
        one_mat = tactile_mat_label[sel_indices]
        one_one_hot_tactile_label = [0 for _ in range(15)]
        one_one_hot_tactile_label[one_mat] = 1
        local_one_mat_labels.append(torch.tensor(one_one_hot_tactile_label))
        
        # get all local tactile
        subset_corrds = [coor_list[i].tolist() for i in face_indices]
        tactile_patch_index = get_patch_index(subset_corrds, corrdinate, self.geometric_augmentation_local.flip_state, img_size=16, num_patches_side=16)
        local_patch_indices.append(tactile_patch_index.tolist())
        
        # get all material labels
        subset_mat = [tactile_mat_label[i] for i in face_indices]
        one_hot_tactile_label = [0 for _ in range(15)]
        for each_idx in subset_mat:
            one_hot_tactile_label[each_idx] = 1
        
        local_mat_labels.append(torch.tensor(one_hot_tactile_label))
        
        # get all tactile image data
        # subset_tactile_data = [tactile_data[i] for i in face_indices]
        # for each_tactile_data in subset_tactile_data:
        #     local_corrps_tactiles.append(self.local_transfo(each_tactile_data))
        
        # get global material labels in current views
        
        global_one_hot_tactile_label = [0 for _ in range(15)]
        for each_idx in tactile_mat_label:
            if each_idx != -1:
                global_one_hot_tactile_label[each_idx] = 1
        global_mat_labels.append(torch.tensor(global_one_hot_tactile_label))
        
        # get all tactiles
        for each_tactile_data in tactile_data:
            local_all_tactiles.append(self.local_transfo(each_tactile_data))
        
        output["global_mat_labels"] = global_mat_labels
        output["global_scaned_pc_path"] = global_scaned_pc_path
        output["local_crops"] = local_crops
        # output["local_tactile_crops"] = torch.stack(local_corrps_tactiles)
        output["local_mat_labels"] = local_mat_labels
        output["local_patch_indices"] = local_patch_indices
        output["local_one_corrps_tactiles"] = local_one_corrps_tactiles
        output["local_one_mat_labels"] = local_one_mat_labels
        output["local_all_tactiles"] = torch.stack(local_all_tactiles)
        output["local_tactile_pc_path"] = local_tactile_pc_path
        output["pc_contact_corrds"] = subset_corrds
        output["tactile_3d_location_data"] = sel_tactile_3d_location_data
        # output["local_all_coords"] = local_all_coords
        output["index"] = index
        output["inst_index"] = inst_idx
        output["offsets"] = ()
        return output


class LinearTacDataAugmentationDINO(object):
    def __init__(
        self,
        local_crops_scale = (0.05, 0.3),
        crop_size: int = 224,
        local_crop_size: int = 98,
        local_crops_number:int = 1,
        interpolation=transforms.InterpolationMode.BICUBIC,
        hflip_prob: float = 0.5,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
        local_data="tactile"
    ):  
        self.local_crops_number = local_crops_number
        self.local_data = local_data
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {crop_size}")
        logger.info(f"local_crops_scale: {local_crop_size}")
        logger.info(f"mean: {mean}")
        logger.info(f"std: {std}")
        logger.info("###################################")

        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(hflip_prob),
                # MaybeToTensor(),
                # make_normalize_transform(mean=mean, std=std),
            ]
        )
        
        
        self.geometric_augmentation_local = RandomResizedCropAndFlipWithCoords(size = local_crop_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC, flip_p=0.5)
        
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
        self.totensor = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.local_transfo = transforms.Compose([self.totensor])
        
    def __call__(self, index, image, tactile_data,tactile_mat_label, tactile_corr_data, tactile_3d_location_data, global_scaned_pc_path, local_tactile_pc_path, inst_idx):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]
        local_crops = []
        local_corrps_tactiles = []
        local_all_coords = []
        
        coor_list = torch.tensor(list(tactile_corr_data.values()))
        
        for _ in range(self.local_crops_number):
            # get local image 
            local_image, corrdinate = self.geometric_augmentation_local(image)
            local_image = self.local_transfo(local_image)
            # get tactile indices in local image region
            face_indices = get_indices_in_region_torch(coor_list, torch.tensor(corrdinate))
            
            if isinstance(face_indices, torch.Tensor):
                if face_indices.numel() == 0:
                    face_indices = []
                else:
                    face_indices = face_indices.tolist()
            
            if isinstance(face_indices, (int, float)):
                face_indices = [face_indices,]
                
            if self.local_data in ["tac_patchalign", "vis_patchalign","vistac_patchalign"]:
                # one tactile corresponding to one local vision
                idx = 0
                
                while len(face_indices) != 1:
                    local_image, corrdinate = self.geometric_augmentation_local(image)
                    local_image = self.local_transfo(local_image)
                    face_indices = get_indices_in_region_torch(coor_list, torch.tensor(corrdinate))
                    idx += 1
                    if idx == 5:
                        face_indices = get_index_closest_to_center(coor_list, torch.tensor(corrdinate))
                    
                    
                    if isinstance(face_indices, torch.Tensor):
                        if face_indices.numel() == 0:
                            face_indices = []
                        elif face_indices.ndim == 0:
                            face_indices = [face_indices.item()]
                        else:
                            face_indices = face_indices.tolist()
                    
                    if isinstance(face_indices, (int, float)):
                        face_indices = [face_indices,]
                        
            elif self.local_data in ["tactile", "vistac_patchfusion"]:
                idx = 0
                while len(face_indices) == 0:
                    local_image, corrdinate = self.geometric_augmentation_local(image)
                    local_image = self.local_transfo(local_image)
                    face_indices = get_indices_in_region_torch(coor_list, torch.tensor(corrdinate))
                    idx += 1
                    if idx == 5:
                        face_indices = get_index_closest_to_center(coor_list, torch.tensor(corrdinate))

                    if isinstance(face_indices, torch.Tensor):
                        if face_indices.numel() == 0:
                            face_indices = []
                        elif face_indices.ndim == 0:
                            face_indices = [face_indices.item()]
                        else:
                            face_indices = face_indices.tolist()
                    
                    if isinstance(face_indices, (int, float)):
                        face_indices = [face_indices,]
                        
                        
            local_crops.append(local_image)
            
            if self.local_data in ["tactile", "tac_patchalign", "vis_patchalign", "tac_globalalign", "vis_globalalign", "vistac_patchalign"]:
                sel_indices = random.choice(face_indices)
                local_one_tactile = self.local_transfo(tactile_data[sel_indices])
                local_corrps_tactiles.append(local_one_tactile)
        
            # all tactile
            if self.local_data == "vistac_patchfusion":
                subset_corrds = [coor_list[i].tolist() for i in face_indices]
                tactile_patch_index = get_patch_index(subset_corrds, corrdinate, self.geometric_augmentation_local.flip_state, img_size=16, num_patches_side=16)
                local_all_coords.append(tactile_patch_index.tolist())
        
        output_tactile_feature = []
        for each_tactile_data in tactile_data:
            output_tactile_feature.append(self.local_transfo(each_tactile_data))
        
        # raw local vision data
        output["local_crops"] = local_crops
        if self.local_data in ["tactile", "vistac_patchalign", "tac_patchalign", "tac_globalalign", "vis_globalalign", "vis_patchalign"]:
            # raw tactile image (only one tactile & random select one tactile in local vision region)
            output["local_tactile_crops"] = local_corrps_tactiles
        
        if self.local_data == "vistac_patchfusion":
            # all tactiles in complete RGB views.
            output["local_all_tactiles"] = torch.stack(output_tactile_feature)
            output["local_all_coords"] = local_all_coords
        
        output["inst_idx"] = inst_idx
        output["offsets"] = ()

        return output
        

def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)




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