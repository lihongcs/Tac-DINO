# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified for Tac-DINO.

import json
from typing import Any, Tuple

from torchvision.datasets import VisionDataset

from .decoders import DecoderType, TargetDecoder


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        image_decoder_type = kwargs.pop("image_decoder_type", DecoderType.ImageDataDecoder)
        self._decoder_params = {}
        self._image_decoder_class = image_decoder_type.get_class()
        if "image_decoder_params" in kwargs:
            self._decoder_params = kwargs.pop("image_decoder_params")

        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = self._image_decoder_class(image_data, **self._decoder_params).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
    

class TacExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        image_decoder_type = kwargs.pop("image_decoder_type", DecoderType.ImageDataDecoder)
        self._decoder_params = {}
        self._image_decoder_class = image_decoder_type.get_class()
        if "image_decoder_params" in kwargs:
            self._decoder_params = kwargs.pop("image_decoder_params")

        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data, tactile_path, tactile_mat_label, tactile_corr_data, tactile_3d_location_data, global_scaned_pc_path, local_tactile_pc_path, class_id, inst_idx = self.get_image_data(index)
            image = self._image_decoder_class(image_data, **self._decoder_params).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(class_id)
        target = TargetDecoder(target).decode()
        
        if self.transforms is not None:
            if self.transforms.transform is not None:
                if getattr(self.transforms.transform, "needs_rgb_source_path", False):
                    image.info["tacdino_rgb_path"] = self.get_image_path(index)
                if getattr(self.transforms.transform, "defer_tactile_loading", False):
                    # Temporal methods first select a small, unique contact
                    # set, then decode only those videos inside the worker.
                    # Legacy methods retain eager canonical-frame decoding.
                    tactile_raw_feature = tactile_path
                else:
                    tactile_raw_feature = self.get_tactile_mp4(tactile_path)
                image = self.transforms.transform(index, image, tactile_raw_feature, tactile_mat_label, tactile_corr_data, tactile_3d_location_data, global_scaned_pc_path, local_tactile_pc_path, inst_idx)
                
            if self.transforms.target_transform is not None:
                target = self.transforms.target_transform(target)
            # image, target = self.transforms(image, tactile_path, tactile_corr_data)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError
