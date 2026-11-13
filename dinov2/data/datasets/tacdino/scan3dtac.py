# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified for Tac-DINO.

import os
import cv2
import csv
import json
import time
import random
import re
import logging
import numpy as np
from PIL import Image
from enum import Enum
from ..extended import TacExtendedVisionDataset
from typing import Callable, List, Optional, Tuple, Union
from .contact_alignment import (
    align_contact_view,
    index_contact_mapping,
    usable_image_names,
)


logger = logging.getLogger("dinov2")
_Target = int


def _natural_tactile_key(path):
    """Keep tactile videos aligned with correspondence/material index order."""
    match = re.search(r"tactile_(\d+)", os.path.basename(path))
    return (int(match.group(1)) if match else 10**12, os.path.basename(path))


def _sort_contact_mapping(mapping, source_path):
    return {str(index): value for index, value in index_contact_mapping(mapping, source_path).items()}


def load_random_frame_from_mp4(mp4_path, rng=None, max_retries=20, frame_index=10):
    
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"Video path does not exist: {mp4_path}")

    cap = None
    is_opened = False
    
    for try_idx in range(max_retries):
        cap = cv2.VideoCapture(mp4_path)
        if cap.isOpened():
            is_opened = True
            break
        
        cap.release()
        time.sleep(0.1 * (try_idx + 1))

    if not is_opened:
        raise IOError(f"Cannot open video after {max_retries} retries: {mp4_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Video has invalid frame count ({frame_count}): {mp4_path}")

        if frame_index is None:
            rng = rng or random
            frame_idx = rng.randrange(frame_count)
        else:
            frame_idx = int(frame_index)
            if frame_idx < 0 or frame_idx >= frame_count:
                raise ValueError(
                    f"Canonical frame {frame_idx} is unavailable (frame_count={frame_count}): {mp4_path}"
                )
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        
        if not ok or frame is None or frame.size == 0:
            raise IOError(f"Failed to read canonical frame {frame_idx} from {mp4_path}")
        frame = cv2.resize(frame, (98, 98))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        
        return pil_image

    except Exception as e:
        raise RuntimeError(f"Error processing video {mp4_path}: {str(e)}") from e
        
    finally:
        if cap is not None:
            cap.release()


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 5_662,
            _Split.VAL: 3_724,
            _Split.TEST: 3_724,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class Scan3dTac(TacExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "Scan3dTac.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split_file: Optional[str] = None,
        no3d: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        self.root = root
        self.no3d = bool(no3d)
        self._split_file = (
            None
            if split_file is None
            else os.path.abspath(split_file)
            if os.path.isabs(split_file)
            else os.path.abspath(os.path.join(self.root, split_file))
        )

        self._entries = None
        self._class_ids = None
        self._class_names = None
        self.mat2idx = json.load(open(os.path.join(self.root, "mat2idx.json")))
        self.inst2idx = json.load(open(os.path.join(self.root, "catemap", "label_to_idx.json")))
        self._contact_filter = {}
        contact_filter_path = os.environ.get("TACDINO_CONTACT_FILTER")
        if contact_filter_path:
            if not os.path.isfile(contact_filter_path):
                raise FileNotFoundError(f"TACDINO_CONTACT_FILTER does not exist: {contact_filter_path}")
            filter_data = json.load(open(contact_filter_path, "r"))
            self._contact_filter = {
                str(instance): set(int(value) for value in contact_ids)
                for instance, contact_ids in filter_data.get("excluded_contact_ids", {}).items()
            }

    def _excluded_contacts(self, instance):
        return self._contact_filter.get(str(instance), set())

    @property
    def split(self) -> "Scan3dTac.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return json.load(open(extra_full_path, "r"))

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        if self._split_file is not None:
            return self._split_file
        return os.path.join(self.root, "split0111", f"{self._split.value.lower()}_split.json")

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)

            label_to_idx_dict  = json.load(open(os.path.join(self.root, "catemap", "label_to_idx.json")))
            
            out_entries = []
            sum_idx = 0
            for each_entries in self._entries:
                data_dir = os.path.join(self._extra_root, each_entries, "Preprocessed0111", "tacdino", "rgb_images_processed")
                assert os.path.isdir(data_dir)
                label = each_entries.split("/")[0]
                
                label_idx = label_to_idx_dict[label]
                
                image_lists = sorted(
                    each_image for each_image in os.listdir(data_dir) if each_image.endswith(".png")
                )

                instance_root = os.path.join(self._extra_root, each_entries)
                image_lists = usable_image_names(
                    instance_root,
                    image_lists,
                    excluded_contact_ids=self._excluded_contacts(each_entries),
                    minimum_visible_contacts=int(
                        getattr(
                            getattr(
                                getattr(self, "transforms", None),
                                "transform",
                                None,
                            ),
                            "minimum_visible_contacts_per_view",
                            1,
                        )
                    ),
                )
                
                for each_image in image_lists:
                    assert os.path.exists(os.path.join(data_dir, each_image))
                    out_entries.append([each_entries, each_image, label_idx])
                    
                    sum_idx += 1
                    
            self._entries = out_entries
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._split == _Split.TEST:
            assert False, "Class IDs are not available in TEST split"
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._split == _Split.TEST:
            assert False, "Class names are not available in TEST split"
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_path(self, index: int) -> str:
        actual_index, image_index, _ = self._get_entries()[index]
        return os.path.join(self._extra_root, actual_index, "Preprocessed0111", "tacdino", "rgb_images_processed", image_index)

    def get_image_data(self, index: int) -> bytes:
        
        entries = self._get_entries()
        actual_index, image_index, class_id = entries[index]
        
        tactile_path = os.path.join(self._extra_root,actual_index, "Tactile")
        tactile_material_path = os.path.join(self._extra_root,actual_index, "Annotation", "tactile_material.json")
        
        # inst_name = actual_index.replace("/", "-")
        inst_name = actual_index.split("/")[0]
        
        inst_idx = self.inst2idx[inst_name]
        
        clean_img_name = image_index.split(".")[0]
        tactile_corr_path =  os.path.join(self._extra_root, actual_index, "Preprocessed0111", "tacdino", "tactile_corr_label", f"{clean_img_name}_tactile_correspondence.json")
        contact_ids, tactile_mat_data, tactile_corr_data, tactile_path = align_contact_view(
            json.load(open(tactile_material_path, "r")),
            json.load(open(tactile_corr_path, "r")),
            tactile_path,
            tactile_material_path,
            tactile_corr_path,
        )
        excluded = self._excluded_contacts(actual_index)
        if excluded:
            keep_indices = [index for index, contact_id in enumerate(contact_ids) if contact_id not in excluded]
            contact_ids = [contact_ids[index] for index in keep_indices]
            tactile_mat_data = [tactile_mat_data[index] for index in keep_indices]
            tactile_path = [tactile_path[index] for index in keep_indices]
            tactile_corr_data = {
                str(contact_ids[new_index]): tactile_corr_data[str(contact_ids[new_index])]
                for new_index in range(len(contact_ids))
            }
        if not contact_ids:
            raise ValueError(f"all contacts were excluded by the physical quality manifest: {actual_index}")

        try:
            tactile_mat_idx = [self.mat2idx[each_mat.lower()] for each_mat in tactile_mat_data]
        except KeyError as e:
            print(f"{tactile_material_path} existing error!")
            raise ValueError(f"Material '{e}' not found in mat2idx mapping.")
        
        transform = getattr(getattr(self, "transforms", None), "transform", None)
        requires_3d = not self.no3d and bool(getattr(transform, "requires_3d", True))
        if requires_3d:
            global_scaned_pc_path = os.path.join(
                self._extra_root,
                actual_index,
                "Preprocessed0111",
                "scan.npy",
            )
            local_tactile_pc_path = os.path.join(
                self._extra_root,
                actual_index,
                "Tactile",
                f"tactile_3d_{image_index}",
                f"contact_{image_index}_10.npy",
            )
        else:
            # The no-3D contract does not even construct scan/point-cloud input
            # paths, so a geometry-free dataset can omit those files entirely.
            global_scaned_pc_path = None
            local_tactile_pc_path = None
        
        for tactile_idx, tactile_corrds in enumerate(list(tactile_corr_data.values())):
            if tactile_corrds == [0,0]:
                tactile_mat_idx[tactile_idx] = -1
        
        image_full_path = os.path.join(self._extra_root, actual_index, "Preprocessed0111", "tacdino", "rgb_images_processed", image_index)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        
        if requires_3d:
            tactile_3d_location_path = os.path.join(
                self._extra_root,
                actual_index,
                "Annotation",
                "tactile_normalized_pc_location.json",
            )
            tactile_3d_location_data = (
                json.load(open(tactile_3d_location_path, "r"))
                if os.path.isfile(tactile_3d_location_path)
                else None
            )
        else:
            tactile_3d_location_data = None
        
        return image_data, tactile_path, tactile_mat_idx, tactile_corr_data, tactile_3d_location_data, global_scaned_pc_path, local_tactile_pc_path, class_id, inst_idx

    
    def get_tactile_mp4(self, tactile_dir: str) -> [List]:
        if isinstance(tactile_dir, (list, tuple)):
            return [load_random_frame_from_mp4(path, frame_index=10) for path in tactile_dir]
        tactile_file_list = sorted(
            (
                each_tac
                for each_tac in os.listdir(tactile_dir)
                if each_tac.startswith("tactile") and each_tac.endswith(".mp4")
            ),
            key=_natural_tactile_key,
        )
        
        tactile_raw_data = []
        for each_tac in tactile_file_list:
            tac_path = os.path.join(tactile_dir, each_tac)
            # Frame 10 is the canonical contact state used by evaluation and
            # by the corresponding contact_*_10 point-cloud naming.
            tactile_raw_data.append(load_random_frame_from_mp4(tac_path, frame_index=10))
        
        return tactile_raw_data

    def get_target(self, class_id: int) -> Optional[Target]:
        # entries = self._get_entries()
        # class_index = entries[index]["class_index"]
        return None if self.split == _Split.TEST else int(class_id)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return None if self.split == _Split.TEST else entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return None if self.split == _Split.TEST else str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        return None if self.split == _Split.TEST else str(class_name)

    def __len__(self) -> int:
        entries = self._get_entries()
        # print(len(entries), self.split.length)
        # assert len(entries) == self.split.length
        return len(entries)

    def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
        labels_full_path = os.path.join(self.root, labels_path)
        labels = []

        try:
            with open(labels_full_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e

        return labels

    def _dump_entries(self) -> None:
        split = self.split
        if split == Scan3dTac.Split.TEST:
            dataset = None
            sample_count = split.length
            max_class_id_length, max_class_name_length = 0, 0
        else:
            labels_path = "labels.txt"
            logger.info(f'loading labels from "{labels_path}"')
            labels = self._load_labels(labels_path)

            # NOTE: Using torchvision ImageFolder for consistency
            from torchvision.datasets import ImageFolder

            dataset_root = os.path.join(self.root, split.get_dirname())
            dataset = ImageFolder(dataset_root)
            sample_count = len(dataset)
            max_class_id_length, max_class_name_length = -1, -1
            for sample in dataset.samples:
                _, class_index = sample
                class_id, class_name = labels[class_index]
                max_class_id_length = max(len(class_id), max_class_id_length)
                max_class_name_length = max(len(class_name), max_class_name_length)

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        if split == Scan3dTac.Split.TEST:
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                actual_index = index + 1
                class_index = np.uint32(-1)
                class_id, class_name = "", ""
                entries_array[index] = (actual_index, class_index, class_id, class_name)
        else:
            class_names = {class_id: class_name for class_id, class_name in labels}

            assert dataset
            old_percent = -1
            for index in range(sample_count):
                percent = 100 * (index + 1) // sample_count
                if percent > old_percent:
                    logger.info(f"creating entries: {percent}%")
                    old_percent = percent

                image_full_path, class_index = dataset.samples[index]
                image_relpath = os.path.relpath(image_full_path, self.root)
                class_id, actual_index = split.parse_image_relpath(image_relpath)
                class_name = class_names[class_id]
                entries_array[index] = (actual_index, class_index, class_id, class_name)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:
        split = self.split
        if split == Scan3dTac.Split.TEST:
            return

        entries_array = self._load_extra(self._entries_path)

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_class_ids_and_names()
