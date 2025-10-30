#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" MVFoul dataset. """

import json
import os
import cv2
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torch
from tadaconv.datasets.utils.random_erasing import RandomErasing
from tadaconv.datasets.utils.transformations import ColorJitter, KineticsResizedCrop
import tadaconv.utils.logging as logging
from tadaconv.datasets.base.builder import DATASET_REGISTRY
from tadaconv.utils.mvfoul_translation import translate_annotation
from utils import crop_center

logger = logging.get_logger(__name__)



@DATASET_REGISTRY.register()
class Mvfoul(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super(Mvfoul, self).__init__() 
        self.cfg = cfg
        self.split = split
        self.data_root_dir  = cfg.DATA.DATA_ROOT_DIR
        self._construct_dataset()
        self._config_transform()

    
    def _construct_dataset(self):
        if self.split == "train":
            self.crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            self.crop_size = self.cfg.DATA.TEST_CROP_SIZE


        path = os.path.join(
            self.data_root_dir,
            self.split)
        assert os.path.exists(path), "{} does not exist".format(path)
        self.data_root_dir = path

        entries = sorted(os.listdir(self.data_root_dir))

        with open(os.path.join(self.data_root_dir, "annotations.json"), "r") as f:
            self.annotations = json.load(f)["Actions"]
        
        self.labels: list = self._process_labels(self.annotations)

        self.dirs = [e for e in entries if os.path.isdir(os.path.join(self.data_root_dir, e))]
        
        self.meta_data = {e: len(os.listdir(os.path.join(self.data_root_dir, e))) for e in self.dirs}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        dir_name = self.dirs[index]
        feature = self._read_videos_from_dir(os.path.join(self.data_root_dir, dir_name))
        return feature[0], feature[1], self.labels[index]

    def _read_videos_from_dir(self, dir_path):
        video_files = sorted(os.listdir(dir_path))
        video_tensors = []
        for vf in video_files:
            video_path = os.path.join(dir_path, vf)
            video_tensor = self._load_video(video_path)
            video_tensors.append(video_tensor)
            if len(video_tensors) == self.cfg.DATA.NUM_VIEWS:
                break
        actual_views = len(video_tensors)
        if actual_views < self.cfg.DATA.NUM_VIEWS:
            # pad with empty tensors
            num_missing = self.cfg.DATA.NUM_VIEWS - actual_views
            C, T, H, W = video_tensors[0].shape
            for _ in range(num_missing):
                video_tensors.append(torch.zeros((C, T, H, W)))
        video_tensor = torch.cat(video_tensors, dim=0)  #(num_views, C, T, H, W)

        mask = torch.arange(self.cfg.DATA.NUM_VIEWS)
        mask[actual_views:] = -1

        return video_tensor, mask
    
    def _load_video(self, video_path):
        vid = cv2.VideoCapture(video_path)

        count, success = 0, True
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        height, width = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

        indices = self._custom_sampling(
            vid_length=num_frames,
            vid_fps=fps,
            num_frames=self.cfg.DATA.NUM_INPUT_FRAMES,
            interval=2,
            height=height,
            width=width,
        )
        indices = torch.linspace(0, num_frames - 1, steps=16).tolist()
        frames = []
        for idx in indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = vid.read()
            if success:
                frames.append(crop_center(frame, self.crop_size, self.crop_size))


        vid.release()
        selected = torch.stack([torch.from_numpy(frame) for frame in frames])
        
        if not self.cfg.PRETRAIN.ENABLE:
            selected = self.transform(selected)  #  C, T, H, W
        else:
            selected = selected.permute(3,0,1,2)
        return selected.unsqueeze(0)


    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                "path": indicating the target's path w.r.t. index
                "supervised_label": indicating the class of the target 
        """
        sample_info = {}
        return sample_info

    def _custom_sampling(self, vid_length,
            vid_fps,
            num_frames,
            interval,
            height,
            width):
        indices = torch.linspace(0, vid_length - 1, steps=num_frames).long()
        return indices
    
    def _process_labels(self, annotations):
        labels = []
        for idx, annotation in annotations.items():
            label = translate_annotation(annotation)
            labels.append(label)
        return labels
    

    def _config_transform(self):
        """
        Configs the transform for the dataset.
        For train, we apply random cropping, random horizontal flip, random color jitter (optionally),
            normalization and random erasing (optionally).
        For val and test, we apply controlled spatial cropping and normalization.
        The transformations are stored as a callable function to "self.transforms".
        
        Note: This is only used in the supervised setting.
            For self-supervised training, the augmentations are performed in the 
            corresponding generator.
        """
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo()
            ]
            
            if self.cfg.DATA.TRAIN_JITTER_SCALES[0] <= 1:
                std_transform_list += [transforms.RandomResizedCropVideo(
                        size=self.cfg.DATA.TRAIN_CROP_SIZE,
                        scale=[
                            self.cfg.DATA.TRAIN_JITTER_SCALES[0],
                            self.cfg.DATA.TRAIN_JITTER_SCALES[1]
                        ],
                        ratio=self.cfg.AUGMENTATION.RATIO
                    ),]
            else:
                std_transform_list += [KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]
            if self.cfg.AUGMENTATION.AUTOAUGMENT.ENABLE:
                from tadaconv.datasets.utils.auto_augment import creat_auto_augmentation
                std_transform_list.append(creat_auto_augmentation(self.cfg.AUGMENTATION.AUTOAUGMENT.TYPE, self.cfg.DATA.TRAIN_CROP_SIZE, self.cfg.DATA.MEAN))
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        color=self.cfg.AUGMENTATION.COLOR_P,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        ),
                )
            std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
            ]
            self.transform = Compose(std_transform_list)
        elif self.split == 'val' or self.split == 'test':
            self.resize_video = KineticsResizedCrop(
                    short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)
