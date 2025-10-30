#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" MVFoul dataset. """

import json
import os
import cv2
import torch
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


    def _config_transform(self):
        pass

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
        return selected.permute(3, 0, 1, 2).unsqueeze(0).float()/ 255.0


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
