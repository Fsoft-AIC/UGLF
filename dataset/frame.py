#!/usr/bin/env python3

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from util.io import load_json
from .transform import RandomGaussianNoise, RandomHorizontalFlipFLow, \
    RandomOffsetFlow, SeedableRandomSquareCrop, ThreeCrop


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FrameReader:

    IMG_NAME = '{:06d}.jpg'

    def __init__(self, frame_dir, modality, crop_transform, img_transform,
                 same_transform):
        self._frame_dir = frame_dir
        self._is_flow = modality == 'flow'
        self._crop_transform = crop_transform
        self._img_transform = img_transform
        self._same_transform = same_transform

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path).float() / 255
        if self._is_flow:
            img = img[1:, :, :]     # GB channels contain data
        return img

    def load_frames(self, video_name, start, end, pad=False, stride=1,
                    randomize=False):
        rand_crop_state = None
        rand_state_backup = None
        ret = []
        frame_num_list = []
        n_pad_start = 0
        n_pad_end = 0
        for frame_num in range(start, end, stride):
            if randomize and stride > 1:
                frame_num += random.randint(0, stride - 1)

            if frame_num < 0:
                n_pad_start += 1
                continue

            frame_path = os.path.join(
                self._frame_dir, video_name,
                FrameReader.IMG_NAME.format(frame_num))
            try:
                img = self.read_frame(frame_path)
                if self._crop_transform:
                    if self._same_transform:
                        if rand_crop_state is None:
                            rand_crop_state = random.getstate()
                        else:
                            rand_state_backup = random.getstate()
                            random.setstate(rand_crop_state)

                    img = self._crop_transform(img)

                    if rand_state_backup is not None:
                        # Make sure that rand state still advances
                        random.setstate(rand_state_backup)
                        rand_state_backup = None

                if not self._same_transform:
                    img = self._img_transform(img)
                ret.append(img)
                frame_num_list.append(frame_num)
            except RuntimeError:
                # print('Missing file!', frame_path)
                n_pad_end += 1

        # In the multicrop case, the shape is (B, T, C, H, W)
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))
        if self._same_transform:
            ret = self._img_transform(ret)

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
            
        # Pad for frame num list
        for _ in range(n_pad_start):
            frame_num_list.insert(-1, 0)
        for _ in range(n_pad_end):
            frame_num_list.append(-1)
        return ret, frame_num_list


# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5


def _get_deferred_rgb_transform():
    img_transforms = [
        # Jittering separately is faster (low variance)
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(saturation=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(brightness=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(contrast=(0.7, 1.2))
            ]), p=0.25),

        # Jittering together is slower (high variance)
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.ColorJitter(
        #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
        #             saturation=(0.7, 1.2), hue=0.2)
        #     ]), 0.8),

        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _get_deferred_bw_transform():
    img_transforms = [
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(brightness=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(contrast=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        RandomGaussianNoise()
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _load_frame_deferred(gpu_transform, batch, device):
    frame = batch['frame'].to(device)
    with torch.no_grad():
        for i in range(frame.shape[0]):
            frame[i] = gpu_transform(frame[i])

        if 'mix_weight' in batch:
            weight = batch['mix_weight'].to(device)
            frame *= weight[:, None, None, None, None]

            frame_mix = batch['mix_frame']
            for i in range(frame.shape[0]):
                frame[i] += (1. - weight[i]) * gpu_transform(
                    frame_mix[i].to(device))
    return frame


def _get_img_transforms(
        is_eval,
        crop_dim,
        modality,
        same_transform,
        defer_transform=False,
        multi_crop=False
):
    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = transforms.CenterCrop(crop_dim)
        elif same_transform:
            print('=> Using seeded crops!')
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        else:
            crop_transform = transforms.RandomCrop(crop_dim)

    img_transforms = []
    if modality == 'rgb':
        if not is_eval:
            img_transforms.append(
                transforms.RandomHorizontalFlip())

            if not defer_transform:
                img_transforms.extend([
                    # Jittering separately is faster (low variance)
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(hue=0.2)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(saturation=(0.7, 1.2))
                        ]), p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(brightness=(0.7, 1.2))
                        ]), p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(contrast=(0.7, 1.2))
                        ]), p=0.25),

                    # Jittering together is slower (high variance)
                    # transforms.RandomApply(
                    #     nn.ModuleList([
                    #         transforms.ColorJitter(
                    #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
                    #             saturation=(0.7, 1.2), hue=0.2)
                    #     ]), p=0.8),

                    transforms.RandomApply(
                        nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25)
                ])

        if not defer_transform:
            img_transforms.append(transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD))
    elif modality == 'bw':
        if not is_eval:
            img_transforms.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25)])
        img_transforms.append(transforms.Grayscale())

        if not defer_transform:
            if not is_eval:
                img_transforms.extend([
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(brightness=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(contrast=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
                ])

            img_transforms.append(transforms.Normalize(
                mean=[0.5], std=[0.5]))

            if not is_eval:
                img_transforms.append(RandomGaussianNoise())
    elif modality == 'flow':
        assert not defer_transform

        img_transforms.append(transforms.Normalize(
            mean=[0.5, 0.5], std=[0.5, 0.5]))

        if not is_eval:
            img_transforms.extend([
                RandomHorizontalFlipFLow(),
                RandomOffsetFlow(),
                RandomGaussianNoise()
            ])
    else:
        raise NotImplementedError(modality)

    img_transform = torch.jit.script(nn.Sequential(*img_transforms))
    return crop_transform, img_transform


def load_glip(glip_dir, video_name, frame_num_list, max_object = 50):
    file_name = os.path.join(glip_dir, video_name+'.pt')
    df = torch.load(file_name, map_location='cpu')
    
    frame_num_list = torch.Tensor(frame_num_list)
    frame_num = df[:, 0]
    mask = (((frame_num.view(-1, 1) - frame_num_list.view(-1)) == 0).sum(dim=-1))!=0
    keep = df[mask]

    feat_dict = {}
    feat_size = 256
    for row in keep:
        frame_idx = int(row[0].item())
        class_id = int(row[1].item())
        boxes = row[2:6]
        feat = row[6:]

        if (class_id == 0):
            continue

        if (frame_idx not in feat_dict):
            feat_dict[frame_idx] = []

        feat_dict[frame_idx].append({
            'frame': frame_idx,
            'class': class_id,
            'boxes': boxes,
            'feature': feat
        })   

    # Output for feature
    # Feature size: Frames x Max_objects x Feat_size
    # Frames: Number of frame fetched
    # Max_objects: The number of  objects (after padding)
    # Feat_size: The dimension of features
    # ----------------------------------------------------
    # Output for padding mask
    # Feature size: Frames x Max_objects
    # Frames: Number of frames fetched
    # Max_objects: Bit mask to keep or not
    ret = []
    mask = []
    for frame_num in frame_num_list:
        frame_feat = []
        num = int(frame_num.item())

        # Have object
        if (num in feat_dict):

            # Get all objects
            ls_obj = feat_dict[num]
            n_object = len(ls_obj)
        
            # Handle object exceed max objects
            assert n_object <= max_object, f'GLIP objects are exceeded at {glip_dir} (frame {num})'

            # Append objects
            for obj in ls_obj:
                frame_feat.append(obj['feature'])
        else:
            n_object = 0

        # Append nothing to ensure the size
        for _ in range(max_object - n_object):
            frame_feat.append(torch.zeros(feat_size))
        frame_feat = torch.stack(frame_feat)

        # Create mask
        frame_mask = torch.concat(
            (torch.ones(n_object), torch.zeros(max_object - n_object))
        )
        
        ret.append(frame_feat)
        mask.append(frame_mask)
    ret = torch.stack(ret)
    mask = torch.stack(mask)
    return ret, mask

def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        num_events = sum([len(x['events']) for x in labels])
        print('{} : {} videos, {} frames, {:0.5f}% non-bg'.format(
            src_file, len(labels), num_frames,
            num_events / num_frames * 100))


IGNORED_NOT_SHOWN_FLAG = False


class ActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,                    # dict of class names to idx
            label_file,                 # path to label json
            frame_dir,                  # path to frames
            modality,                   # [rgb, bw, flow]
            clip_len,
            dataset_len,                # Number of clips
            is_eval=True,               # Disable random augmentation
            crop_dim=None,
            stride=1,                   # Downsample frame rate
            same_transform=True,        # Apply the same random augmentation to
                                        # each frame in a clip
            dilate_len=0,               # Dilate ground truth labels
            mixup=False,
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            fg_upsample=-1,             # Sample foreground explicitly
            label_type='one_hot',       # Type of label encoding
            glip_dir=None,              # Path to Glip feature
            max_object=35               # Maximum object in GLIP feat
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._mixup = mixup
        self._glip_dir = glip_dir
        self._max_object = max_object

        if (self._glip_dir is not None and self._mixup):
            self._mixup = False
            print("Turn off mixup to use GLIP")

        # Sample videos weighted by their length
        num_frames = [v['num_frames'] for v in self._labels]
        self._weights_by_length = np.array(num_frames) / np.sum(num_frames)

        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0
        self._is_eval = is_eval
        assert label_type in ['integer', 'one_hot']
        self.label_type = label_type

        # Label modifications
        self._dilate_len = dilate_len
        self._fg_upsample = fg_upsample

        # Sample based on foreground labels
        if self._fg_upsample > 0:
            self._flat_labels = []
            for i, x in enumerate(self._labels):
                for event in x['events']:
                    if event['frame'] < x['num_frames']:
                        self._flat_labels.append((i, event['frame']))


        # Try to do defer the latter half of the transforms to the GPU
        self._gpu_transform = None
        if not is_eval and same_transform:
            if modality == 'rgb':
                print('=> Deferring some RGB transforms to the GPU!')
                self._gpu_transform = _get_deferred_rgb_transform()
            elif modality == 'bw':
                print('=> Deferring some BW transforms to the GPU!')
                self._gpu_transform = _get_deferred_bw_transform()

        crop_transform, img_transform = _get_img_transforms(
            is_eval, crop_dim, modality, same_transform,
            defer_transform=self._gpu_transform is not None)

        self._frame_reader = FrameReader(
            frame_dir, modality, crop_transform, img_transform, same_transform)

    def load_frame_gpu(self, batch, device):
        if self._gpu_transform is None:
            frame = batch['frame'].to(device)
        else:
            frame = _load_frame_deferred(self._gpu_transform, batch, device)
        return frame

    def _sample_uniform(self):
        video_meta = random.choices(
            self._labels, weights=self._weights_by_length)[0]

        video_len = video_meta['num_frames']
        base_idx = -self._pad_len * self._stride + random.randint(
            0, max(0, video_len - 1
                       + (2 * self._pad_len - self._clip_len) * self._stride))
        return video_meta, base_idx

    def _sample_foreground(self):
        video_idx, frame_idx = random.choices(self._flat_labels)[0]
        video_meta = self._labels[video_idx]
        video_len = video_meta['num_frames']

        lower_bound = max(
            -self._pad_len * self._stride,
            frame_idx - self._clip_len * self._stride + 1)
        upper_bound = min(
            video_len - 1 + (self._pad_len - self._clip_len) * self._stride,
            frame_idx)

        base_idx = random.randint(lower_bound, upper_bound) \
            if upper_bound > lower_bound else lower_bound

        assert base_idx <= frame_idx
        assert base_idx + self._clip_len > frame_idx
        return video_meta, base_idx

    def _get_one(self):
        if self._fg_upsample > 0 and random.random() >= self._fg_upsample:
            video_meta, base_idx = self._sample_foreground()
        else:
            video_meta, base_idx = self._sample_uniform()

        if (self.label_type=='one_hot'):
            label_shape = (self._clip_len, len(self._class_dict) + 1)
            labels = np.zeros(label_shape, np.float16)
            labels[:, 0] = 1.
        else:
            label_shape = self._clip_len
            labels = np.zeros(label_shape, np.float16)

        for event in video_meta['events']:
            event_frame = event['frame']

            # Index of event in label array
            label_idx = (event_frame - base_idx) // self._stride
            if (label_idx >= -self._dilate_len
                and label_idx < self._clip_len + self._dilate_len
            ):
                label = self._class_dict[event['label']]
                for i in range(
                    max(0, label_idx - self._dilate_len),
                    min(self._clip_len, label_idx + self._dilate_len + 1)
                ):
                    if (self.label_type=='one_hot'):
                        labels[i][label] = 1.
                        labels[i][0] = 0.
                    else:
                        labels[i] = label

        frames, frame_num_list = self._frame_reader.load_frames(
            video_meta['video'], base_idx,
            base_idx + self._clip_len * self._stride, pad=True,
            stride=self._stride, randomize=not self._is_eval)

        glip_feat = None
        glip_mask = None
        if (self._glip_dir is not None):
            glip_feat, glip_mask = load_glip(
                self._glip_dir, video_meta['video'], frame_num_list)

        if (glip_feat is not None):
            ret = {
                'frame': frames, 
                'contains_event': int(np.sum(labels) > 0),
                'glip_feature': glip_feat,  # frame x obj x feat
                'glip_mask': glip_mask,
                'label': labels
            }
        else:
            ret = {
                'frame': frames, 
                'contains_event': int(np.sum(labels) > 0),
                'label': labels
            }
        return ret
        

    def __getitem__(self, unused):
        ret = self._get_one()

        if self._mixup:
            mix = self._get_one()    # Sample another clip
            l = random.betavariate(0.2, 0.2)
            label_dist = np.zeros((self._clip_len, len(self._class_dict) + 1))

            if (self.label_type=='one_hot'):
                label_dist = ret['label']*l
                label_dist += (1. - l)*mix['label']
            else:
                label_dist[range(self._clip_len), ret['label']] = l
                label_dist[range(self._clip_len), mix['label']] += 1. - l

            if self._gpu_transform is None:
                ret['frame'] = l * ret['frame'] + (1. - l) * mix['frame']
            else:
                ret['mix_frame'] = mix['frame']
                ret['mix_weight'] = l

            ret['contains_event'] = max(
                ret['contains_event'], mix['contains_event'])
            ret['label'] = label_dist

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            overlap_len=0,
            crop_dim=None,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            flip=False,
            glip_dir=None,              # Path to Glip feature
            multi_crop=False,
            skip_partial_end=True
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride

        crop_transform, img_transform = _get_img_transforms(
            is_eval=True, crop_dim=crop_dim, modality=modality, same_transform=True, multi_crop=multi_crop)

        # No need to enforce same_transform since the transforms are
        # deterministic
        self._frame_reader = FrameReader(
            frame_dir, modality, crop_transform, img_transform, False)

        self._flip = flip
        self._multi_crop = multi_crop
        self._glip_dir = glip_dir

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)
                        * int(skip_partial_end)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                self._clips.append((l['video'], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        video_name, start = self._clips[idx]
        frames, frame_num_list = self._frame_reader.load_frames(
            video_name, start, start + self._clip_len * self._stride, pad=True,
            stride=self._stride)

        if self._flip:
            frames = torch.stack((frames, frames.flip(-1)), dim=0)

        glip_feat = None
        glip_mask = None
        if (self._glip_dir is not None):
            glip_feat, glip_mask = load_glip(
                self._glip_dir, video_name, frame_num_list)

        if (self._glip_dir is not None):
            return {
                'video': video_name, 
                'start': start // self._stride,
                'frame': frames,
                'glip_feature': glip_feat,  # frame x obj x feat
                'glip_mask': glip_mask,
            }
        else:
            return {
                'video': video_name, 
                'start': start // self._stride,
                'frame': frames
            }
    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta['num_frames']
        num_labels = num_frames // self._stride
        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, np.int64)
        for event in meta['events']:
            frame = event['frame']
            if frame < num_frames:
                labels[frame // self._stride] = self._class_dict[event['label']]
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))
        return labels

    @property
    def augment(self):
        return self._flip or self._multi_crop

    @property
    def videos(self):
        return sorted([
            (v['video'], v['num_frames'] // self._stride,
             v['fps'] / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy['fps'] /= self._stride
                x_copy['num_frames'] //= self._stride
                for e in x_copy['events']:
                    e['frame'] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])
        num_events = sum([len(x['events']) for x in self._labels])
        print('{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg'.format(
            self._src_file, len(self._labels), num_frames, self._stride,
            num_events / num_frames * 100))
