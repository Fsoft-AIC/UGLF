#!/usr/bin/env python3
""" Inference for E2E-Spot """

import os
import argparse
import re
import torch

from dataset.frame import ActionSpotVideoDataset
from util.io import load_json
from util.dataset import load_classes
from train_e2e import E2EModel, evaluate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to the model dir')
    parser.add_argument('frame_dir', help='Path to the frame dir')
    parser.add_argument('-s', '--split',
                        choices=['train', 'val', 'test', 'challenge'],
                        required=True)
    parser.add_argument('--no_overlap', action='store_true')
    parser.add_argument('--recall_thresh', type=float)
    parser.add_argument('--criterion_key', choices=['train', 'val', 'val_mAP'], default='val_mAP')
    parser.add_argument('--glip_dir', type=str, default=None, help="Path to extracted GLIP features", required=False)

    save = parser.add_mutually_exclusive_group()
    save.add_argument('--save', action='store_true',
                      help='Save predictions with default names')
    save.add_argument('--save_as', help='Save predictions with a custom prefix')

    parser.add_argument('-d', '--dataset',
                        help='Dataset name if not inferrable from the config')
    return parser.parse_args()


def get_best_epoch(model_dir, key='val_mAP'):
    data = load_json(os.path.join(model_dir, 'loss.json'))
    if key == 'val_mAP':
        best = max(data, key=lambda x: x[key])
    else: 
        best = min(data, key=lambda x: x[key])
    return best['epoch']


def get_last_epoch(model_dir):
    regex = re.compile(r'checkpoint_(\d+)\.pt')

    last_epoch = -1
    for file_name in os.listdir(model_dir):
        m = regex.match(file_name)
        if m:
            epoch = int(m.group(1))
            last_epoch = max(last_epoch, epoch)
    assert last_epoch >= 0
    return last_epoch


def main(model_dir, frame_dir, split, no_overlap, save, save_as, glip_dir, dataset, criterion_key, recall_thresh):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path) as fp:
        print(fp.read())
    config = load_json(config_path)
    
    if (recall_thresh is not None):
        high_recall_score_threshold = recall_thresh
    elif (config['label_type']=='one_hot'):
        high_recall_score_threshold = 0.5
    else:
        high_recall_score_threshold = 0.01
    print("Recall threshold:", high_recall_score_threshold)

    if os.path.isfile(os.path.join(model_dir, 'loss.json')):
        best_epoch = get_best_epoch(model_dir, criterion_key)
        print('Best epoch:', best_epoch)
    else:
        best_epoch = get_last_epoch(model_dir)

    if dataset is None:
        dataset = config['dataset']
    else:
        if dataset != config['dataset']:
            print('Dataset mismatch: {} != {}'.format(
                dataset, config['dataset']))

    classes = load_classes(os.path.join('data', dataset, 'class.txt'))

    model = E2EModel(
        len(classes) + 1, config['feature_arch'], config['temporal_arch'],
        clip_len=config['clip_len'], 
        glip_feature = config['glip_feature'],
        modality=config['modality'],
        label_type=config['label_type'],
        multi_gpu=config['gpu_parallel'])
    model.load(torch.load(os.path.join(
        model_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

    split_path = os.path.join('data', dataset, '{}.json'.format(split))
    split_data = ActionSpotVideoDataset(
        classes, split_path, frame_dir, config['modality'], config['clip_len'],
        overlap_len=0 if no_overlap else config['clip_len'] // 2,
        glip_dir=glip_dir,
        crop_dim=config['crop_dim'])

    pred_file = None
    if save_as is not None:
        pred_file = save_as
    elif save is not None:
        pred_file = os.path.join(
            model_dir, 'pred-{}.{}'.format(split, best_epoch))
        assert not os.path.exists(pred_file), pred_file

    if pred_file is not None:
        print('Saving predictions:', pred_file)

    evaluate(model, split_data, split.upper(), classes, pred_file,
             high_recall_score_threshold = high_recall_score_threshold,
             calc_stats=True)


if __name__ == '__main__':
    main(**vars(get_args()))