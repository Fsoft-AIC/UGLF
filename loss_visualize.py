import json
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Path to loss file (.json) or directory of model'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Path to save image'
    )
    args = parser.parse_args()
    return args

def main(args):
    in_name = args.input
    out_name = args.output

    if ('.json' in in_name):
        file = open(in_name)
    else:
        file = open(os.path.join(in_name,"loss.json"))
    data = json.load(file)

    train_loss = []
    val_loss = []
    epoch = []
    for i in range(len(data)):
        epoch.append(data[i]['epoch'])
        train_loss.append(data[i]['train'])
        val_loss.append(data[i]['val'])

    idx = np.argmin(val_loss)
    print(f"Best epoch: {epoch[idx]}")
    print(f"Best train: {train_loss[idx]}")
    print(f"Best val: {val_loss[idx]}")

    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Val loss')
    plt.legend()
    plt.savefig(out_name)

if (__name__=="__main__"):
    main(get_args())