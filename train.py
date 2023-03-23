import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg',
                        help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--epochs', default=32, type=int,
                        help='Epochs to run [default: 32]')
    parser.add_argument('--npoint', type=int, default=4096,
                        help='Point Number [default: 4096]')

    return parser.parse_args()


def main(args):
    # train dataset and train loader
    # _Random for now_


if __name__ == '__main__':
    args = parse_args()
    main(args)
