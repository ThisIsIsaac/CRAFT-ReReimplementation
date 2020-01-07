import torch
import argparse
import os
from datetime import datetime
import logging
import sys
import numpy as np
from util.writer import MyWriter
from train import train



def make_duplicate_id(path):
    if os.path.exists(path):
        path = path + "_copy"
        numeric_index = len(path)

        if os.path.exists(path) == False:
            return path

        i = 1
        while os.path.exists(path):
            path = path[:numeric_index] + str(i)
            i+=1

    return path

def make_results_dir_path(path):
    # set the directory that logs, models, and statistics will be saved
    if path is None or path == "":
        results_dir = os.path.join('results', datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    if path is not None and path != "":
        results_dir = os.path.join('results', path)
        if os.path.exists(results_dir):
            results_dir = make_duplicate_id(results_dir)
    return results_dir


if __name__ == '__main__':
    """
    Supervised learning
    """

    parser = argparse.ArgumentParser(description='CRAFT reimplementation')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size of training')
    parser.add_argument('--lr', '--learning-rate', default=3.2768e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=32, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--dataset_path', default='/data/CRAFT-pytorch/syntext/SynthText/SynthText', type=str,
                        help='Path to root directory of SynthText dataset')
    parser.add_argument('--results_dir', default=None, type=str,
                        help='Path to save checkpoints')
    parser.add_argument("--ckpt_path", default='/DATA1/isaac/CRAFT-Reimplemetation/pretrain/official_pretrained/craft_mlt_25k.pth', type=str,
                        help="path to pretrained model")
    parser.add_argument('--use_vgg16_pretrained', default=False, action='store_true',
                        help="use Pytorch's pretrained model for vgg16_bn")
    parser.add_argument("--kaist", action="store_true",
                        help="use KAIST dataset")
    parser.add_argument("--synthtext", action="store_true",
                        help="use SynthText dataset")
    parser.add_argument('--log_level', type=str, default="INFO",
                        help='Set logging level to one of: CRITICAL, ERROR, WARNING, INFO, DEBUG')
    parser.add_argument('--print_logs', action='store_true', help='Print log to stdout as well as to a file')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.45, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--freeze_vgg16', default=False, action='store_true', help='freeze weights for vgg16')
    parser.add_argument("--epoch", default=1000, type=int,
                        help="number of training epochs, starting from the epoch saved in checkpoint")
    parser.add_argument("--valid_path", type=str, help="path to validation set")
    parser.add_argument("--log_interval", type=int, default= 10, help="step interval to log loss during training")
    parser.add_argument("--val_epoch_interval", type=int, default= 1, help="epoch interval to validate")
    parser.add_argument("--save_interval", type=int, default=300, help="step interval to save during training")
    parser.add_argument("--seed", type=int, default=13, help="random seed for Numpy and Pytorch")
    parser.add_argument("--is_unsupervised", default=False, type=bool, action="store_true")
    args = parser.parse_args()

    # build path to save checkpoints
    results_dir = make_results_dir_path(args.results_dir)
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    images_path = os.path.join(results_dir, "images")
    if not os.path.exists(images_path):
        os.mkdir(images_path)

    ########## config logging ##########
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    logging.basicConfig(
        level=numeric_log_level,
        filename=os.path.join(results_dir, "logs.log"),
        filemode='w',
        format='%(levelname)s - File \"%(filename)s\", Function \"%(funcName)s\", line %(lineno)dr (%(asctime)s):\n'
               '  %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p')

    if (args.print_logs):  # add handler to logger to print the output
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(handler)

    writer = MyWriter(results_dir)

    ########################################

    # set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args, writer, results_dir, images_path)
