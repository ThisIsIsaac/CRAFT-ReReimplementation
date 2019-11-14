import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
import os
from util.data_loader import Synth80k
from datetime import datetime
import logging
from util.mseloss import Maploss
from collections import OrderedDict
from model.craft import CRAFT
from torch.autograd import Variable
from util.kaist_data_loader import KAIST
import sys
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from util.validate import validate
import numpy as np
from util.writer import MyWriter
import math
from util import craft_utils

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    args = parser.parse_args()

    # build path to save checkpoints
    results_dir = make_results_dir_path(args.results_dir)
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    ########## config logging ##########
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    logging.basicConfig(
        level=numeric_log_level,
        filename=os.path.join(results_dir, "logs.log"),
        filemode='w',
        format='%(levelname)s - File \"%(filename)s\", Function \"%(funcName)s\", line %(lineno)d (%(asctime)s):\n'
               '  %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p')

    if (args.print_logs): # add handler to logger to print the output
        root_logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(handler)

    writer = MyWriter(results_dir)
    images_path = os.path.join(results_dir, "images")
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    ########################################



    ########## initialize network ##########
    if args.ckpt_path != None:
        ckpt = torch.load(args.ckpt_path)
        if "craft" in list(ckpt.keys()):
            craft_state_dict = ckpt["craft"]
            init_epoch = ckpt["epoch"]
            step = ckpt["step"]
            optim_state_dict = ckpt["optim"]

        else:
            craft_state_dict = ckpt
            init_epoch = 0
            step = 0
            optim_state_dict = None

    net = CRAFT(use_vgg16_pretrained=args.use_vgg16_pretrained, freeze=args.freeze_vgg16)
    net = torch.nn.DataParallel(net, device_ids=[0])

    net.load_state_dict(craft_state_dict)

    net = net.cuda()
    net.train()

    # initialize optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optim_state_dict != None:
        optimizer.load_state_dict(optim_state_dict)

    # since input images are resized to a fixed size during training, all input sizes are identical
    cudnn.benchmark = True
    ########################################


    ########## load training datawset ##########
    # check command-line inputs
    if args.kaist == args.synthtext:
        raise ValueError("One of --kaist or --synthtext must be given to specify the dataset used to train.")

    if args.kaist:
        dataset = KAIST(args.dataset_path, target_size=768)

    if args.synthtext:
        dataset = Synth80k(args.dataset_path, target_size = 768)

    # split dataset into two subsets: train and valid
    # source: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    indices = list(range(len(dataset)))
    val_split = 0.1
    val_dataset_size = int(np.floor(val_split * len(dataset)))
    val_indices, train_indices = indices[:val_dataset_size], indices[val_dataset_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, # for SynthText, batch size 8 takes about 29GB at most
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        sampler=val_sampler)
    ########################################


    # set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    criterion = Maploss()

    for epoch in tqdm(range(init_epoch, args.epoch)):

        # restore this after debugging validate
        #if epoch % args.val_epoch_interval == 0 and epoch != 0:
        if epoch % args.val_epoch_interval == 0:
            validate(args, net, val_loader, writer, step, images_path)

        for images, gh_label, gah_label, mask, _, unnormalized_images, img_paths in train_loader:
            step += 1

            # source: https://github.com/clovaai/CRAFT-pytorch/issues/18#issuecomment-513258344
            # initial lr is 1e-4 multiplied by 0.8 for every 10k iterations
            if step % 20000 == 0 and step != 0:
                adjust_learning_rate(optimizer, args.gamma, step)

            images = images.cuda()
            gh_label = gh_label.cuda()
            gah_label = gah_label.cuda()
            mask = mask.cuda()

            out, _ = net(images)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()

            loss = criterion(gh_label, gah_label, out1, out2, mask)

            loss.backward()
            optimizer.step()

            if loss > 1e8 or math.isnan(loss):
                imgs_paths_str = ""
                for img_path in img_paths:
                    imgs_paths_str += img_path + "\n"

                # create path and directories to save imags
                error_images_path = os.path.join(images_path, "train_error")
                if not os.path.exists(error_images_path):
                    os.mkdir(error_images_path)

                output_images_path = os.path.join(error_images_path, "network_" + str(step))
                if not os.path.exists(output_images_path):
                    os.mkdir(output_images_path)

                ref_images_path = os.path.join(error_images_path, "ref_" + str(step))
                if not os.path.exists(ref_images_path):
                    os.mkdir(ref_images_path)

                output_images = craft_utils.save_outputs_from_tensors(unnormalized_images, out1, out2,
                                                                      args.text_threshold, args.link_threshold, args.low_text,
                                                                      output_images_path, img_paths)
                ref_images = craft_utils.save_outputs_from_tensors(unnormalized_images, gh_label, gah_label,
                                                                   args.text_threshold, args.link_threshold, args.low_text,
                                                                   ref_images_path, img_paths)

                writer.log_output_images(output_images, ref_images, step)
                writer.log_training(loss, step)

                logging.error("loss %.01f at training step %d!" % (loss, step))
                logging.error("above error occured while processing images:\n" + imgs_paths_str)
                raise Exception("Loss exploded")

            if step % args.log_interval == 0 and step != 0:
                writer.log_training(loss, step)

            if step % args.save_interval == 0 and step != 0:
                ckpt_path = os.path.join(results_dir, str(step) + '.pth')
                logging.info('Saving ' + ckpt_path + ', step:' + str(step))
                torch.save({
                    'craft': net.state_dict(),
                    'optim': optimizer.state_dict(),
                    'step': step,
                    'epoch':epoch
                    } , ckpt_path)