from EndoFinder import datasets, transforms, models
import torch
import numpy as np
import time
import torch.backends.cudnn as cudnn
from engine import inference, test, evaluate
import argparse
import os

import functools
import logging
import vits.vision_transformer as vits
from vits.model import VisionTransformer

def get_args_parser():
    parser = argparse.ArgumentParser('EndoFinder-Scene eval', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='EndoFinder_S', type=str, metavar='MODEL', #vit_large_patch16, EndoFinder_S, sscd, sscd_pretrained
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mode', default="cosine", type=str,
                        help='hamming or cosine')
    parser.add_argument('--boostrap', default=False, type=str,
                        help='train use boostrap')

    # Dataset parameters
    parser.add_argument('--jsonl_path', default="jsonl/results",
                        help='path where to save, empty for no saving')
    parser.add_argument('--train_jsonl_path', default="jsonl/PolypScene-2k.jsonl", type=str,
                        help='train jsonl path')
    parser.add_argument('--val_jsonl_path', default="jsonl/PolypScene-250.jsonl",
                        help='val jsonl path')
    parser.add_argument('--total_jsonl_path', default="jsonl/PolypScene-80.jsonl", type=str,
                        help='total jsonl path')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default="EndoFinder_Scene.pth",
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser

def main(args):

    print(args)
    start_time = time.time()
    device = torch.device(args.device)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True


    # Re-ID
    val_dataset = datasets.MultiPolypJsonlDataset(mode='val', 
                                    args = args,
                                    transform=transforms.ValTransform(input_size=args.input_size))
    # val_dataset = datasets.MultiImagesPolypJsonlDataset( 
    #                                 args = args,
    #                                 transform=transforms.ValTransform(input_size=args.input_size))
    # CLS
    # val_dataset = datasets.MultiViewPolypJsonlDataset( 
    #                                 args = args,
    #                                 mode = "val",
    #                                 transform=transforms.ValTransform(input_size=args.input_size))
    # val_dataset = datasets.TestPolypJsonlDataset( 
    #                             args = args,
    #                             transform=transforms.ValTransform(input_size=args.input_size))

    # val_dataset = datasets.PolypTwinDataset( 
    #                                 transform=transforms.ValTransform(input_size=args.input_size))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=1,
        shuffle=False, pin_memory=False, persistent_workers=True)  

    model = models.LoadModelSetting.get_model(models.LoadModelSetting[args.model], args.resume)
    # model = models.LoadModelSetting.get_model(models.LoadModelSetting["vit_large_patch16"], "pretrain_pth/EndoFinder.pth")

    model.to(device)

    print(f"dataset len: {len(val_dataset)}")

    # inference(args, val_loader, model, device, mean=False, cls=False)
    # test(args, val_loader, model, device)
    evaluate(args, 10000, val_loader, model, device)

if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()

    main(args)
    


