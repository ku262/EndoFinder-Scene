from EndoFinder import datasets, transforms, models, util
from EndoFinder.datasets import CustomBatchSampler
from engine import train_one_epoch, evaluate

import torch
import numpy as np
import json
import time
import torch.backends.cudnn as cudnn
from engine import inference
import argparse
import datetime
import os
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory



def get_args_parser():
    parser = argparse.ArgumentParser('EndoFinder training', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mode', default="cosine", type=str,
                        help='hamming or cosine')
    parser.add_argument('--boostrap', default=False, type=str,
                        help='train use boostrap')
    parser.add_argument('--frozen_IEncoder', default=True, type=int,
                        help='frozen image encoder')
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--entropy_weight', default=5, type=float,
                        help='entropy weight')
    parser.add_argument('--mse_weight', default=0.5, type=float, #2
                        help='mae weight')
    parser.add_argument('--sscd_weight', default=1, type=float,
                        help='sscd weight')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', #40
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_jsonl_path', default="jsonl/PolypScene-2k.jsonl", type=str,
                        help='train jsonl path')
    parser.add_argument('--val_jsonl_path', default="jsonl/PolypScene-250.jsonl",
                        help='val jsonl path')
    parser.add_argument('--output_dir', default="outputs",
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default="outputs",
                        help='path where to save, empty for no saving')
    parser.add_argument('--finetune', default="pretrain_pth/EndoFinder.pth",
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default="",
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser

# 估算参数所占用的显存空间
def estimate_memory_usage(model):
    total_memory = 0
    for param in model.parameters():
        total_memory += param.element_size() * param.nelement()
    return total_memory

def main(args):

    print(args)
    start_time = time.time()
    device = torch.device(args.device)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True

    train_dataset = datasets.MultiPolypJsonlDataset(mode='train', 
                                    args = args,
                                    transform=transforms.AdvancedTransform(input_size=args.input_size), 
                                    )
    val_dataset = datasets.MultiPolypJsonlDataset(mode='val', 
                                    args = args,
                                    transform=transforms.ValTransform(input_size=args.input_size))

    train_sampler = CustomBatchSampler(train_dataset, batch_size=args.batch_size, num_views_per_polyp=4, drop_last=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=1,
        pin_memory=True, 
        worker_init_fn=datasets.worker_init_fn, persistent_workers=True,
        batch_sampler=train_sampler)      
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=1,
        shuffle=False, pin_memory=True, 
        worker_init_fn=datasets.worker_init_fn, persistent_workers=True)                

    print(f"train dataset len: {len(train_dataset)}, val dataset len: {len(val_dataset)}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True) 
    
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)


    model = models.model.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    if args.resume:
        print("Load checkpoint from: %s" % args.resume)
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
        model.to(device)
        eval_metric = evaluate(args, 10000, val_loader, model, device)
        exit(0)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'decoder_embed.weight', 'decoder_embed.bias', 'mask_token', 'decoder_pos_embed', 'decoder_pred.weight', 'decoder_pred.bias']:
            # if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

        # interpolate position embedding
        util.interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        if args.frozen_IEncoder:
            for name, param in model.named_parameters():
                if name in checkpoint_model.keys():
                    # print(name)
                    param.requires_grad = False


    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Requires grad: {param.requires_grad}")



    print(model)
    model.to(device)
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    # 计算可训练参数量
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )
    
    max_uAP = 0
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):

        train_data = train_one_epoch(args, epoch, model, train_loader, device, optimizer, scheduler, log_writer)

        scheduler.step()

        eval_metric = evaluate(args, epoch, val_loader, model, device, log_writer)
        
        if args.output_dir and (epoch + 1 == args.epochs): #or (epoch > 75 and epoch % 5 == 0)):

            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint-{epoch}.pth"))

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            
            if eval_metric["uAP"]>max_uAP:
                max_uAP = eval_metric["uAP"]
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_checkpoint.pth"))
            
            eval_metric["Epoch"] = epoch
            with open(os.path.join(args.output_dir, "eval_log.txt"), mode="a", encoding="utf-8") as f:
                json.dump(eval_metric, f, indent=None)
                f.write("\n")
        
        print(f"[max eval uAP: {max_uAP}]")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()

    main(args)
    


