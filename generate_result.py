import sys
import os
import numpy as np
import cv2
import torch
from model import VideoSaliencyModel
from scipy.ndimage.filters import gaussian_filter
from loss import kldiv, cc, nss
import argparse

from torch.utils.data import DataLoader
from dataloader import DHF1KDataset
from utils import *
import time
from tqdm import tqdm
from torchvision import transforms, utils
from os.path import join

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def validate(args):
    path_indata = args.path_indata
    file_weight = args.file_weight

    len_temporal = args.clip_size

    if args.img_backbone in ['squeezenet', 's3d']:
        model = VideoSaliencyModel(
            img_backbone=args.img_backbone,
            # transformer_in_channel=args.transformer_in_channel,
            # nhead=args.nhead,
            use_upsample=bool(args.decoder_upsample),
            num_hier=args.num_hier,
            num_clips=args.clip_size
        )
        model.load_state_dict(torch.load(file_weight))
    elif 'tsm' in args.img_backbone:
        from tsm.ops.models import TSN
        model = TSN(base_model=args.img_backbone.split('_')[
                    1], shift_div=args.shift_div, is_shift=args.shift, shift_place=args.shift_place)
        if 'mobilenetv2' in args.img_backbone:
            sd = torch.load(file_weight)
            model_dict = model.state_dict()
            replace_dict = []
            for k, v in sd.items():
                if k not in model_dict and k.replace('.net', '') in model_dict:
                    print('=> Load after remove .net: ', k)
                    replace_dict.append((k, k.replace('.net', '')))
            for k, v in model_dict.items():
                if k not in sd and k.replace('.net', '') in sd:
                    print('=> Load after adding .net: ', k)
                    replace_dict.append((k.replace('.net', ''), k))

            for k, k_new in replace_dict:
                sd[k_new] = sd.pop(k)
            keys1 = set(list(sd.keys()))
            keys2 = set(list(model_dict.keys()))
            set_diff = (keys1 - keys2) | (keys2 - keys1)
            print('#### Notice: keys that failed to load: {}'.format(set_diff))
            model_dict.update(sd)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(torch.load(file_weight))
    model = torch.nn.DataParallel(model)

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    list_indata = [d for d in os.listdir(
        path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()

    if args.start_idx != -1:
        _len = (1.0/float(args.num_parts))*len(list_indata)
        list_indata = list_indata[int(
            (args.start_idx-1)*_len): int(args.start_idx*_len)]

    for dname in list_indata:
        print('processing ' + dname, flush=True)
        list_frames = [f for f in os.listdir(os.path.join(path_indata, dname, 'images')) if os.path.isfile(
            os.path.join(path_indata, dname, 'images', f))]
        list_frames.sort()
        os.makedirs(join(args.save_path, dname), exist_ok=True)

        # process in a sliding window fashion
        if len(list_frames) >= 2*len_temporal-1:

            snippet = []
            for i in range(len(list_frames)):
                torch_img, img_size = torch_transform(os.path.join(
                    path_indata, dname, 'images', list_frames[i]))

                snippet.append(torch_img)

                if i >= len_temporal-1:
                    clip = torch.FloatTensor(
                        torch.stack(snippet, dim=0)).unsqueeze(0)
                    clip = clip.permute((0, 2, 1, 3, 4))

                    process(model, clip, path_indata, dname,
                            list_frames[i], args, img_size)

                    # process first (len_temporal-1) frames
                    if i < 2*len_temporal-2:
                        process(model, torch.flip(
                            clip, [2]), path_indata, dname, list_frames[i-len_temporal+1], args, img_size)

                    del snippet[0]
        else:
            print(' more frames are needed')


def torch_transform(path):
    img_transform = transforms.Compose([
        transforms.Resize((224, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(path).convert('RGB')
    sz = img.size
    img = img_transform(img)
    return img, sz


def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img, (k_size, k_size), 0)
    return torch.FloatTensor(bl)


def process(model, clip, path_inpdata, dname, frame_no, args, img_size):
    with torch.no_grad():
        smap = model(clip.to(device)).cpu().data[0]

    smap = smap.numpy()
    smap = cv2.resize(smap, (img_size[0], img_size[1]))
    smap = blur(smap)

    img_save(smap, join(args.save_path, dname, frame_no), normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_weight', default="./saved_models/ViNet_DHF1K.pt", type=str)
parser.add_argument('--img_backbone', default='s3d',
                    choices=['s3d', 'squeezenet', 'tsm_resnet50', 'tsm_mobilenetv2'], type=str)
parser.add_argument('--nhead', default=4, type=int)
parser.add_argument('--num_encoder_layers', default=3, type=int)
parser.add_argument('--transformer_in_channel', default=32, type=int)
parser.add_argument(
    '--save_path', default='/ssd_scratch/cvit/varunc/Results/theatre_hollywood', type=str)
parser.add_argument('--start_idx', default=-1, type=int)
parser.add_argument('--num_parts', default=4, type=int)
parser.add_argument(
    '--path_indata', default='/ssd_scratch/cvit/varunc/DHF1K/val', type=str)
parser.add_argument('--multi_frame', default=0, type=int)
parser.add_argument('--decoder_upsample', default=1, type=int)
parser.add_argument('--num_decoder_layers', default=-1, type=int)
parser.add_argument('--num_hier', default=3, type=int)
parser.add_argument('--clip_size', default=32, type=int)

parser.add_argument('--shift', default=False,
                    action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int,
                    help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres',
                    type=str, help='place for shift (default: stageres)')

args = parser.parse_args()
print(args)
validate(args)
