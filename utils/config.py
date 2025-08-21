import os
import random
import torch
from PIL import Image
import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='black-forest-labs/FLUX.1-Depth-dev')
    parser.add_argument('--redux_model', type=str, default='black-forest-labs/FLUX.1-Redux-dev')
    parser.add_argument('--lora_model', type=str, default="./ckpts/lora/lora-final.safetensors")
    parser.add_argument('--outpainter_model', type=str, default='./ckpts/outpainter/texgen_v1.ckpt')
    parser.add_argument('--weighter_model', type=str, default='./ckpts/weighternet/model.safetensors')
    parser.add_argument('--blank_path', type=str, default='./blank')
    parser.add_argument('--result_path', type=str, default='./test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_steps', type=int, default=30)
    parser.add_argument('--mixing_step', type=int, default=10)
    parser.add_argument('--cfg_scale', type=float, default=6.0)
    parser.add_argument('--true_cfg', type=float, default=2.0)
    parser.add_argument('--render_azim', type=float, default=-1)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--texture_size', type=int, default=1024)
    parser.add_argument('--outpainter_sample_steps', type=int, default=30)
    parser.add_argument('--outpainter_cfg_scale', type=float, default=3.5)
    parser.add_argument('--frame_num', type=int, default=90)
    parser.add_argument('--render_ele', type=float, default=15.0)

    parser.add_argument('--mesh_path', type=str, default='./demo/jeep/5598a12768c74031b4d1d426b3105a61.obj')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--image_prompt', type=str, default=None)
    parser.add_argument('--stylize', action='store_true', default=False)
    parser.add_argument('--image_strength', type=float, default=0.3)

    args = parser.parse_args()

    fix_prompt = 'a grid of 2x2 multi-view image. white background.'
    if args.prompt is not None:
        args.outpainter_prompt = args.prompt
        args.prompt = fix_prompt + ' ' + args.prompt
    else:
        args.outpainter_prompt = None
        args.prompt = fix_prompt

    if args.image_prompt is not None:
        args.image_prompt = Image.open(args.image_prompt).convert('RGB')
    
    if args.prompt is None and args.image_prompt is None:
        raise ValueError('Please provide either a prompt or an image prompt.')
    
    if args.render_azim < 0 or args.render_azim >= 360:
        args.render_azim = random.uniform(0, 360)

    if args.dtype == 'bfloat16':
        args.dtype = torch.bfloat16
    elif args.dtype == 'float32':
        args.dtype = torch.float32

    args.result_path = os.path.join(args.result_path, args.mesh_path.split('/')[-2])
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    return args
