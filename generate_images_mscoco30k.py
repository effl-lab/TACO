import os, sys
import torch, torchvision
import torch.nn.functional as F


from transformers import CLIPTextModel, AutoTokenizer
import lpips
from PIL import Image

from models import TACO
from config.config import model_config 
from utils.utils import *

from pytorch_msssim import ms_ssim as ms_ssim_func

import json
import shutil
import math
import pandas as pd

from tqdm import tqdm

# ============
device = 'cuda'

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_alex = loss_fn_alex.to(device)
loss_fn_alex.requires_grad_(False)

# ============

def parse_args_for_inference(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    
    parser.add_argument(
        "--image_folder_root", type=str, default='/data/MSCOCO/val2014', help="image folder path"
    )

    parser.add_argument(
        "--checkpoint", type=str, default='./checkpoint/0.0004.pth.tar', help="path of the pretrained checkpoint"
    )

    args = parser.parse_args(argv)
    return args

def main(argv):

    args = parse_args_for_inference(argv)

    clip_model_name = "openai/clip-vit-base-patch32"

    CLIP_text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    CLIP_text_model.requires_grad_(False)
    CLIP_tokenizer = AutoTokenizer.from_pretrained(clip_model_name)


    with open('./materials/mscoco_30k_list.json', 'r') as f:
        image_list = json.load(f)
    image_list.sort()

    with open('./materials/mscoco_val41k_img_cap_pair.json', 'r') as f:
        image_cap_dict = json.load(f)

    params_name = f'{args.checkpoint}'

    taco_config = model_config()
    net = TACO(taco_config, text_embedding_dim = CLIP_text_model.config.hidden_size)
    net = net.to(device)

    print(f"checkpoint: {params_name}")
    
    params_path = f'{params_name}' 

    state_dict = torch.load(params_path, map_location = device)['state_dict']
    
    try:
        try:
            net.load_state_dict(state_dict)
        except:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            net.load_state_dict(new_state_dict)
    except:
        try:
            net.module.load_state_dict(state_dict)
        except:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            net.module.load_state_dict(new_state_dict)

    del state_dict

    net.requires_grad_(False)
    net.update()
    
    stat_csv = {
        'image_name': [],
        'bpp': [],
        'psnr': [],
        'ms_ssim': [],
        'lpips': []
        }

    save_folder = f'./compression_mscoco_val30k'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    try:
        os.mkdir(f"{save_folder}")
        os.mkdir(f"{save_folder}/figures")
        os.mkdir(f"{save_folder}/temp")
    except:
        os.makedirs(f"{save_folder}")
        os.makedirs(f"{save_folder}/figures")
        os.makedirs(f"{save_folder}/temp")

    mean_csv = {
        'bpp':0.0,
        'psnr': 0.0,
        'ms_ssim': 0.0,
        'lpips':0.0
    }

    for img_name in tqdm(image_list, desc=f"compress:") :
    
        img_path = f'{args.image_folder_root}/{img_name}'

        img = torchvision.transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
        x = img.unsqueeze(0)

        _, _, H, W = x.shape
        pad_h = 0
        pad_w = 0
        if H % 64 != 0:
            pad_h = 64 * (H // 64 + 1) - H
        if W % 64 != 0:
            pad_w = 64 * (W // 64 + 1) - W

        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
        pred_image_list = []
        pred_bpp_list = []
        
        for caption in image_cap_dict[img_name] :    
            clip_token = CLIP_tokenizer([caption], padding="max_length", max_length=38, truncation=True, return_tensors="pt").to(device)
            text_embeddings = CLIP_text_model(**clip_token).last_hidden_state
            
            out_enc = net.compress(x_padded, text_embeddings)
            shape = out_enc["shape"]
            
            output = os.path.join(f'{save_folder}/temp', f'{img_name}')
            with Path(output).open("wb") as f:
                write_uints(f, (H, W))
                write_body(f, shape, out_enc["strings"])
            
            size = filesize(output)
            bpp = float(size) * 8 / (H * W)

            with Path(output).open("rb") as f:
                original_size = read_uints(f, 2)
                strings, shape = read_body(f)

            out = net.decompress(strings, shape, text_embeddings)
            x_hat = out["x_hat"]
            x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
            
            pred_image_list.append(x_hat.detach().clone())
            pred_bpp_list.append(bpp)
        
        best_image = {
            'BPP':0.0,
            'PSNR': 0.0,
            'MS-SSIM': 0.0,
            'LPIPS':0.0,
            'caption': "",
            'image': None
        }    
            
        for i, pred_image in enumerate(pred_image_list):
            bpp = pred_bpp_list[i]
            
            psnr = compute_psnr(x, pred_image)
            try:
                ms_ssim = ms_ssim_func(x, pred_image, data_range=1.).item()
            except:
                ms_ssim = ms_ssim_func(torchvision.transforms.Resize(256)(x), torchvision.transforms.Resize(256)(pred_image), data_range=1.).item()
                
            lpips_score = loss_fn_alex(x, pred_image).item()
            
            # we choose the lowest lpips image as best 
            if best_image['LPIPS'] > lpips_score :
                best_image['BPP']=bpp
                best_image['PSNR']=psnr
                best_image['MS-SSIM']=ms_ssim
                best_image['LPIPS']=lpips_score
                best_image['caption'] = image_cap_dict[img_name][i]
                best_image['image'] = pred_image.detach().clone()   
            
        final_caption = best_image['caption']
        print(f'Checkpoint: {params_name}, img_name: {img_name}, Caption: {final_caption}')
        
        torchvision.utils.save_image(best_image['image'], f'{save_folder}/figures/{img_name}', nrow=1)    
        mean_csv['bpp'] += bpp
        mean_csv['psnr'] += psnr
        mean_csv['ms_ssim'] += ms_ssim
        mean_csv['lpips'] += lpips_score

        stat_csv['image_name'].append(img_name)
        stat_csv['bpp'].append(bpp)
        stat_csv['psnr'].append(psnr)
        stat_csv['ms_ssim'].append(ms_ssim)
        stat_csv['lpips'].append(lpips_score)  

    shutil.rmtree(f"{save_folder}/temp")

    data_lpips_best = pd.DataFrame(stat_csv)
    data_lpips_best.to_csv(f'{save_folder}/stat_per_image.csv')
        
    mean_csv['bpp'] /= len(image_list)
    mean_csv['psnr'] /= len(image_list)
    mean_csv['ms_ssim'] /= len(image_list)
    mean_csv['lpips'] /= len(image_list)

    print(f"checkpoint: {params_name}")

    print(f"\nBPP: {mean_csv['bpp']}, PSNR: {mean_csv['psnr']}, MS-SSIM: {mean_csv['ms_ssim']}, LPIPS: {mean_csv['lpips']}\n")

    with open(f'{save_folder}/mean_stat.json', 'w') as f:
        json.dump(mean_csv, f, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])