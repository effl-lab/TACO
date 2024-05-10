import os
import argparse
import math
import shutil
import logging
import json
import struct
from pathlib import Path

import PIL.Image as Image

import torch
from torchvision.transforms import ToPILImage
from torchvision import transforms
import torch.nn as nn

import lpips, clip

from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection 

def logger_setup(log_file_name=None, log_file_folder_name = None, filepath=os.path.abspath(__file__), package_files=[]):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', 
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO'.upper())

    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    info_file_handler = logging.FileHandler(log_file_folder_name + '/' + log_file_name , mode="a")
    info_file_handler.setLevel('INFO'.upper())
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger

def save_checkpoint(epoch, state_dict, optimizer, aux_optimizer, 
                    lr_scheduler, best_psnr, best_ms_ssim, best_bpp, best_lpips, 
                    recent_saved_model_path, best_psnr_model_path, best_ms_ssim_model_path, best_bpp_model_path, best_lpips_model_path, save_type = 'recent'):
    
    save_path = ''
    if save_type == 'recent':
        save_path = recent_saved_model_path
    elif save_type == 'psnr':
        save_path = best_psnr_model_path
    elif save_type == 'ms_ssim':
        save_path = best_ms_ssim_model_path
    elif save_type == 'lpips':
        save_path = best_lpips_model_path
    elif save_type == 'bpp':
        save_path = best_bpp_model_path

    torch.save({
            "epoch": epoch,
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "best_psnr":best_psnr,
        "best_ms_ssim":best_ms_ssim,
        "best_bpp":best_bpp,
        "best_lpips":best_lpips,
        "recent_saved_model_path":recent_saved_model_path,
        "best_psnr_model_path":best_psnr_model_path,
        "best_ms_ssim_model_path":best_ms_ssim_model_path,
        "best_bpp_model_path":best_bpp_model_path,
        "best_lpips_model_path":best_lpips_model_path,
        }, save_path)

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)
        
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, epochs=50, clip_model_name = "openai/clip-vit-base-patch32", jit_coefficient=0.005, lpips_coefficient = 1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.epochs = epochs # 학습 시 사용하는 epochs

        self.k_P = lpips_coefficient
        self.loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
        self.loss_fn_alex.net.requires_grad_(False)

        self.k_J = jit_coefficient # Multimodal 어쩌구 loss
        self.beta = 40.0 # Multimodal 어쩌구 loss에 들어감

        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_name).eval()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).eval()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # mimic the image transform function of CLIP
        self.transform_for_clip = transforms.Compose([
            lambda x: x*255.0, 
            transforms.Resize(224), # do_resize
            transforms.CenterCrop(224), # do_center_crop
            lambda x: x*0.00392156862745098, # do_rescale
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # do_normalize
            ])

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # "openai/clip-vit-base-patch32" 이 가진 값을 그대로 사용
    
    ################################################
    # code from modeling_clip.py in huggingface
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    
    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
    ################################################

    def get_joint_image_text_loss(self, output, target, text_tokens, attention_mask) :
        
        self.image_encoder = self.image_encoder.to(output["x_hat"].device)
        self.text_encoder = self.text_encoder.to(output["x_hat"].device)

        compressed_image = self.transform_for_clip(output["x_hat"])
        target_image = self.transform_for_clip(target)

        compressed_image_embeddings = self.image_encoder(compressed_image).image_embeds
        target_image_embeddings = self.image_encoder(target_image).image_embeds

        # loss 계산을 위해 정규화 시도
        compressed_image_embeddings = compressed_image_embeddings / compressed_image_embeddings.norm(p=2, dim=-1, keepdim=True)
        target_image_embeddings = target_image_embeddings / target_image_embeddings.norm(p=2, dim=-1, keepdim=True)

        text_embeddings = self.text_encoder(input_ids = text_tokens, attention_mask = attention_mask).text_embeds
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeddings, compressed_image_embeddings.t()) * logit_scale
        
        joint_image_text_loss = self.clip_loss(logits_per_text) + self.beta * torch.norm(compressed_image_embeddings - target_image_embeddings, p=2)

        return joint_image_text_loss

    def forward(self, output, target, text_tokens, attention_mask):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["ms_ssim_loss"] = None
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"] 

        out["joint_image_text_loss"] = torch.zeros(1)
        out["perceptual_loss"] = torch.zeros(1)        

        if self.k_J > 0.0:
            out["joint_image_text_loss"] = self.get_joint_image_text_loss(output, target, text_tokens, attention_mask)
            out["loss"] +=  self.k_J * out["joint_image_text_loss"]
        if self.k_P > 0.0 :
            self.loss_fn_alex = self.loss_fn_alex.to(target.device)
            out["perceptual_loss"] = self.loss_fn_alex(output["x_hat"], target).mean()
            out["loss"] += self.k_P * out["perceptual_loss"]
            
        return out


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "--dist_port", type=int, default=6006, required=True, help="dist_port(default: %(default)s)"
    )
    
    parser.add_argument(
        "--train_dataset_root_path", type=str, required=True, help="MSCOCO root path (e.g. /data/MSCOCO)"
    )

    parser.add_argument(
        "--lpips_coefficient", type=float, default=1.0, required=True, help="the coefficient of lpips loss(default: %(default)s)"
    )

    parser.add_argument(
        "--joint_image_text_loss_coefficient", type=float, default=0.005, required=True, help="the coefficient of multi-modal loss(default: %(default)s)"
    )
    
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0004,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="Gradient clipping max norm (default: %(default)s",
    )

    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )

    args = parser.parse_args(argv)
    return args

def create_exp_folder(save_path) :
    try:
        os.mkdir(save_path)
        os.mkdir(f"{save_path}/figures")
    except:
        os.makedirs(save_path)
        os.makedirs(f"{save_path}/figures")
