# train example
# python -u train.py --train_dataset_root_path /data/MSCOCO --lambda 0.0004 --epochs 50 --lr_epoch 45 48 --batch-size 8 --seed 100 --multimodal_loss_coefficient 0.0025 --lpips_coefficient 3.50 --dist_port 6411

import os
import random
import sys
import pandas as pd
import socket

import torch
import torch.optim as optim
import torch.distributed as dist


import lpips

import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets_MSCOCO import MSCOCO_train_dataset
from datasets_image_cap import Image_Cap_pair_dataset

from pytorch_msssim import ms_ssim as ms_ssim_func

from transformers import CLIPTextModel
from transformers import logging as tr_logging
tr_logging.set_verbosity_error()

from config.config import model_config 
from models import TACO
from utils.utils import *
from utils.optimizers import *

def train_one_epoch(
    model, CLIP_text_model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger, node_rank = 0
):
    model.train()
    device = next(model.parameters()).device

    for i, (d, tokens, attention_mask) in enumerate(train_dataloader):
        d = d.to(device)
        tokens = tokens.to(device)
        attention_mask = attention_mask.to(device)
    
        text_embeddings = CLIP_text_model(input_ids = tokens, attention_mask = attention_mask).last_hidden_state

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d, text_embeddings)
        out_criterion = criterion(epoch, out_net, d, tokens, attention_mask)

        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        try:
            aux_loss = model.aux_loss()
        except:
            aux_loss = model.module.aux_loss()

        aux_loss.backward()
        aux_optimizer.step()

        if i % 1000 == 0 and node_rank == 0:
            logger.info(
                f"Train epoch {epoch + 1}: ["
                f"{i*len(d)}/{len(train_dataloader)*d.size(0)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tLPIPS loss: {out_criterion["perceptual_loss"].item():.3f} |'
                f'\tMultimodal Semantic Consistent loss: {out_criterion["joint_image_text_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
        
        dist.barrier()

def test_epoch(epoch, test_dataset, loss_fn_alex, model, CLIP_text_model, tokenizer, save_path, logger):

    avg_bpp = 0.0
    mean_PSNR = 0
    mean_MS_SSIM_prob = 0
    mean_LPIPS = 0
    
    device = next(model.parameters()).device
    loss_fn_alex = loss_fn_alex.to(device)

    for img_name, image, caption in test_dataset :

        img = image.to(device)
        x = img.unsqueeze(0).to(device)
        
        _, _, H, W = x.shape
        pad_h = 0
        pad_w = 0
        if H % 64 != 0:
            pad_h = 64 * (H // 64 + 1) - H
        if W % 64 != 0:
            pad_w = 64 * (W // 64 + 1) - W

        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        token = tokenizer([caption], padding='max_length', max_length=38, truncation=True, return_tensors="pt").to(device)
        text_embeddings = CLIP_text_model(**token).last_hidden_state

        out_enc = model.compress(x_padded, text_embeddings)
        shape = out_enc["shape"]

        output = os.path.join(save_path, img_name)
        with Path(output).open("wb") as f:
            write_uints(f, (H, W))
            write_body(f, shape, out_enc["strings"])

        size = filesize(output)
        
        bpp = float(size) * 8 / (H * W)

        with Path(output).open("rb") as f:
            original_size = read_uints(f, 2)
            strings, shape = read_body(f)

        out = model.decompress(strings, shape, text_embeddings)

        x_hat = out["x_hat"]
        x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]

        avg_bpp += bpp
        
        ms_ssim_prob = ms_ssim_func(x, x_hat, data_range=1.).item()
        psnr = compute_psnr(x, x_hat)
        lpips_score = loss_fn_alex(x, x_hat).item()
        
        mean_PSNR += psnr
        mean_MS_SSIM_prob += ms_ssim_prob
        mean_LPIPS += lpips_score

        logger.info(f"Test epoch {epoch + 1}, File name: {img_name}, PSNR: {psnr}, MS-SSIM: {mean_MS_SSIM_prob}, LPIPS: {lpips_score}, BPP: {bpp}")
        torchvision.utils.save_image(x_hat, output, nrow=1)

    avg_bpp /= len(test_dataset)
    mean_PSNR /= len(test_dataset)
    mean_MS_SSIM_prob /= len(test_dataset)
    mean_LPIPS /= len(test_dataset)

    return mean_PSNR, mean_MS_SSIM_prob, mean_LPIPS, avg_bpp

def main(opts):

    node_rank = getattr(opts, "ddp.rank", 0)
    device_id = getattr(opts, "dev.device_id", torch.device("cpu"))

    logger = logger_setup(log_file_name = 'logs', log_file_folder_name = opts.save_path)

    taco_config = model_config()

    if node_rank == 0:
        logger.info("Create experiment save folder")

    csv_file = pd.DataFrame(None, index = ['Best_PSNR','Best_LPIPS','Best_MS-SSIM','Best_bpp'], columns=['bpp', 'PSNR', 'MS-SSIM', 'LPIPS', 'epoch'])

    clip_model_name = "openai/clip-vit-base-patch32"

    train_dataset = MSCOCO_train_dataset(dataset_folder=opts.train_dataset_root_path, image_size = opts.patch_size, clip_name = clip_model_name, node_rank=node_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, opts.batch_size, drop_last=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        num_workers=opts.num_workers,
        pin_memory=True,
    )
    
    CLIP_text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device_id)
    CLIP_text_model.requires_grad_(False)

    net = TACO(taco_config, text_embedding_dim = CLIP_text_model.config.hidden_size)
    net = net.to(device_id)

    net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,
        )

    optimizer, aux_optimizer = configure_optimizers(net, opts)
    
    milestones = opts.lr_epoch
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    criterion = RateDistortionLoss(lmbda=opts.lmbda, epochs=opts.epochs, clip_model_name = clip_model_name, multimodal_coefficient=opts.multimodal_loss_coefficient, lpips_coefficient = opts.lpips_coefficient)

    if node_rank == 0:  

        test_dataset = Image_Cap_pair_dataset(image_dataset_folder = './kodak', path_caption_json = './materials/kodak_ofa.json')

        logger.info("Training mode : scratch!")
        logger.info(f"lambda : {opts.lmbda}")
        logger.info(f"milestones: {milestones}")

        logger.info(f"batch_size : {opts.batch_size}")

        logger.info(f"Multimodal loss coefficient: {opts.multimodal_loss_coefficient}")
        logger.info(f"LPIPS coefficient: {opts.lpips_coefficient}")

    best_psnr = 0.0
    best_ms_ssim = 0.0
    best_bpp = float("inf")
    best_lpips = float("inf")

    recent_saved_model_path = ''
    best_psnr_model_path = ''
    best_ms_ssim_model_path = ''
    best_bpp_model_path = ''
    best_lpips_model_path = ''
    
    last_epoch = 0
    checkpoint = opts.checkpoint
    if checkpoint != "None":  # load from previous checkpoint

        if node_rank == 0:  
            logger.info(f"Loading {checkpoint}")

        checkpoint = torch.load(checkpoint, map_location=device_id)
        last_epoch = checkpoint["epoch"] + 1
        
        try:
            try:
                net.load_state_dict(checkpoint["state_dict"])
            except:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    new_state_dict[k.replace("module.", "")] = v
                net.load_state_dict(new_state_dict)
        except:
            try:
                net.module.load_state_dict(checkpoint["state_dict"])
            except:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    new_state_dict[k.replace("module.", "")] = v
                net.module.load_state_dict(new_state_dict)


        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        best_psnr = checkpoint["best_psnr"]
        best_ms_ssim = checkpoint["best_ms_ssim"]
        best_bpp = checkpoint["best_bpp"]
        best_lpips = checkpoint["best_lpips"]

        recent_saved_model_path = checkpoint["recent_saved_model_path"]

        best_psnr_model_path = checkpoint["best_psnr_model_path"]
        best_ms_ssim_model_path = checkpoint["best_ms_ssim_model_path"]
        best_bpp_model_path = checkpoint["best_bpp_model_path"]
        best_lpips_model_path = checkpoint["best_lpips_model_path"]

        del checkpoint

    save_path = opts.save_path

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex.requires_grad_(False)

    for epoch in range(last_epoch, opts.epochs):

        train_one_epoch(
            net,
            CLIP_text_model,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            opts.clip_max_norm,
            logger,
            node_rank
        )
        lr_scheduler.step()

        torch.cuda.empty_cache()

        dist.barrier()

        if (node_rank == 0): 
            with torch.no_grad() :
                if os.path.exists(f"{save_path}/figures/{epoch + 1}"):
                    shutil.rmtree(f"{save_path}/figures/{epoch + 1}")
                
                try:
                    os.mkdir(f"{save_path}/figures/{epoch + 1}")
                except:
                    os.makedirs(f"{save_path}/figures/{epoch + 1}")
                    
                net_eval = TACO(taco_config, text_embedding_dim = CLIP_text_model.config.hidden_size)
                    
                try:
                    net_eval.load_state_dict(net.module.state_dict())
                except:
                    net_eval.load_state_dict(net.state_dict())
                
                net_eval = net_eval.eval().to(device_id)
                net_eval.requires_grad_(False)
                net_eval.update()

                mean_PSNR, mean_MS_SSIM_prob, mean_LPIPS, Bit_rate = test_epoch(epoch, test_dataset, loss_fn_alex, net_eval, CLIP_text_model, train_dataset.tokenizer, f"{save_path}/figures/{epoch + 1}", logger)
                
                logger.info(f'average_PSNR: {mean_PSNR:.4f}dB, average_MS-SSIM: {mean_MS_SSIM_prob:.4f}, average_LPIPS: {mean_LPIPS:.4f}, average_Bit-rate: {Bit_rate:.4f} bpp')
            
            torch.cuda.empty_cache()

            # save recent_checkpoint
            try :
                os.remove(recent_saved_model_path) 
            except :
                logger.info("can not find recent_saved_model!")

            # save the model   
            try:
                state_dict = net.state_dict()
            except:
                state_dict = net.module.state_dict()
            
            
            if mean_PSNR > best_psnr :
                best_psnr = mean_PSNR

                try :
                    os.remove(best_psnr_model_path)
                except :
                    logger.info("can not find prev_bpp_best_model!")

                csv_file.loc['Best_PSNR',:] = [round(Bit_rate, 8), round(mean_PSNR, 8), round(mean_MS_SSIM_prob, 8), round(mean_LPIPS, 8), (epoch + 1)]

                best_psnr_model_path = save_path + '/' + f'best_psnr_model_PSNR_{round(mean_PSNR, 5)}_MS_SSIM_{round(mean_MS_SSIM_prob, 5)}_BPP_{round(Bit_rate, 5)}_LPIPS_{round(mean_LPIPS, 5)}_epoch_{epoch + 1}.pth.tar'
                save_checkpoint(epoch, state_dict, optimizer, aux_optimizer,
                                    lr_scheduler, best_psnr, best_ms_ssim, best_bpp, best_lpips,
                                    recent_saved_model_path, best_psnr_model_path, best_ms_ssim_model_path, best_bpp_model_path, best_lpips_model_path, save_type = 'psnr')

            if mean_MS_SSIM_prob > best_ms_ssim :
                best_ms_ssim = mean_MS_SSIM_prob

                try :
                    os.remove(best_ms_ssim_model_path)
                except :
                    logger.info("can not find prev_ms_ssim_best_model!")

                csv_file.loc['Best_MS-SSIM',:] = [round(Bit_rate, 8), round(mean_PSNR, 8), round(mean_MS_SSIM_prob, 8), round(mean_LPIPS, 8), (epoch + 1)]

                best_ms_ssim_model_path = save_path + '/' + f'best_ms_ssim_model_PSNR_{round(mean_PSNR, 5)}_MS_SSIM_{round(mean_MS_SSIM_prob, 5)}_BPP_{round(Bit_rate, 5)}_LPIPS_{round(mean_LPIPS, 5)}_epoch_{epoch + 1}.pth.tar'
                save_checkpoint(epoch, state_dict, optimizer, aux_optimizer,
                    lr_scheduler, best_psnr, best_ms_ssim, best_bpp, best_lpips,
                    recent_saved_model_path, best_psnr_model_path, best_ms_ssim_model_path, best_bpp_model_path, best_lpips_model_path, save_type = 'ms_ssim')

            if Bit_rate < best_bpp : 
                best_bpp = Bit_rate

                try :
                    os.remove(best_bpp_model_path)
                except :
                    logger.info("can not find prev_bpp_best_model!")

                csv_file.loc['Best_bpp',:] = [round(Bit_rate, 8), round(mean_PSNR, 8), round(mean_MS_SSIM_prob, 8), round(mean_LPIPS, 8), (epoch + 1)]

                best_bpp_model_path = save_path + '/' + f'best_bpp_model_PSNR_{round(mean_PSNR, 5)}_MS_SSIM_{round(mean_MS_SSIM_prob, 5)}_BPP_{round(Bit_rate, 5)}_epoch_LPIPS_{round(mean_LPIPS, 5)}_{epoch + 1}.pth.tar'
                save_checkpoint(epoch, state_dict, optimizer, aux_optimizer,
                    lr_scheduler, best_psnr, best_ms_ssim, best_bpp, best_lpips,
                    recent_saved_model_path, best_psnr_model_path, best_ms_ssim_model_path, best_bpp_model_path, best_lpips_model_path, save_type = 'bpp')
            
            if mean_LPIPS < best_lpips : 
                best_lpips = mean_LPIPS

                try :
                    os.remove(best_lpips_model_path)
                except :
                    logger.info("can not find prev_lpips_best_model!")

                csv_file.loc['Best_LPIPS',:] = [round(Bit_rate, 8), round(mean_PSNR, 8), round(mean_MS_SSIM_prob, 8), round(mean_LPIPS, 8), (epoch + 1)]

                best_lpips_model_path = save_path + '/' + f'best_lpips_model_PSNR_{round(mean_PSNR, 5)}_MS_SSIM_{round(mean_MS_SSIM_prob, 5)}_BPP_{round(Bit_rate, 5)}_epoch_LPIPS_{round(mean_LPIPS, 5)}_{epoch + 1}.pth.tar'
                save_checkpoint(epoch, state_dict, optimizer, aux_optimizer,
                    lr_scheduler, best_psnr, best_ms_ssim, best_bpp, best_lpips,
                    recent_saved_model_path, best_psnr_model_path, best_ms_ssim_model_path, best_bpp_model_path, best_lpips_model_path, save_type = 'lpips')

            recent_saved_model_path = save_path + '/' + f'recent_model_PSNR_{round(mean_PSNR, 5)}_MS_SSIM_{round(mean_MS_SSIM_prob, 5)}_BPP_{round(Bit_rate, 5)}_LPIPS_{round(mean_LPIPS, 5)}_epoch_{epoch + 1}.pth.tar'
            save_checkpoint(epoch, state_dict, optimizer, aux_optimizer,
                lr_scheduler, best_psnr, best_ms_ssim, best_bpp, best_lpips,
                recent_saved_model_path, best_psnr_model_path, best_ms_ssim_model_path, best_bpp_model_path, best_lpips_model_path, save_type = 'recent')

            csv_file.to_csv(save_path + '/' + 'best_res.csv')

            del net_eval

            torch.cuda.empty_cache()

        dist.barrier()

def distributed_init(opts) -> int:
    ddp_url = getattr(opts, "ddp.dist_url", None)

    node_rank = getattr(opts, "ddp.rank", 0)
    is_master_node = (node_rank == 0)

    if ddp_url is None:
        # 따로 지정한게 없어서 무조건 이쪽으로 들어옴
        ddp_port = opts.dist_port
        hostname = socket.gethostname()
        ddp_url = "tcp://{}:{}".format(hostname, ddp_port)
        setattr(opts, "ddp.dist_url", ddp_url)

    node_rank = getattr(opts, "ddp.rank", 0)
    world_size = getattr(opts, "ddp.world_size", 0)

    if torch.distributed.is_initialized():
        print("DDP is already initialized and cannot be initialize twice!")
    else:
        print("distributed init (rank {}): {}".format(node_rank, ddp_url))

        dist_backend = getattr(opts, "ddp.backend", "nccl")  # "gloo"

        if dist_backend is None and dist.is_nccl_available():
            dist_backend = "nccl"
            if is_master_node:
                print(
                    "Using NCCL as distributed backend with version={}".format(
                        torch.cuda.nccl.version()
                    )
                )
        elif dist_backend is None:
            dist_backend = "gloo"

        dist.init_process_group(
            backend=dist_backend,
            init_method=ddp_url,
            world_size=world_size,
            rank=node_rank,
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    node_rank = torch.distributed.get_rank()
    setattr(opts, "ddp.rank", node_rank)
    return node_rank

def distributed_worker(i, main, opts):
    setattr(opts, "dev.device_id", i)
    torch.cuda.set_device(i)
    setattr(opts, "dev.device", torch.device(f"cuda:{i}"))

    ddp_rank =  i
    setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts)


def ddp_or_single_process(argvs):

    opts = parse_args(argvs)

    checkpoint = "None"
    
    save_path = f'./checkpoint/exp_lambda_{opts.lmbda}_seed_{opts.seed}_batch_size_{opts.batch_size}_multimodal_loss_coefficient_{opts.multimodal_loss_coefficient}_lpips_coefficient_{opts.lpips_coefficient}'
    
    if os.path.exists(save_path):
        logger = logger_setup(log_file_name = 'logs', log_file_folder_name = save_path)
        logger.info("find checkpoint...")
        
        file_list = os.listdir(save_path)

        for file_name in file_list:
            if ('recent_model' in file_name) and ('.pth.tar' in file_name):
                logger.info(f"checkpoint exist, name: {file_name}")
                checkpoint = f'{save_path}/{file_name}'
        
        if checkpoint == 'None':
            logger.info("no checkpoint is here")
        

    else:
        create_exp_folder(save_path)
        logger = logger_setup(log_file_name = 'logs', log_file_folder_name = save_path)
        logger.info("Create new exp folder!")


    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"seed : {opts.seed}")
    logger.info(f"exp name : exp_lambda_{opts.lmbda}_seed_{opts.seed}_batch_size_{opts.batch_size}_multimodal_loss_coefficient_{opts.multimodal_loss_coefficient}_lpips_coefficient_{opts.lpips_coefficient}")

    setattr(opts, "checkpoint", checkpoint)
    setattr(opts, "save_path", save_path)

    setattr(opts, "dev.num_gpus", torch.cuda.device_count())
    setattr(opts, "ddp.world_size", torch.cuda.device_count())

    logger.info(f"opts: {opts}")

    torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts),
            nprocs=getattr(opts, "dev.num_gpus"),
        )


if __name__ == "__main__":

    ddp_or_single_process(sys.argv[1:])
