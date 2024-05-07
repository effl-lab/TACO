# TACO: <u>T</u>ext-<u>A</u>daptive <u>CO</u>mpression

This repository has the source code of "Neural Image Compression with Text-guided Encoding for both Pixel-level and Perceptual Fidelity (ICML 2024)." <br>
[[project page](https://taco-imagecompression.github.io), [paper](https://arxiv.org/abs/2403.02944)]



If you have any questions about this, please send a mail to the one of co-first authors (hagyeonglee@postech.ac.kr,  minkyu.kim@postech.ac.kr) 


## How to train
For training TACO, you type the following command.
```
python -u train.py --dist_port (integer) --train_dataset_root_path (string) --lpips_coefficient (float) --joint_image_text_loss_coefficient (float) --epochs (int) --learning_rate (float) --aux-learning-rate (float)  --num-workers (int) --lambda (float) --batch-size (int) --patch-size (int int) --seed (int) --clip_max_norm (float) --lr_epoch (int int ...)  

# e.g. python -u train.py --dist_port 6411 --train_dataset_root_path /data/MSCOCO --lpips_coefficient 3.50 --joint_image_text_loss_coefficient 0.0025 --epochs 50 --learning_rate 1e-4 --aux-learning-rate 1e-3 --num-workers 8 --lambda 0.0004 --batch-size 8 --patch-size 256 256 --seed 100 --clip_max_norm 1.0 --lr_epoch 45 48 
```

The descriptions about each arguments are like following
(설명 쓰기)

* dist_port: port for using Distributed Data Parallel (DDP) (default: 6006)
* train_dataset_root_path: root folder of MSCOCO
* lpips_coefficient: coefficient of LPIPS loss (default: 1.0)
* joint_image_text_loss_coefficient: coefficient of joint image-text loss (default: 0.005)
* epochs: 
* learning_rate: 
* aux-learning-rate: 
* num-workers: 
* lambda: 
* batch-size: 
* patch-size: 
* seed: 
* clip_max_norm: 
* lr_epoch

## How to inference 


## Citation
```
@inproceedings{lee2024taco,
        title={Neural Image Compression with Text-guided Encoding for both Pixel-level and Perceptual Fidelity},
        author={Lee, Hagyeong and Kim, Minkyu and Kim, Jun-Hyuk and Kim, Seungeon and Oh, Dokwan and Lee, Jaeho},
        booktitle={International Conference on Machine Learning},
        year={2024}
```