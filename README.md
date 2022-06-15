# 尋找花中君子 - 蘭花種類辨識及分類競賽

## Supervised Learning for Few-Shot Orchid types Classification with Prior Guided Feature



## Setup


<details>

<summary>Conda environment</summary>
  
```bash
conda create -n ViT python==3.9 -y
conda activate ViT
```

</details>




<details>

<summary>Clone Repository</summary>
  
```bash
git clone https://github.com/TW-yuhsi/ViT-Orchids-Classification
pip install -r requirements.txt
```

</details>




<details>

<summary>A PyTorch Extension (Apex)</summary>
  
```bash
git clone https://github.com/NVIDIA/apex
cd apex/
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./    # if error occur, run the following command
python setup.py install
```

</details>




<details>

<summary>Get pretrained weight</summary>
  
```bash
cd ViT-Orchids-Classification-main/
mkdir checkpoint
cd checkpoint/
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16.npz
```
  
## Usage
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16
```
### imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

### imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz
```

</details>



<details>

<summary>Useful commands</summary>
  
```bash=
unzip \*.zip    # Unzip all ZIP files
ls -l | grep "^-" | wc -l    # Check the number of files
ls -lR | grep "^-" | wc -l
for i in `seq 0 218`; do ls ${i} -lR | grop "^-" | wc -l; done
```
  
</details>



<details>

<summary>Folder Structure on PC</summary>


```
├── data/
    └── baseline data/
        └── test/
            └── 0/ 1/ 2/ ...
        └── train/
            └── 0/ 1/ 2/ ...
        └── val/
            └── 0/ 1/ 2/ ...
    └── fold1/
        └── test/
            └── 0/ 1/ 2/ ...
        └── train/
            └── 0/ 1/ 2/ ...
        └── val/
            └── 0/ 1/ 2/ ...
    └── fold2/
        └── test/
            └── 0/ 1/ 2/ ...
        └── train/
            └── 0/ 1/ 2/ ...
        └── val/
            └── 0/ 1/ 2/ ...
    └── fold3/
        └── test/
            └── 0/ 1/ 2/ ...
        └── train/
            └── 0/ 1/ 2/ ...
        └── val/
            └── 0/ 1/ 2/ ...
    └── fold4/
        └── test/
            └── 0/ 1/ 2/ ...
        └── train/
            └── 0/ 1/ 2/ ...
        └── val/
            └── 0/ 1/ 2/ ...
├── ViT-Orchids-Classification-main/
│   ├── apex/
│   ├── checkpoint/
│   │   ├── ViT-B_16.npz
│   ├── models/
│   ├── utils/
│   ├── requirements.txt
│   ├── test.py
│   ├── train.py
```
</details>



## Train
```
python train.py --name <name of this run> \
                --dataset <task> \
                --foldn <fold n> \
                --model_type <model type> \
                --pretrained_dir <pretrained> \
                --img_size <image size> \
                --train_batch_size <batch size> \
                --optim <optimizer> \
                --learning_rate <learning rate> \ 
                --weight_decay <weight decay> \
                --num_steps <num steps> \
                --use_imagenet_mean_std <mean and std in imagenet> \
                --rot_degree <rotate degree> \
                --fliplr <prob. flip> \
                --noise <prob. gaussian noise> \
                --loss_fct <loss functoin> \
                --fp16 \
                --fp16_opt_level O2
```
#### example
```
python train.py --name orchid --dataset orchid --foldn 1 --train_batch_size 4 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --img_size 480 --loss_fct CE --optim SGD --learning_rate 3e-2 --num_steps 40000 --fp16 --fp16_opt_level O2
```




## Test
```
python test.py --model_type ViT-B_16 \
               --checkpoint output/orchid_ViT-B_16_checkpoint.bin \
               --img_size 384 \
               --test_dir ../data \
               --foldn 1 \
               --dataset test \
               --use_imagenet_mean_std \
               --use_test_aug
```
#### example
```
python test.py --model_type ViT-B_16 --checkpoint output/orchid_ViT-B_16_checkpoint.bin --img_size 480 --foldn 1 --dataset test --use_imagenet_mean_std --use_test_aug
```


## Related URLs
- [Competition Link](https://tbrain.trendmicro.com.tw/Competitions/Details/20)
- [Google Drive](https://drive.google.com/drive/folders/1x_rb6bu0riJuouAtK-xjFGDkCP7ZbhbL?usp=sharing)



## GitHub Acknowledgement
### Member
- [Jia-Wei Liao](https://github.com/Jia-Wei-Liao/Orchid_Classification)

### Others
- Augmentation
  - AutoAugment: https://github.com/DeepVoltaire/AutoAugment
  - TTAch: https://github.com/qubvel/ttach
- Optimizer
  - Ranger: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
  - Ranger21: https://github.com/lessw2020/Ranger21 
  - SAM: https://github.com/davda54/sam
- Loss function
  - MCCE: https://github.com/Kurumi233/Mutual-Channel-Loss
  - FLSD: https://github.com/torrvision/focal_calibration


## Citation
```
@article{
    title  = {Crop classification},
    author = {Yu-Hsi Chen, Jia-Wei Liao, Kuok-Tong Ng},
    url    = {https://github.com/TW-yuhsi/ViT-Orchids-Classification},
    year   = {2022}
}
```
