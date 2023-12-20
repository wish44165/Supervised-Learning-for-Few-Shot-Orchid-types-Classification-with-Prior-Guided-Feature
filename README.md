# 尋找花中君子 - 蘭花種類辨識及分類競賽 (TEAM_142)

## Supervised Learning for Few-Shot Orchid types Classification with Prior Guided Feature

$\large{\textbf{Overview}}$

Image classification has been widely used in engineering, agriculture, and medical applications. Nowadays, with the rapid development of deep neural network and the computing power of graphic cards, the performance of image classification has been greatly improved. In particular, pattern recognition plays an important role in image classification. Even though this field looks very mature, there are still some intractable problems, such as lack of labeled data and lack of understanding of species. Here, we followed a pipeline that can find the most suitable process for each task systematically. Using this method, our final scores are 0.911492 and 0.809624582 in public and private datasets, respectively. In addition, our overall ranking is 15th out of 743 teams.


## 1. Environment Setup

<details>

<summary>Hardware information</summary>
  
- CPU: i7-11700F / GPU: GeForce GTX 1660 SUPER™ VENTUS XS OC (6G)
- CPU: i7-10700K / GPU: NVIDIA GeForce RTX 2070 SUPER (8G)
- TWCC GPU: NVIDIA V100 (32G)

</details>



<details>

<summary>Conda environment</summary>
  
```bash
conda create -n ViT python=3.9 -y
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
  
### Usage
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16
```
#### imagenet21k pre-train ####
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

#### imagenet21k pre-train + imagenet2012 fine-tuning ####
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz
```

</details>




<details>

<summary>Folder Structure on Local Machine</summary>

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
    └── apex/
    └── checkpoint/
    └── compare.py
    └── convert.py
    └── models/
    └── utils/
    └── requirements.txt
    └── test.py/
    └── train.py/
    └── submit.py/
```
</details>




## 2. Train and Inference

<details>
 
  <summary>Train</summary>

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

</details>

  
<details>
  
<summary>Train demo</summary>

```
python train.py --name orchid \
                --dataset orchid \
                --foldn 1 \
                --model_type ViT-B_16 \
                --pretrained_dir checkpoint/ViT-B_16.npz \
                --img_size 480 \
                --train_batch_size 4 \
                --optim SGD \
                --learning_rate 3e-2 \ 
                --num_steps 20000 \
                --use_imagenet_mean_std \
                --rot_degree 10 \
                --fliplr 0.5 \
                --loss_fct CE \
                --fp16 \
                --fp16_opt_level O2
```

</details>
  


  
<details>

<summary>Inference</summary>

```
python test.py --model_type <model type> \
               --checkpoint <trained> \
               --img_size <image size> \
               --test_dir <test folder> \
               --foldn <fold n> \
               --dataset <task> \
               --use_imagenet_mean_std \
               --use_test_aug
```

</details>
  
  
  
  
<details>
  
<summary>Inference demo</summary>

```
python test.py --model_type ViT-B_16 \
               --checkpoint output/orchid_ViT-B_16_checkpoint.bin \
               --img_size 480 \
               --test_dir ../data \
               --foldn 1 \
               --dataset test \
               --use_imagenet_mean_std \
               --use_test_aug
```
  
</details>
  

<details>

<summary>Useful commands in Terminal</summary>
  
```bash=
unzip \*.zip    # Unzip all ZIP files
ls -l | grep "^-" | wc -l    # Check the number of files
ls -lR | grep "^-" | wc -l
for i in `seq 0 218`; do ls ${i} -lR | grop "^-" | wc -l; done    # loop under terminal
```
  
</details>

  
  

## 3. Submitted Results
  
<table>
  <tr>
    <td>Filename</td>
    <td>Upload time</td>
    <td>Public score</td>
    <td>Private score</td>
  </tr>
  <tr>
    <td>submit_enEnsemble.csv</td>
    <td>2022-06-06 04:57:23</td>
    <td>0.909891</td>
    <td></td>
  </tr>
  <tr>
    <td>submit_meanEnsemble_convert.csv</td>
    <td>2022-06-06 04:49:39</td>
    <td>0.911492</td>
    <td>0.809624582</td>
  </tr>
  <tr>
    <td>submit_convert_swin_1.csv</td>
    <td>2022-06-06 04:28:35</td>
    <td>0.904925</td>
    <td></td>
  </tr>
  <tr>
    <td>submit_convert.csv</td>
    <td>2022-06-06 03:29:10</td>
    <td>0.901620</td>
    <td></td>
  </tr>
  <tr>
    <td>submit_meanEnsemble_convert.csv</td>
    <td>2022-06-06 03:22:44</td>
    <td>0.890142</td>
    <td></td>
  </tr>



</table>
  
  
  
## 4. Reproduce the Best Result (public: 0.911492, private: 0.809624582)
  
<details>
  
<summary>Google Colaboratory Version</summary>
  
- Step 1. Setup the Folder Structure as follows.
  
  The easiest way is to copy the entire [folder](https://drive.google.com/drive/folders/1x_rb6bu0riJuouAtK-xjFGDkCP7ZbhbL?usp=sharing), but be aware that there is a lot of weights in this folder.

  ```
  尋找花中君子 - 蘭花種類辨識及分類競賽 [TBrain]/
  ├── datasets/
      └── test/
          └── 0/    # orchid_public_set, 40285
          └── 1/    # orchid_private_set, 41425
  ├── Reproduce the Best Result/
      └── ViT/
          └── output/
              └── A1.bin, A2.bin, ID_4.bin, ID_5.bin, ID12.bin, ID27.bin    # ViT-B_16
          └── Reproduce.ipynb
  ```
  
- Step 2. Execute the Notebook named [Reproduce.ipynb](https://colab.research.google.com/drive/1K8_v-LuEhkpefGfIOdvRMUS7zCbOpzlF?usp=sharing).
  
  After the setup, ready to execute [Reproduce.ipynb](https://colab.research.google.com/drive/1K8_v-LuEhkpefGfIOdvRMUS7zCbOpzlF?usp=sharing), no additional steps are needed.
  
- Step 3. Submit the file named `submit_meanEnsemble_convert.csv`.
  
  After finishing [Reproduce.ipynb](https://colab.research.google.com/drive/1K8_v-LuEhkpefGfIOdvRMUS7zCbOpzlF?usp=sharing), we can get the file named `submit_meanEnsemble_convert.csv` which has the highest Macro-F$_1$ score.
  
</details>
  
  
<details>
  
<summary>Local Machine Version</summary>
  
- Step 0. Follow **1. Environment Setup** step by step.
  
- Step 1. Setup the Folder Structure as follows.
  
  ```
  ├── test/
      └── 0/    # orchid_public_set, 40285
      └── 1/    # orchid_private_set, 41425
  ├── ViT-Orchids-Classification-main/
      └── apex/
      └── checkpoint/
      └── compare.py
      └── convert.py
      └── models/
      └── output/
          └── A1.bin, A2.bin, ID_4.bin, ID_5.bin, ID12.bin, ID27.bin    # ViT-B_16
      └── utils/
      └── requirements.txt
      └── test.py/
      └── train.py/
      └── submit.py/
  ```

- Step 2. Execute [submit.py](https://github.com/TW-yuhsi/ViT-Orchids-Classification/blob/main/submit.py) by using the following command.

  After the setup, ready to execute [submit.py](https://github.com/TW-yuhsi/ViT-Orchids-Classification/blob/main/submit.py), no additional steps are needed.

  ```bash
  python submit.py --model_type ["ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16","ViT-B_16"] --checkpoint ["output/A1.bin","output/A2.bin","output/ID_4.bin","output/ID_5.bin","output/ID12.bin","output/ID27.bin"] --img_size [480,480,480,480,480,480] --use_imagenet_mean_std [0,0,0,0,1,1]
  ```
  
  
  
- Step 3. Execute [convert.py](https://github.com/TW-yuhsi/ViT-Orchids-Classification/blob/main/convert.py) by using the following command.
  
  After executing [submit.py](https://github.com/TW-yuhsi/ViT-Orchids-Classification/blob/main/submit.py), we can get two files named `submit_voteEnsemble.csv` and `submit_meanEnsemble.csv`, respectively.
  
  Now, we are ready to execute [convert.py](https://github.com/TW-yuhsi/ViT-Orchids-Classification/blob/main/convert.py).
  
  ```bash
  python convert.py
  ```
  
  
  
- Step 4. Submit the file named `submit_meanEnsemble_convert.csv`.
  
  After executing [convert.py](https://github.com/TW-yuhsi/ViT-Orchids-Classification/blob/main/convert.py), we can get the file named `submit_meanEnsemble_convert.csv` which has the highest Macro-F$_1$ score.
  
</details>
  

  
  

## 5. Related URLs
### [Competition Link](https://tbrain.trendmicro.com.tw/Competitions/Details/20)
### [Google Drive](https://drive.google.com/drive/folders/1x_rb6bu0riJuouAtK-xjFGDkCP7ZbhbL?usp=sharing)
<details>

<summary>Folder Structure on Google Drive</summary>


```
尋找花中君子 - 蘭花種類辨識及分類競賽 [TBrain]/
├── checkpoints/
    └── ResNet/
        └── ResNeSt269/
        └── ResNet50/
        └── ResNet101/
    └── Swin/
    └── ViT/
        └── R50+ViT-B_16/    # 5 weights
        └── ViT_Linformer/
            └── params1/
            └── params2/
        └── ViT-B_16/    # 59 weights
        └── ViT-B_32/    # 3 weights
        └── ViT-L_16/    # 1 weights
        └── ViT-L_32/    # 3 weights
├── Colab Notebooks/
    └── Images/
    └── Ranger-Deep-Learning-Optimizer/
    └── Attention Map.ipynb
    └── ResNet50_3.ipynb
    └── ResNet101_2_Ranger.ipynb
    └── ResNet101_3.ipynb
    └── ResNet101_Ranger_2.ipynb
    └── SwinT_2.ipynb
    └── SwinT.ipynb
    └── ViT_distilled_params1.ipynb
    └── ViT_Linformer_params1.ipynb
    └── ViT_Linformer_params2.ipynb
├── datasets/
    └── test/
        └── 0/    # orchid_public_set, 40285
        └── 1/    # orchid_private_set, 41425
    └── train/
        └── 4-Fold/
            └── fold1/
            └── fold2/
            └── fold3/
            └── fold4/
        └── baseline data/
        └── training/
├── Reproduce the Best Result/
    └── ViT/
        └── output/
        └── Reproduce.ipynb
├── src/
    └── getInfo/
        └── readLabel.py    # read label.csv file
        └── readImage.py    # get the shape of image
    └── preprocessing/
        └── split.py    # split the training data
    └── statistics/
        └── trainLoss.py    # plot training loss curve
├── Submitted Files/
    └── submit_1.csv    # public: 0.890142
    └── submit_2.csv    # public: 0.901620
    └── submit_3.csv    # public: 0.904925
    └── submit_4.csv    # public: 0.911492, private: 0.809624582
    └── submit_5.csv    # public: 0.909891
├── tables/
    └── Baseline.csv/    # experimental results for baseline models
    └── ViT.csv/    # experimental results for whole ViT trials
    └── ViT_Linformer.csv/    # experimental results for ViT_Linformer
```
</details>






## 6. GitHub Acknowledgement
<details>

<summary>Teammate</summary>  
  
- [Jia-Wei Liao](https://github.com/Jia-Wei-Liao/Orchid_Classification)
  
</details>
  

  

<details>

<summary>Useful Techniques</summary>  

- Augmentation
  - [AutoAugment](https://github.com/DeepVoltaire/AutoAugment), [TTAch](https://github.com/qubvel/ttach)
- Optimizer
  - [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer), [Ranger21](https://github.com/lessw2020/Ranger21), [SAM](https://github.com/davda54/sam)
- Loss function
  - [MCCE](https://github.com/Kurumi233/Mutual-Channel-Loss), [FLSD](https://github.com/torrvision/focal_calibration)
- A PyTorch Extension
  - [Apex](https://github.com/NVIDIA/apex)

</details>


## 7. Reference
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)


## Citation
```
@article{
    title  = {Crop classification},
    author = {Yu-Hsi Chen, Jia-Wei Liao, Kuok-Tong Ng},
    url    = {https://github.com/TW-yuhsi/ViT-Orchids-Classification},
    year   = {2022}
}
```
