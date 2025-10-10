# [CVPR2025] 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models
[Wanhua Li*](https://li-wanhua.github.io/), [Renping Zhou*](https://github.com/zrporz), [Jiawei Zhou](https://joezhouai.com/), [Yingwei Song](https://github.com/wrencanfly), [Johannes Herter](https://www.linkedin.com/in/johannes-herter-48a549155/), [Minghan Qin](https://github.com/minghanqin), [Gao Huang†](https://www.gaohuang.net/), [Hanspeter Pfister†](https://seas.harvard.edu/person/hanspeter-pfister) \
(* indicates equal contribution, † means Co-corresponding author) \
| [Project page](https://4d-langsplat.github.io) | [Full Paper](https://arxiv.org/abs/2503.10437) | [Video](https://youtu.be/L2OzQ91eRG4) |\
| Datasets Annotations | [Google Drive](https://drive.google.com/drive/folders/1C-ciHn38vVd47TMkx2-93EUpI0z4ZdZW?usp=sharing) | [BaiduWangpan](https://pan.baidu.com/s/1ZMOk0UFQ39WJ7TtTXy9gkA?pwd=g9rg)\
| Pretrained Model | [Google Drive](https://drive.google.com/drive/folders/1-G8I5cJCD66fjpvejUzF9QPRJU_GNxj0?usp=sharing) | [BaiduWangpan](https://pan.baidu.com/s/1TmBW1ZjZfjLQTGxpDXZzlg?pwd=3kmw)\
| Pregenerated Point Clouds by COLMAP | [Google Drive](https://drive.google.com/drive/folders/1_JOObfpXrCq3v_NYKwDt6vRHIbb0oVek?usp=sharing) | [BaiduWangpan](https://pan.baidu.com/s/15jDvS-zSW7pfdvzdwP32mQ?pwd=9y2u)
<img src="./assets/teaser.png"> 
This repository contains the official implementation of the paper "4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models" (CVPR 2025).

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">😊LangSplat Family</h2>

  <pre><code>@inproceedings{li20254d,
  title={4d langsplat: 4d language gaussian splatting via multimodal large language models},
  author={Li, Wanhua and Zhou, Renping and Zhou, Jiawei and Song, Yingwei and Herter, Johannes and Qin, Minghan and Huang, Gao and Pfister, Hanspeter},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={22001--22011},
  year={2025}
}</code></pre>
<p><strong>🎉 Our work is based on LangSplat, and we thank them for their contributions!</strong> 
    This work ground CLIP features into a set of 3D language Gaussians, which attains precise 3D language fields while being <strong>199 ×</strong> faster than LERF. <a href="https://langsplat.github.io/" target="_blank" style="text-decoration: underline;">[CVPR 2024] LangSplat </a>
<pre><code>@inproceedings{qin2024langsplat,
  title={Langsplat: 3d language gaussian splatting},
  author={Qin, Minghan and Li, Wanhua and Zhou, Jiawei and Wang, Haoqian and Pfister, Hanspeter},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20051--20060},
  year={2024}
}</code></pre>

  <p><strong>🎉 We have released LangSplat V2!</strong>  
  The new version significantly improves performance, achieving over <strong>450+ FPS</strong> in rendering.  <a href="https://langsplat-v2.github.io/" target="_blank" style="text-decoration: underline;">[NeurIPS 2025] LangSplat V2</a>
  </p>

  <pre><code>@article{li2025langsplatv2,
  title={LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS},
  author={Li, Wanhua and Zhao, Yujie and Qin, Minghan and Liu, Yang and Cai, Yuanhao and Gan, Chuang and Pfister, Hanspeter},
  journal={arXiv preprint arXiv:2507.07136},
  year={2025}
}</code></pre>
  
  </div> 
</section>

  
  </div>
</section>

## BibTeX
```
@inproceedings{li20254dlangsplat4dlanguage,
    title={4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models}, 
    author={Wanhua Li and Renping Zhou and Jiawei Zhou and Yingwei Song and Johannes Herter and Minghan Qin and Gao Huang and Hanspeter Pfister},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```
## Cloning the Repository
The repository contains submodules, thus please check it out with
```bash
git clone git@github.com:zrporz/4DLangSplat.git --recursive
```

## Setup
4D LangSplat uses the following software versions:
- Python 3.10
- CUDA 12.4
- GCC 10.2.0

On default, run the following commands to install the relative packages
```bash
conda create -n 4DLangSplat python=3.10
conda activate 4DLangSplat
pip install -r requirements.txt
### submodules for gaussian rasterization ###
pip install -e submodules/simple-knn
pip install -e submodules/4d-langsplat-rasterization
### submodules for generate segmentation map ###
pip install -e submodules/4d-langsplat-tracking-anything-with-deva
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Prepare Datasets
Our models are trained and evaluated on [HyperNeRF](https://github.com/google/hypernerf) and [Neu3D](https://github.com/facebookresearch/Neural_3D_Video) datasets. Please follow their instructions to prepare your dataset, or run the following commands:
```bash
bash scripts/download_hypernerf.sh data/hypernerf
bash scripts/download_neu3d.sh data/neu3d
```

To evaluate the rendering results, we use [RoboFlow](https://roboflow.com/) to annotate the datasets. The annotations can be accessed through this link: [Download the Annotations](https://drive.google.com/drive/folders/1C-ciHn38vVd47TMkx2-93EUpI0z4ZdZW?usp=sharing). \
Follow [4DGaussians](https://github.com/hustvl/4DGaussians), we use COLMAP to generate the point clouds. Please follow their pipeline, or use ours: [Download the Point Clouds](https://drive.google.com/drive/folders/1_JOObfpXrCq3v_NYKwDt6vRHIbb0oVek?usp=sharing)

Then put them under `data/<hypernerf or neu3d>/<dataset name>`. You need to ensure that the data folder is organized as follows:
```
|——data
|   | hypernerf
|       | americano
|           |——annotations
|               |——train
|               |——README
|               |——video_annotations.json
|           |——camera
|           |——rgb
|               |——1x
|                   |——000001.png
|                   ...
|               |——2x        
|               ...
|           |——dataset.json
|           |——metadata.json
|           |——points.npy
|           |——scene.json
|           |——points3D_downsample2.ply
|       |——chickchicken
|       ...
|   | neu3d
|       | coffee_martini
|           |——annotations
|               |——train
|               |——README
|           |——cam00
|               |——images
|                   |——0000.png
|                   ...
|           |——cam01
|           ...
|           |——cam00.mp4
|           |——cam01.mp4
|           ...
|           |——poses_bounds.npy
|           |——points3D_downsample2.ply
|      |——cur_roasted_beef
|      ...
```

## QuickStart
We provide the pretrained checkpoints of gaussian model and autoencoder: [Download Pretrained Checkpoint](https://drive.google.com/drive/folders/1-G8I5cJCD66fjpvejUzF9QPRJU_GNxj0?usp=sharing).

For HyperNeRF dataset, take `americano` as an example. Put checkpoint folder upder the  `output/hypernerf/americano` and run the following commands for rendering and evaluation
```bash
bash scripts/render-hypernerf.sh
bash scripts/eval-hypernerf.sh
```
For Neu3D dataset, take `coffee_martini` as an example. Put checkpoint folder under the  `output/neu3d/coffee_martini` and run the following commands for rendering and evaluation
```bash
bash scripts/render-neu3d.sh
bash scripts/eval-neu3d.sh
```

The evaluation results will be saved under `eval/eval_results`.

## Training Guide
### Step 1: Generate Segmentation Map using DEVA
First execute the demo script to generate segmentation maps:
```bash
cd submodules/4d-langsplat-tracking-anything-with-deva
bash scripts/download_models.sh # Download the model parameters if you are a first time user 
bash scripts/demo-chickchicken.sh
```
The output segmentation maps will be saved in `submodules/4d-langsplat-tracking-anything-with-deva/output`

### Step 2: Extract CLIP and Video Features
Extract CLIP features:
```bash
bash scripts/extract_clip_features.sh
```
Generate video features:
```bash
bash scripts/generate-video-feature.sh
```
These commands will create two feature directories under your dataset path:
- `clip_features`: Extracted by CLIP model
- `video_features`: Extracted by E5 model

### Step 3: Train and Evaluate 4D LangSplat
Run the training and evaluation script:
```bash
bash scripts/train_eval.sh
```
This will train the 4D LangSplat field and perform evaluation.

## TODO list
- [x] release the code of the 4d-langsplat-rasterization
- [x] release the code of the 4d-langsplat-tracking-anything-with-deva
- [x] release the code of the evaluation
- [x] release the code of the autoencoder
- [x] release the code of preprocessing
- [x] release the code of training
- [x] release the the pretrained model
- [ ] release the preprocessed dataset
- [x] update the arxiv link
