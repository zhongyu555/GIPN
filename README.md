# GIP

**💡 This is the official implementation of the paper "Graph Interaction Prompt Network for Few-shot
Medical Image Anomaly Detection".  


## 🔧 Installation

```
conda create -n HGINet python=3.8
conda activate HGINet
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim==0.3.9
mim install mmcv-full==1.7.0 mmengine==0.8.4
pip install mmsegmentation==0.30.0 timm h5py einops fairscale imageio fvcore pysodmetrics
pip install Pillow==9.1.1
pip install scikit-image==0.19.3  
pip install scikit-learn==1.1.2
pip install sklearn==0.0
pip install opencv-python==4.6.0.66
pip install grad_cam==1.4.3   
pip install tqdm==4.61.2
pip install PyYAML==6.0
pip install easydict==1.9
pip install ftfy==6.1.3
pip install ==2023.12.25
pip install imgaug==0.4.0
pip install numpy==1.22.4
```  
### Data preparation 
Download the following datasets:
- **BUSI  [[Baidu Cloud (pwd8866)]](https://pan.baidu.com/s/1EVt96fExiqrvMQslPDRRRg?pwd=8866)   [[Google Drive]](https://drive.google.com/file/d/1PyvMXdNEVY86BY1PV8yKhPVS30TAmS6X/view?usp=drive_link)  [[Official Link]](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)**  
- **BrainMRI  [[Baidu Cloud (pwd8866)]](https://pan.baidu.com/s/1--5vPMN-eTqePPYjpKTwvA?pwd=8866)  [[Google Drive]](https://drive.google.com/file/d/1kldE-5_wXaN-JR_8Y_mRCKQ6VZiyv3km/view?usp=drive_link)  [[Official Link]](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)**  
- **CheXpert [[Baidu Cloud (pwd8866)]](https://pan.baidu.com/s/15-V5wobA_7ICvZAXBraDGA?pwd=8866)  [[Google Drive]](https://drive.google.com/file/d/1pVYRipGC2VqjYP-wHdDFR-lLf7itLiUi/view?usp=drive_link)  [[Official Link]](https://stanfordmlgroup.github.io/competitions/chexpert/)**  

Unzip them to the `data`. Please refer to [data/README](data/README.md).  
  
##  Experiments

To train the GIPN on the BrainMRI dataset with the support set size is 16:  
```
$ python  train.py --config_path config/brainmri.yaml  --k_shot 16
```  
   
To test the MediCLIP on the BrainMRI dataset:  
```
$ python  test.py --config_path config/brainmri.yaml  --checkpoint_path xxx.pkl
```  
Replace ``xxx.pkl`` with your checkpoint path.

---
Code reference: **[[CLIP]](https://github.com/OpenAI/CLIP)**  **[[CoOp]](https://github.com/KaiyangZhou/CoOp)** **[[MediCLIP]](https://github.com/cnulab/MediCLIP)** **[[HGINet]](https://github.com/Garyson1204/HGINet)**.


