# End to end 3D reconstruction and open vocabulary semantic segmentation
## Input:
* Images
* Text prompt
## Output:
* Reconstructed 3D scene
* Interactive 3D semantic segmentation
![](https://github.com/junyuan-fang/go_vocation/blob/main/reconstruct_lang.gif)
## visualization and segmentation in point cloud
![](https://github.com/junyuan-fang/go_vocation/blob/main/pc_lang.gif)



The implementation is based on DUSt3R and Lseg.

## Get Started
1. Clone 
```bash
git clone --recursive git@github.com:junyuan-fang/go_vocation.git

# If you have already cloned:
# git submodule update --init --recursive
```

2. Create the environment, here we show an example using conda.
```bash
conda env create -f environment.yml 
```

3. Clone feature extraction enabled Lseg
 ```  git clone git@github.com:junyuan-fang/lseg_feature_extraction.git```

## Run
### 1. Feature extraction 
```
cd lang-seg
python3 extract_lseg_features.py
```            
### 2. End to end reconstruction
#### Interactive demo with language,  interactive semantic segmentation
`python3 terminal_demo_global_aligned.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt`


### 2. We also provide a version without opening a port for semantic segmentation, but it is also not an interactive version.
#### Terminal demo with language, without interactive semantic segmentation
`python3 terminal_demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt`

