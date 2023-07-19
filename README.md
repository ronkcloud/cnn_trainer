# cnn_trainer

## How to use

### For hyperspectral images

execute `python CNN_trainer_3.py`

### For RGB images

execute `python CNN_trainer_2_RGB.py`

## Important library to install

1. pyTorch, check this link -> https://pytorch.org/get-started/locally/

2. CUDA:
    1. Check your GPU
    2. Check CUDA compatibility, check this link -> https://en.wikipedia.org/wiki/CUDA#GPUs_supported 
    3. Download CUDA version according to your GPU Compute capability in this link -> https://developer.nvidia.com/cuda-toolkit-archive
       
   #### Example
   
    1. GPU: GTX 1080
    2. Compute capability: 6.1, so you can use any CUDA between version 8.0 - 12.2
    3. Chose CUDA version, download and install
