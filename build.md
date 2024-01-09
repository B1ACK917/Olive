## Conda

```
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

conda create -n torch
conda activate torch
```

## Torch

```
git clone https://github.com/pytorch/pytorch.git
# git clone https://github.com/B1ACK917/pytorch.git
```



## Dependencies

```
conda install cmake ninja
pip install -r requirements.txt

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} USE_CUDA=0
USE_XNNPACK=0
python setup.py develop
```

