#!/bin/bash

pip3 install -e .
pip3 install -r requirements.txt

#pip install torchvision=1.10.0
#pip install torchaudio==0.9.0
# for sm86
#pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

#conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

#mkdir -p $CONDA_PREFIX/etc/conda/activate.d

#echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow-gpu==2.9.2

pip3 install scikit-learn

pip3 install zmq

pip3 install sysv_ipc aiofiles>=0.8.0 aiohttp>=3.6.2 aiosqlite>=0.17.0 anyio>=3.4.0 asyncclick==8.0.1.3 cached-property>=1.5.2