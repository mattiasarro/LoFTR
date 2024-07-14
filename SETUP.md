## Setup

This setup assumes we have an Ubuntu machine with a compatible GPU. This was tested on Ubuntu 22.04.4 with NVIDIA RTX 6000 Ada GPUs.

We use the following env vars (inside the Ubuntu machine and localhost), which you should customise:

```bash
export UBUNTU_HOST=""
export UBUNTU_USER=""
export UBUNTU_LOFTR_HOME="/home/LoFTR"
```

## Setup CUDA

Run on `$UBUNTU_HOST`:

```bash
sudo apt-get install linux-headers-$(uname -r)
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get install linux-headers-5.19.0-1010-nvidia-lowlatency
sudo apt-get install -y cuda-12-3 nvidia-gds

# might not be needed on first install, but I had to run:
sudo apt-get install --reinstall nvidia-driver-555
sudo reboot
```

Ensure these env vars are set in your shell (e.g. add them to .bashrc):

```bash
export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## Setup pyenv

Install and init pyenv by running the below commands (or look for a possibly updated way to do it in the docs). 

Run on `$UBUNTU_HOST`:

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

## Setup LoFTR 

Run on `$UBUNTU_HOST`:

```bash
cd $UBUNTU_LOFTR_HOME/..
git clone https://github.com/mattiasarro/LoFTR.git
cd LoFTR
pyenv virtualenv 3.8.10 loftr
pyenv activate loftr
pip install -r requirements.txt
```

## Fix scripts

Run on `$UBUNTU_HOST`:

1. `cmod +x scripts/reproduce_train/outdoor_ds.sh`
1. remove `-l` from the first line, so that it's just `#!/bin/bash`
1. comment out parts modifying `PYTHONPATH`
1. set `n_gpus_per_node=1` and `torch_num_workers=1` or an appropriate number that matches the number of GPUs of the instance you created

## Get the train data

Run on `$UBUNTU_HOST`:

```bash
# to ensure we can SCP files into the machine, run on the ec2 instance
sed -i '1i [ -z "$PS1" ] && return' ~/.bashrc

# download megadepth
nohup wget https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz &
```

The above takes >2h since it's a 200GB dataset. `nohup` and trailing `&` ensure it continues to download if your ssh connection times out. Once it's finished extract it:

```bash
mkdir -p $UBUNTU_LOFTR_HOME/data/megadepth_v1 && tar -xzvf MegaDepth_v1.tar.gz -C $UBUNTU_LOFTR_HOME/data/megadepth_v1
ln -sv $UBUNTU_LOFTR_HOME/data/megadepth_v1/phoenix $UBUNTU_LOFTR_HOME/data/megadepth/train
ln -sv $UBUNTU_LOFTR_HOME/data/megadepth_v1/phoenix $UBUNTU_LOFTR_HOME/data/megadepth/test
```

## Get the test data

On localhost (macOS):

1. Go to https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf
1. Right-click on the three dots of `testdata` and click "Download"
1. Right-click on the three dots of `train-data` and click "Download"
1. Double-click on `weights` and download these 1-by-1.
1. Upload the downloaded files to the EC2 instance. Note that due to how Drive downloads the files or new versions uploaded by the authors, the file names might be slightly different - adjust the steps in this and the following section accordingly.

On localhost (macOS):

```bash
scp ~/Downloads/testdata-20240628T083243Z-001.zip $UBUNTU_USER@$UBUNTU_HOST:$UBUNTU_LOFTR_HOME/data/
scp ~/Downloads/scannet_indices-001.tar $UBUNTU_USER@$UBUNTU_HOST:$UBUNTU_LOFTR_HOME/data/
scp ~/Downloads/train-data-20240628T083709Z-003.zip $UBUNTU_USER@$UBUNTU_HOST:$UBUNTU_LOFTR_HOME/data/
scp ~/Downloads/cfg_1513_-1_0.2_0.8_0.15_reduced_v2-002.tar $UBUNTU_USER@$UBUNTU_HOST:$UBUNTU_LOFTR_HOME/data/
scp ~/Downloads/outdoor_ot.ckpt $UBUNTU_USER@$UBUNTU_HOST:$UBUNTU_LOFTR_HOME/data/
scp ~/Downloads/outdoor_ds.ckpt $UBUNTU_USER@$UBUNTU_HOST:$UBUNTU_LOFTR_HOME/data/
scp ~/Downloads/indoor_ot.ckpt $UBUNTU_USER@$UBUNTU_HOST:$UBUNTU_LOFTR_HOME/data/
scp ~/Downloads/indoor_ds_new.ckpt $UBUNTU_USER@$UBUNTU_HOST:$UBUNTU_LOFTR_HOME/data/
```

### Extract the data to correct locations

Run on `$UBUNTU_HOST`:

```bash
cd $UBUNTU_LOFTR_HOME/data
unzip train-data-20240628T083709Z-003.zip
unzip testdata-20240628T083243Z-001.zip
mv cfg_1513_-1_0.2_0.8_0.15_reduced_v2-002.tar train-data/
mv scannet_indices-001.tar train-data/
tar xf train-data/megadepth_indices.tar
mv megadepth_indices/* megadepth/index/
tar xf train-data/scannet_indices-001.tar
mv scannet_indices/* scannet/index/
tar xf testdata/megadepth_test_1500.tar
mv megadepth_test_1500/Undistorted_SfM/* megadepth/test/
tar xf testdata/scannet_test_1500.tar
mv scannet_test_1500/* scannet/test/
cd ..
mkdir weights
mv data/*.ckpt weights/
```

## Run the tests

Run on `$UBUNTU_HOST`:

```bash
cd $UBUNTU_LOFTR_HOME
pyenv activate loftr
./scripts/reproduce_test/indoor_ds_new.sh # fixed as described in "Fix scripts" section
```

Runs the "ds" (dual softmax) eval (similar "ot" script is for Optimal Transport)

## Run training

Run on `$UBUNTU_HOST`:

```bash
cd $UBUNTU_LOFTR_HOME
pyenv activate loftr
mkdir data/megadepth/index/scene_info_0.1_0.7_no_sfm/
python fix_datasets.py # run this once, after megadepth is downloaded and extracted
./scripts/reproduce_train/outdoor_ds.sh # fixed as described in "Fix scripts" section
```

We need to use the outdoor dataset, since for this we have the megadepth dataset.
