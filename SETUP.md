## Create a new EC2 instance

1. AMI: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 1.13.1 (Amazon Linux 2)`
1. Instance type: `g5.*` (depending on how many GPUs you want)
1. Key pair: generate or use an existing one. Below examples assume you use the key `my_key.pem`.
1. Configure storge: 1000GB (since the MedaDepth dataset is large)
1. Log into the instance with SSH (instructions in AWS console under "Connect" button). It's convenient to also attach VSCode to the instance over SSH using the same CLI command. Replace `root@` with `ec2-user@` and use an absolute path for the key file and chmod 600 the file.

## Setup LoFTR

```bash
git clone https://github.com/mattiasarro/LoFTR.git
cd /home/ec2-user/LoFTR
conda env create -f environment.yaml
conda init
# log out of SSH session and log back in (for conda init to take effect)
cd /home/ec2-user/LoFTR
conda activate loftr
```

## Fix scripts

For a train/test script such as `outdoor_ds.sh` you need to run the following steps:

1. `cmod +x scripts/reproduce_train/outdoor_ds.sh`
1. remove `-l` from the first line, so that it's just `#!/bin/bash`
1. comment out parts modifying `PYTHONPATH`
1. set `n_gpus_per_node=1` and `torch_num_workers=1` or an appropriate number that matches the number of GPUs of the instance you created

## Get the train data

Run on the EC2 instance:

```bash
# to ensure we can SCP files into the machine, run on the ec2 instance
sed -i '1i [ -z "$PS1" ] && return' ~/.bashrc

# download megadepth
nohup wget https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz &
```

The above takes >2h since it's a 200GB dataset. `nohup` and trailing `&` ensure it continues to download if your ssh connection times out. Once it's finished extract it:

```bash
mkdir -p /home/ec2-user/LoFTR/data/megadepth_v1 && tar -xzvf MegaDepth_v1.tar.gz -C /home/ec2-user/LoFTR/data/megadepth_v1
ln -sv /home/ec2-user/LoFTR/data/megadepth_v1/phoenix /home/ec2-user/LoFTR/data/megadepth/train
ln -sv /home/ec2-user/LoFTR/data/megadepth_v1/phoenix /home/ec2-user/LoFTR/data/megadepth/test
```

## Get the test data

1. Go to https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf
1. Right-click on the three dots of `testdata` and click "Download"
1. Right-click on the three dots of `train-data` and click "Download"
1. Double-click on `weights` and download these 1-by-1.
1. Upload the downloaded files to the EC2 instance. Note that due to how Drive downloads the files or new versions uploaded by the authors, the file names might be slightly different - adjust the steps in this and the following section accordingly.

Run on localhost:

```bash
KEY_LOC="/Users/m/code/smfl/loftr/mattias_crunch_london.pem" # TODO change this
EC2_HOST="ec2-13-40-101-148.eu-west-2.compute.amazonaws.com" # TODO change this
scp -i $KEY_LOC ~/Downloads/testdata-20240628T083243Z-001.zip ec2-user@$EC2_HOST:/home/ec2-user/LoFTR/data/
scp -i $KEY_LOC ~/Downloads/scannet_indices-001.tar ec2-user@$EC2_HOST:/home/ec2-user/LoFTR/data/
scp -i $KEY_LOC ~/Downloads/train-data-20240628T083709Z-003.zip ec2-user@$EC2_HOST:/home/ec2-user/LoFTR/data/
scp -i $KEY_LOC ~/Downloads/cfg_1513_-1_0.2_0.8_0.15_reduced_v2-002.tar ec2-user@$EC2_HOST:/home/ec2-user/LoFTR/data/
scp -i $KEY_LOC ~/Downloads/outdoor_ot.ckpt ec2-user@$EC2_HOST:/home/ec2-user/LoFTR/data/
scp -i $KEY_LOC ~/Downloads/outdoor_ds.ckpt ec2-user@$EC2_HOST:/home/ec2-user/LoFTR/data/
scp -i $KEY_LOC ~/Downloads/indoor_ot.ckpt ec2-user@$EC2_HOST:/home/ec2-user/LoFTR/data/
scp -i $KEY_LOC ~/Downloads/indoor_ds_new.ckpt ec2-user@$EC2_HOST:/home/ec2-user/LoFTR/data/
```

### Extract the data to correct locations

Run on $EC2_HOST:

```bash
cd ~/LoFTR/data
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
cd ~/LoFTR
mkdir weights
mv data/*.ckpt weights/
```

## Run the tests

Run on $EC2_HOST:

```bash
cd ~/LoFTR
conda activate loftr
./scripts/reproduce_test/indoor_ds_new.sh # fixed as described in "Fix scripts" section
```

Runs the "ds" (dual softmax) eval (similar "ot" script is for Optimal Transport)

## Run training

Run on $EC2_HOST:

```bash
cd ~/LoFTR
conda activate loftr
python fix_datasets.py # run this once, after megadepth is downloaded and extracted
./scripts/reproduce_train/outdoor_ds.sh # fixed as described in "Fix scripts" section
```

We need to use the outdoor dataset, since for this we have the megadepth dataset.
