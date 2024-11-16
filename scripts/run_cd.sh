#!/usr/bin/env bash
# sleep 14500
gpus=1
checkpoint_root=checkpoints/BCD_TESTAAA2
data_name=BCD

img_size=256
batch_size=8
lr=0.01
max_epochs=200
net_G=DMINet
lr_policy=linear

split=trainval2
split_val=test
project_name=CD_BCD_TESTAAA2

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
