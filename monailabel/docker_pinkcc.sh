#!/bin/bash

docker run -it --rm --name monailabel --net=host --gpus=all \
  --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=2 \
  -v $PWD:/apps/radiology \
  -v /neodata/open_dataset/PINKCC/PINKCC_phase1_data:/data \
  nanaha1003/monailabel:0.8.5 \
  monailabel start_server \
    --app /apps/radiology \
    --port 8011 \
    --studies /data \
    --conf models manafaln,sam2,scribbles \
    --conf mfn_config config/active_learning_pinkcc_segresnet.yaml,config/active_learning_pan.yaml \
