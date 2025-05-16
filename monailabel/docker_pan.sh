#!/bin/bash

# docker run \
#   -it --rm \
#   --ipc=host \
#   --name monailabel \
#   --net=host \
#   --gpus=all \
#   -e NVIDIA_VISIBLE_DEVICES=2 \
#   -v $PWD/MONAILabel/sample-apps:/workspace \
#   -v /neodata/open_dataset/totalsegmentator_mri/test:/data \
#   -w /workspace/radiology \
#   nanaha1003/monailabel:latest \
#   monailabel start_server \
#     --app /workspace/radiology \
#     --port 8011 \
#     --studies /data \
#     --conf models manafaln_segmentation \
#     --conf mfn_config config/active_learning_pan.yaml


docker run -it --rm --name monailabel --net=host --gpus=all \
  --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=2 \
  -v $PWD:/apps/radiology \
  -v /neodata/open_dataset/totalsegmentator_mri/test:/data \
  nanaha1003/monailabel:0.8.5 \
  monailabel start_server \
    --app /apps/radiology \
    --port 8011 \
    --studies /data \
    --conf models manafaln,sam2,scribbles \
    --conf mfn_config config/active_learning_pan.yaml
