docker run \
  -it --rm \
  --ipc=host \
  --name monailabel \
  --net=host \
  --gpus=all \
  -e NVIDIA_VISIBLE_DEVICES=1 \
  -v $PWD/MONAILabel/sample-apps:/workspace \
  -v /data2/open_dataset/AMOS22/imagesVa:/data \
  -w /workspace/radiology \
  nanaha1003/monailabel:latest \
  monailabel start_server \
    --app /workspace/radiology \
    --port 8011 \
    --studies /data \
    --conf models manafaln_segmentation \
    --conf mfn_config config/active_learning_pan.yaml
