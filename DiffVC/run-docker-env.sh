IMAGE_NAME = diff-vc
docker run -itd --gpus $GPUS \
--name vutt-dev \
-p 1400-1500:1400-1500 \
-v $(pwd)/workspace:/workspace  \
$IMAGE_NAME