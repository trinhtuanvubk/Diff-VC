IMAGE_NAME = diffvc
CONTAINER_NAME = diff-vc-dev
PORT = 1402
GPUS = all 


docker run -itd --gpus $GPUS \
--name $CONTAINER_NAME \
-p $PORT:$PORT \
-v $(pwd)/:/workspace  \
$IMAGE_NAME