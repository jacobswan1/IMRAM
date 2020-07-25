data_name=f30k_precomp
Batch_size=128
DATA_PATH=data
MODEL_DIR=checkpoint/f30k_scan

GPU_ID="1"
iteration_step=1
lambda_softmax=9
model_mode=full_IMRAM #full_IMRAM, text_IMRAM, image_IMRAM

logger_name=${MODEL_DIR}/gpus_${model_mode}_steps_${iteration_step}_softmax_${lambda_softmax}

model_name=${logger_name}

echo ${logger_name}

echo "---------------evaluation--------------"
MODEL_PATH=${model_name}/model_23.pth.tar
echo ${MODEL_PATH}

SPLIT=test

python test_gpus.py --gpuid ${GPU_ID} --model_path ${MODEL_PATH} --data_path ${DATA_PATH} --split ${SPLIT}
