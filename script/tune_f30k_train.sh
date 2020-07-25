data_name=f30k_precomp
Batch_size=128
DATA_PATH=data
MODEL_DIR=checkpoint/f30k_scan
RESUME=checkpoint/f30k_scan/gpus_full_IMRAM_steps_3_softmax_9/model_best.pth.tar

GPU_ID="0,1,2,3"
iteration_step=3
lambda_softmax=9
model_mode=full_IMRAM #full_IMRAM, text_IMRAM, image_IMRAM

echo "---------------training--------------"
logger_name=${MODEL_DIR}/gpus_${model_mode}_steps_${iteration_step}_softmax_${lambda_softmax}

model_name=${logger_name}

echo ${logger_name}

python train_gpus.py --gpuid ${GPU_ID} --batch_size ${Batch_size} --data_path ${DATA_PATH} --resume ${RESUME} --data_name ${data_name} --vocab_path vocab --logger_name ${logger_name} --model_name ${model_name} --max_violation --bi_gru --agg_func=Mean --lambda_softmax=${lambda_softmax} --model_mode ${model_mode} --iteration_step ${iteration_step} --no_IMRAM_norm