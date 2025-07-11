CUDA_VISIBLE_DEVICES=0 \
python3 \
train.py \
--batchsize 4 \
--savepath "/nas.dbms/ikuto/datag-tool/resources/model_weights" \
--datapath "../gradio_model_dataset/TrainDataset_AI_car+AIcar_explanation500" \
--lr 0.01 \
--epoch 100 \
--wd 5e-4 \
--fr 0.006