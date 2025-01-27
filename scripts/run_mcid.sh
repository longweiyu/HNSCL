#!/usr/bin/env bash

export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=$1
dataset='mcid'
bert_model_path="/home/long/models/bert-base-uncased"
tokenizer_path="/home/long/models/bert-base-uncased"
for labeled_ratio in 0.1 
do
    for known_cls_ratio in 0
    do
        for topk in 20
        do
            python train.py \
                --data_dir 'data' \
                --dataset ${dataset} \
                --known_cls_ratio ${known_cls_ratio} \
                --labeled_ratio ${labeled_ratio} \
                --seed 0 \
                --lr_pre '5e-5' \
                --lr '1e-5' \
                --save_results_path 'outputs' \
                --update_per_epoch 5 \
                --topk ${topk} \
                --report_pretrain \
                --num_pretrain_epochs 100 \
                --num_train_epochs 100 \
                --pretrain_batch_size 64 \
                --train_batch_size 64 \
                --bert_model ${bert_model_path} \
                --tokenizer ${tokenizer_path}
        done
    done
done 