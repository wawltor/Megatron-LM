#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

export MASTER_PORT=60039
export CUDA_VISIBLE_DEVICES=7
DATA_PATH="input_path/my-gpt2_text_document"
CHECKPOINT_PATH=./save_steps


python pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 4 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 200000 \
       --lr-decay-iters 160000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --no-scaled-masked-softmax-fusion\
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
