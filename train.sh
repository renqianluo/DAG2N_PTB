#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="$(pwd)"

mkdir -p log_ptb

fixed_arc="0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"

nohup python main.py \
  --search_for="enas" \
  --reset_output_dir \
  --data_path="data/ptb.pkl" \
  --output_dir="outputs_ptb" \
  --batch_size=64 \
  --bptt_steps=35 \
  --num_epochs=2000 \
  --fixed_arc="${fixed_arc}" \
  --rhn_depth=12 \
  --num_layers=1 \
  --lstm_hidden_size=748 \
  --lstm_e_keep=0.79 \
  --lstm_x_keep=0.25 \
  --lstm_h_keep=0.75 \
  --lstm_o_keep=0.24 \
  --nochild_lstm_l_skip \
  --grad_bound=0.25 \
  --lr=20.0 \
  --rnn_slowness_reg=1e-3 \
  --l2_reg=5e-7 \
  --lr_dec_start=14 \
  --lr_dec_every=1 \
  --lr_dec_rate=0.9991 \
  --lr_dec_min=0.001 \
  --optim_algo="sgd" \
  --log_every=50 \
  --eval_every_epochs=1 > log_ptb/train.log 2>&1 &

