#!/usr/bin/env bash
# Tiny CPU smoke run using the synthetic 5-patient fixture in data/iu_xray_smoke.
python main.py \
    --image_dir data/iu_xray_smoke/images/ \
    --ann_path data/iu_xray_smoke/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 30 \
    --threshold 1 \
    --num_workers 0 \
    --batch_size 1 \
    --visual_extractor resnet18 \
    --visual_extractor_pretrained "" \
    --d_vf 512 \
    --num_layers 1 \
    --topk 8 \
    --cmm_size 64 \
    --cmm_dim 512 \
    --sample_method greedy \
    --beam_size 1 \
    --block_trigrams 0 \
    --n_gpu 0 \
    --epochs 1 \
    --save_dir results/iu_xray_smoke/ \
    --record_dir records/iu_xray_smoke/ \
    --log_period 1 \
    --save_period 1 \
    --monitor_metric BLEU_4
