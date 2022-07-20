#!/bin/bash
activation='sine'
data_root='data'
dataset='KODAK'
downscaling_factor=1
encoding='nerf'
encoding_scale=1.4
epochs=25000
epochs_til_ckpt=5000
exp_root='exp/basic_kodak'
ff_dims=16
hidden_layers=3
l1_reg=1.0e-05
lr=0.0005
patience=500
steps_til_summary=1000

#Overfitting
for hidden_dims in 32 48 64 96 128
do
    python image_compression/overfitINR.py \
    --activation $activation \
    --data_root $data_root \
    --dataset $dataset \
    --downscaling_factor $downscaling_factor \
    --encoding $encoding \
    --encoding_scale $encoding_scale \
    --epochs $epochs \
    --epochs_til_ckpt $epochs_til_ckpt \
    --exp_root $exp_root \
    --hidden_layers $hidden_layers \
    --hidden_dims $hidden_dims \
    --ff_dims $ff_dims \
    --l1_reg $l1_reg \
    --lr $lr \
    --patience $patience \
    --steps_til_summary $steps_til_summary
done

#Quantization, AdaRound, Retraining

exp_glob='*'
adaround_iterations=1000
adaround_reg=0.0001
code='arithmetic'
retrain_epochs=300
retrain_lr=1.0e-06

for bitwidth in 7 8
do
    python image_compression/quantize_and_test.py \
    --exp_glob "$exp_glob" \
    --exp_root $exp_root \
    --data_root $data_root \
    --dataset $dataset \
    --adaround_iterations $adaround_iterations \
    --adaround_reg $adaround_reg \
    --retrain_epochs $retrain_epochs \
    --retrain_lr $retrain_lr \
    --code $code \
    --bitwidth $bitwidth

done