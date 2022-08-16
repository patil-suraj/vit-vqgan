#rm -rf jax_cache

WANDB_NOTES=$(cat ~/instance_name) python train_vit_vqvae.py \
    --wandb_entity craiyon --assert_TPU_available \
    --output_dir output --overwrite_output_dir \
    --train_folder gs://dataset_imgs/vit/train/0005/ \
    --valid_folder gs://dataset_imgs/vit/valid/*/ \
    --config_name config/base/model \
    --disc_config_name config/base/discriminator \
    --do_eval --do_train --dtype bfloat16 \
    --batch_size_per_node 32 --gradient_accumulation_steps 1 \
    --num_train_epochs 20 \
    --optim adam \
    --learning_rate 0.0001 --disc_learning_rate 0.0001 \
    --logging_steps 20 --eval_steps 200 \
    --mp_devices 1 --use_vmap_trick
