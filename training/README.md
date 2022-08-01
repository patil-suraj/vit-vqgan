# Training a model

## Download sample data

Refer to [dataset/prepare_dataset.ipynb](/dataset/prepare_dataset.ipynb).

## Train a model

Sample command:

```bash
python train_vit_vqvae.py \
    --output_dir output --overwrite_output_dir \
    --train_folder ../dataset/openimages/train \
    --valid_folder ../dataset/openimages/valid \
    --config_name config/base/model \
    --disc_config_name config/base/discriminator \
    --do_eval --do_train \
    --batch_size 64 \
    --format rgb \
    --optim adam \
    --learning_rate 0.001 --disc_learning_rate 0.001
```
