

# Img2Img



```bash
nohup python -m u2net.train     \
  --data_dir ./datasets/regular_masked_faces     \
  --model_dir ./checkpoints/u2net_regular_masked_faces/ \
  --model_name u2net > logs/regular_masked_faces.log &
```


```bash
python -m u2net.batch_predict \
    --data_dir ./datasets/val \
    --output_dir ./outputs/u2net_regular_masked_faces/ \
    --model_dir ./checkpoints/u2net_regular_masked_faces/u2net_bce_itr_22068_train_0.442945_tar_0.039415.pth  \
    --model_name u2net
```