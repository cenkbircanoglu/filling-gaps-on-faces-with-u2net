import numpy as np
from PIL import Image

from u2net.create_dataset.data_loader import dataloader
import os

img_file = '/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/data128x128'
mask_file = 'none'
mask_type = [0, 1, 2]
load_size = [128 + 10, 128 + 10]
fine_size = [128, 128]
is_train = True
resize_or_crop = 'resize_and_crop'
no_augment = False
no_flip = False
no_rotation = False
batch_size = 4
no_shuffle = False
n_threads = 1
counter = 0

os.makedirs('/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/128x128/original/', exist_ok=True)
os.makedirs('/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/128x128/masked/', exist_ok=True)
for batch in dataloader(img_file, mask_file, load_size, fine_size, is_train, resize_or_crop, no_augment, no_flip,
                        no_rotation, mask_type, batch_size, no_shuffle, n_threads):
    for i in range(batch['img'].shape[0]):
        img = batch['img'][i] * batch['mask'][i]
        np_arr = img.numpy().transpose(1, 2, 0)
        img = Image.fromarray(np.uint8(np_arr * 255))
        new_path = batch['img_path'][i].replace('datasets/data128x128/', 'datasets/128x128/masked/').replace('.jpg',
                                                                                                             f'-{counter}.jpg')
        img.save(new_path)

        img = batch['img'][i]
        np_arr = img.numpy().transpose(1, 2, 0)
        img = Image.fromarray(np.uint8(np_arr * 255))
        new_path = batch['img_path'][i].replace('datasets/data128x128/', 'datasets/128x128/original/').replace(
            '.jpg', f'-{counter}.jpg')
        img.save(new_path)
        counter += 1
        print(counter)
