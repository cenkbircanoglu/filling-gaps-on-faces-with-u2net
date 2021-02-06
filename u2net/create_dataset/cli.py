import numpy as np
from PIL import Image

from u2net.create_dataset.data_loader import dataloader
import os

img_file = '/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/web'
mask_file = 'none'
mask_type = [0, 1]
load_size = [1024 + 10, 1024 + 10]
fine_size = [1024, 1024]
is_train = False
resize_or_crop = 'resize_and_crop'
no_augment = True
no_flip = True
no_rotation = True
batch_size = 16
no_shuffle = False
n_threads = 1
counter = 0

os.makedirs('/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/web/original/', exist_ok=True)
os.makedirs('/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/web/masked/', exist_ok=True)
for batch in dataloader(img_file, mask_file, load_size, fine_size, is_train, resize_or_crop, no_augment, no_flip,
                        no_rotation, mask_type, batch_size, no_shuffle, n_threads):
    for i in range(batch['img'].shape[0]):
        img = batch['img'][i] * batch['mask'][i]
        np_arr = img.numpy().transpose(1, 2, 0)
        img = Image.fromarray(np.uint8(np_arr * 255))
        new_path = batch['img_path'][i].replace('web/', 'web/masked/').replace('.jpg',
                                                                                                f'-{counter}.jpg')
        img.save(new_path)

        img = batch['img'][i]
        np_arr = img.numpy().transpose(1, 2, 0)
        img = Image.fromarray(np.uint8(np_arr * 255))
        new_path = batch['img_path'][i].replace('web/', 'web/original/').replace(
            '.jpg', f'-{counter}.jpg')
        img.save(new_path)
        counter += 1
        print(counter)
