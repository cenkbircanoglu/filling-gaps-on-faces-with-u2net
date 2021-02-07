import os
import random
import shutil

src_masked = '/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/regular_masked_faces/masked'
src_original = '/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/regular_masked_faces/original'

dst_masked = '/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/val/masked'
dst_original = '/Users/cenk.bircanoglu/workspace/personal/img2img/datasets/val/original'

os.makedirs(dst_masked, exist_ok=True)
os.makedirs(dst_original, exist_ok=True)
files = os.listdir(src_masked)
for i in range(10):
    file = random.choice(files)
    src = os.path.join(src_masked, file)
    dst = os.path.join(dst_masked, file)
    shutil.copy(src, dst)
    src = os.path.join(src_original, file)
    dst = os.path.join(dst_original, file)
    shutil.copy(src, dst)
    files.remove(file)