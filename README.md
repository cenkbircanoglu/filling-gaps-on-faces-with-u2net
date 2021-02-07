

# Filling the Gaps on Faces

- Original U2Net Model updated to output 3 channel and loss function altered to L1Loss.

- Training is done on Celeba-HQ dataset for ~150K iteration.

- Final L1Loss on training dataset 0.060407.

- Best model can be found [here.](u2net_gap_filling_for_faces.pth)

## Results

Original Image             |  Masked Image |  Model Result
:-------------------------:|:-------------------------:|:-------------------------:
![](examples/original-1.jpg)  |  ![](examples/masked-1.jpg) |  ![](examples/result-1.jpg)
![](examples/original-2.jpg)  |  ![](examples/masked-2.jpg) |  ![](examples/result-2.jpg)
![](examples/original-3.jpg)  |  ![](examples/masked-3.jpg) |  ![](examples/result-3.jpg)
![](examples/original-4.jpg)  |  ![](examples/masked-4.jpg) |  ![](examples/result-4.jpg)

