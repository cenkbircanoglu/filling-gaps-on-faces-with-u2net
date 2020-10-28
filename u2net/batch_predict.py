import argparse
import os

import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils

from u2net.data_loader import RescaleT
from u2net.data_loader import SalObjDataset
from u2net.data_loader import ToTensorLab
from u2net.model import U2NET  # full size version 173.6 MB
from u2net.model import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(os.path.join(d_dir, imidx) + '.png')


def main(args):
    # --------- 1. get image path and name ---------
    model_name = args.model_name

    prediction_dir = args.output_dir  # os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = args.model_dir

    os.makedirs(prediction_dir, exist_ok=True)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(data_root=args.data_dir,
                                        image_name_csv_file=args.image_name_csv_file,
                                        img_folder_name='original',
                                        mask_folder_name='masks',
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if model_name == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 3)
    elif model_name == 'u2netp':
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 3)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", test_salobj_dataset.image_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, :, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(test_salobj_dataset.image_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Batch Prediction for U2Net Architecture')
    parser.add_argument('--image_name_csv_file', type=str, help='CSV Path of Image List')
    parser.add_argument('--data_dir', type=str, help='Path to Load the Images')
    parser.add_argument('--output_dir', type=str, help='Path to Store Output Images')
    parser.add_argument('--model_dir', type=str, help='Path to Store the Model')
    parser.add_argument('--model_name', type=str, choices=['u2net', 'u2netp'],
                        help='Specifiy the Deep Learning Architecture', default='u2net')
    args = parser.parse_args()
    main(args)
