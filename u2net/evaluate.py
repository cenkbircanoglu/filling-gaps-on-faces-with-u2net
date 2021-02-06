import argparse
from statistics import mean

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils
from tqdm import tqdm

from u2net.batch_predict import normPRED
from u2net.data_loader import RescaleT
from u2net.data_loader import SalObjDataset
from u2net.data_loader import ToTensorLab
from u2net.model import U2NET  # full size version 173.6 MB
from u2net.model import U2NETP  # small version u2net 4.7 MB
from u2net.train import muti_bce_loss_fusion, l1_loss


def iou(pred, target, n_classes=2):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return ious


def main(args):
    # --------- 1. get image path and name ---------
    model_name = args.model_name

    model_dir = args.model_dir

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(data_root=args.data_dir,
                                        image_name_csv_file=args.image_name_csv_file,
                                        img_folder_name='masked',
                                        mask_folder_name='original',
                                        transform=transforms.Compose([RescaleT(800),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if model_name == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    accuracies = []
    ite_num4val = 0
    running_loss = 0.
    running_tar_loss = 0.

    mious = []
    for data_test in tqdm(test_salobj_dataloader):
        inputs, labels = data_test['image'], data_test['label']
        ite_num4val = ite_num4val + inputs.shape[0]
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs_test = Variable(inputs)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
        loss2, loss = l1_loss(d1, labels)

        running_loss += loss.item()
        running_tar_loss += loss2.item()

        # TODO check this calculation / To increase the batch size this calculation must be updated
        pred = d1[:, :, :, :]
        pred = normPRED(pred)
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        total = labels.size(1) * labels.size(2) * labels.size(3)
        correct = (pred == labels).sum().item()
        print('Pixel Accuracy %s' % (correct / total))
        accuracies.append(correct / total)
        print('Avg. Pixel Accuracy %s ' % mean(accuracies))
        print("loss: %3f, tar: %3f " % (running_loss / ite_num4val, running_tar_loss / ite_num4val))
        mious.extend(iou(pred, labels))
        print('Avg. IoU %s ' % mean(mious))
    print("Accuracies Above 0.80", len(list(filter(lambda x: x >= 0.80, accuracies))) / len(accuracies))
    print("Accuracies Above 0.90", len(list(filter(lambda x: x >= 0.90, accuracies))) / len(accuracies))
    print("Accuracies Above 0.95", len(list(filter(lambda x: x >= 0.95, accuracies))) / len(accuracies))
    print("Accuracies Above 0.99", len(list(filter(lambda x: x >= 0.99, accuracies))) / len(accuracies))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Batch Prediction for U2Net Architecture')
    parser.add_argument('--image_name_csv_file', type=str, help='CSV Path of Image List')
    parser.add_argument('--data_dir', type=str, help='Path to Load the Images')
    parser.add_argument('--model_dir', type=str, help='Path to Store the Model')
    parser.add_argument('--model_name', type=str, choices=['u2net', 'u2netp'],
                        help='Specifiy the Deep Learning Architecture', default='u2net')
    args = parser.parse_args()
    main(args)
