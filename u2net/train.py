import argparse
import os
from math import ceil
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from u2net.data_loader import RandomCrop
from u2net.data_loader import RescaleT
from u2net.data_loader import SalObjDataset
from u2net.data_loader import ToTensorLab
from u2net.early_stopping import EarlyStopping
from u2net.model import U2NET
from u2net.model import U2NETP

# ------- 1. define loss function --------


bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #    loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))

    return loss0, loss


def load_dataloaders(args, batch_sizes=[24, 6, 2, 1], selected_image_sizes=[284, 568, 853, 1137]):
    dataloaders = []
    for batch_size, selected_image_size in zip(batch_sizes, selected_image_sizes):
        dataset = SalObjDataset(
            data_root=args.data_dir,
            image_name_csv_file=args.image_name_csv_file,
            img_folder_name='original',
            mask_folder_name='masks',
            transform=transforms.Compose([
                RescaleT(selected_image_size),
                RandomCrop(ceil(selected_image_size - (selected_image_size / 10))),
                ToTensorLab(flag=0)])
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=8, pin_memory=True)
        dataloaders.append(dataloader)
    return dataloaders


def train(args):
    # ------- 2. set the directory of training dataset --------
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    model_name = args.model_name  # 'u2net'  # 'u2netp'
    model_dir = args.model_dir

    os.makedirs(model_dir, exist_ok=True)

    epoch_num = args.epochs

    batch_sizes = [24]  # [12, 6, 2, 1]
    selected_image_sizes = [284]  # [284, 568, 853, 1137]
    if torch.cuda.is_available():
        batch_sizes = [i * torch.cuda.device_count() for i in batch_sizes]
    dataloaders = load_dataloaders(args, batch_sizes=batch_sizes, selected_image_sizes=selected_image_sizes)
    # ------- 3. define model --------
    # define the net
    if model_name == 'u2net':
        net = U2NET(3, 3)
    elif model_name == 'u2netp':
        net = U2NETP(3, 3)

    if args.warm_start:
        net.load_state_dict(torch.load(args.warm_start, map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    print('Training is starting')
    for epoch in range(0, epoch_num):
        net.train()
        for dataloader in dataloaders:
            for data in tqdm(dataloader, total=int(len(dataloader.dataset) / dataloader.batch_size)):
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                inputs, labels = data['image'], data['label']
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_v, labels_v = Variable(inputs.cuda(non_blocking=True), requires_grad=False), Variable(
                        labels.cuda(non_blocking=True), requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                # y zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)

                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

                loss.backward()
                optimizer.step()

                # # print statistics
                running_loss += loss.item()
                running_tar_loss += loss2.item()

                # del temporary outputs and loss
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        early_stopping(running_loss, net)
        torch.save(net.state_dict(), model_dir + '/' + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
            ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        running_loss = 0.0
        running_tar_loss = 0.0
        net.train()  # resume train
        ite_num4val = 0
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Training for U2Net Architecture')
    parser.add_argument('--image_name_csv_file', type=str, help='CSV Path of Image List')
    parser.add_argument('--data_dir', type=str, help='Path to Load the Images and Masks')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of the images for each batch')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=25, help='Patience for early stopping')
    parser.add_argument('--image_size', type=int, default=888, help='Resize images to this value')
    parser.add_argument('--model_dir', type=str, help='Path to Store the Model')
    parser.add_argument('--warm_start', type=str, help='Path to Reload the Trained Model')
    parser.add_argument('--model_name', type=str, choices=['u2net', 'u2netp'],
                        help='Specifiy the Deep Learning Architecture', default='u2net')

    args = parser.parse_args()
    train(args)
