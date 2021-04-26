# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
# to avoid QT error "Could not connect to any X display."
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str,default="",
                    help='the path to customized image.')
parser.add_argument('--nets', type=str,default="lenet",
                    help='the net for classification')

args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

dst = datasets.CIFAR100("/data/b/yang/pruning_criteria/data/cifar.python", download=True)
# change
tp = transforms.ToTensor()
# tp = transforms.Compose([
#     transforms.Resize(32),
#     transforms.CenterCrop(32),
#     transforms.ToTensor()
# ])

tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)


gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes=100)

from models.vision import LeNet, weights_init, resnet56

if args.nets== "lenet":
    net = LeNet().to(device)
    path = "/data/b/yang/leak/dlg/savefig/"
elif args.nets== "resnet56":
    net = resnet56().to(device)
    path = "/data/b/yang/leak/dlg/savefig/resnet56_"
print("net is {}".format(args.nets))

# plt.imshow(tt(gt_data[0].cpu()))

path=path+"index_"+ str(args.index)+"/"
os.makedirs(path, exist_ok=True)

fig_gt=tt(gt_data[0].cpu())
fig_gt.save(path+ "fig_gt"+'.png')
print("GT label is %d." % gt_label.item(), "\nOnehot label is %d." % torch.argmax(gt_onehot_label, dim=-1).item())


# torch.manual_seed(1234678)
# torch.manual_seed(1234)
# seed=50
for seed in range(50,60,10):
    torch.manual_seed(seed)

    # only apply weight init for lenet here, resnet has init before
    if args.nets== "lenet":
        net.apply(weights_init)

    criterion = cross_entropy_for_onehot
    #
    # import pdb
    # pdb.set_trace()

    # compute original gradient
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    # plt.imshow(tt(dummy_data[0].cpu()))
    fig_dummy=tt(dummy_data[0].cpu())
    fig_dummy.save(path+'fig_dummy_seed_' + str(seed) +'.png')

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    # optimizer = torch.optim.RMSprop([dummy_data, dummy_label],lr=0.1)

    import pdb
    pdb.set_trace()
    history = []
    loss_history = []
    for iters in range(300):
        # print("iter is {}".format(iters))
        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            # print(dummy_pred)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            # print(dummy_loss)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):

                print("gx={} gy={}".format(gx,gy))
                grad_diff += ((gx - gy) ** 2).sum()
            pdb.set_trace()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))
            loss_history.append(current_loss.item())


    # fig, ax = plt.subplots(figsize=(6, 4))
    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        # plt.figtext( 0, -1,)
        plt.annotate("loss="+"%.4f" % loss_history[i], (0, 0), (0, 40))
        plt.axis('off')

    iter_loss_00001=[n for n, i in enumerate(loss_history) if i < 0.0001]

    if isinstance(iter_loss_00001,list) and len(iter_loss_00001)>0:
        iter_loss_00001_id=iter_loss_00001[0]*10
    else:
        iter_loss_00001_id = "None"

    plt.suptitle('Image: {}, Seed: {}, the iteration for loss < 0.001 is: {}'.format(args.index, seed, iter_loss_00001_id))
    plt.savefig(path+"process_" +"seed_"+ str(seed)+ ".png")