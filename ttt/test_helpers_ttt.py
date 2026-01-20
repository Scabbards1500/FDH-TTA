import numpy as np
import torch
import torch.nn as nn
from utils.misc_ttt import *
from utils.rotation_ttt import rotate_batch
from utils.dice_score import dice_loss
import torch.nn.functional as F


def build_model(args):
    from unet.unet_model import UNet
    from unet.SSHead import ExtractorHead
    from unet.unet_model_ttt import UNettt
    print('Building model...')
    classes = 2

    net = UNet(n_channels=3, n_classes=classes).cuda()
    from unet.SSHead import extractor_from_layer2, head_on_layer2
    ext = extractor_from_layer2(net)
    head = head_on_layer2(net, args.width, 4)
    ssh = ExtractorHead(ext, head).cuda()

    if hasattr(args, 'parallel') and args.parallel:
        net = torch.nn.DataParallel(net)
        ssh = torch.nn.DataParallel(ssh)
    return net, ext, head, ssh


def test(dataloader, model, sslabel=None):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    model.eval()
    correct = []
    losses = []

    for batch in dataloader:
        images, true_masks = batch['image'], batch['mask']
        inputs = images.to(device="cuda", dtype=torch.float32, memory_format=torch.channels_last)
        mask = true_masks.to(device="cuda", dtype=torch.long)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, mask)
            diceloss = dice_loss(F.softmax(outputs, dim=1).float(),
                                 F.one_hot(mask, model.n_classes).permute(0, 3, 1, 2).float(),
                                 multiclass=True)
            print("diceloss", diceloss)
            loss += diceloss
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(mask).cpu())
    correct = torch.cat(correct).numpy()
    losses = torch.cat(losses).numpy()
    model.train()
    return 1 - correct.mean(), correct, losses


def test_ssh(dataloader, model, sslabel=None):
    criterion = nn.CrossEntropyLoss().cuda()
    model.eval()
    correct = []
    losses = []

    for batch in dataloader:
        images, true_masks = batch['image'], batch['mask']
        inputs_ssh, labels_ssh = rotate_batch(images, sslabel)
        inputs_ssh = inputs_ssh.to(device="cuda", dtype=torch.float32, memory_format=torch.channels_last)  # 4，3，256，256
        labels_ssh = labels_ssh.to(device="cuda", dtype=torch.long)  # [4, ] #label对用的直接是batchsize

        with torch.no_grad():
            outputs = model(inputs_ssh)
            loss = criterion(outputs, labels_ssh)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels_ssh).cpu())
    correct = torch.cat(correct).numpy()
    # losses = torch.cat(losses).numpy()
    model.train()
    return 1 - correct.mean(), correct, losses


def test_grad_corr(dataloader, net, ssh, ext):
    criterion = nn.CrossEntropyLoss().cuda()
    net.eval()
    ssh.eval()
    corr = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        net.zero_grad()
        ssh.zero_grad()
        inputs_cls, labels_cls = inputs.cuda(), labels.cuda()
        outputs_cls = net(inputs_cls)
        loss_cls = criterion(outputs_cls, labels_cls)
        grad_cls = torch.autograd.grad(loss_cls, ext.parameters())
        grad_cls = flat_grad(grad_cls)

        ext.zero_grad()
        inputs, labels = rotate_batch(inputs, 'expand')
        inputs_ssh, labels_ssh = inputs.cuda(), labels.cuda()
        outputs_ssh = ssh(inputs_ssh)
        loss_ssh = criterion(outputs_ssh, labels_ssh)
        grad_ssh = torch.autograd.grad(loss_ssh, ext.parameters())
        grad_ssh = flat_grad(grad_ssh)

        corr.append(torch.dot(grad_cls, grad_ssh).item())
    net.train()
    ssh.train()
    return corr


def pair_buckets(o1, o2):
    crr = np.logical_and(o1, o2)
    crw = np.logical_and(o1, np.logical_not(o2))
    cwr = np.logical_and(np.logical_not(o1), o2)
    cww = np.logical_and(np.logical_not(o1), np.logical_not(o2))
    return crr, crw, cwr, cww


def count_each(tuple):
    return [item.sum() for item in tuple]


def plot_epochs(all_err_cls, all_err_ssh, fname, use_agg=True):
    import matplotlib.pyplot as plt
    if use_agg:
        plt.switch_backend('agg')

    plt.plot(np.asarray(all_err_cls) * 100, color='r', label='classifier')
    plt.plot(np.asarray(all_err_ssh) * 100, color='b', label='self-supervised')
    plt.xlabel('epoch')
    plt.ylabel('test error (%)')
    plt.legend()
    plt.savefig(fname)
    plt.close()
