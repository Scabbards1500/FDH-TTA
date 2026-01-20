import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.nn as nn

from utils.data_loading import BasicDataset
from unet import UNet

from utils.utils import plot_img_and_mask
from glob import glob
import torch.optim as optim

from utils import tent
from utils import memorytent
from utils import ourmemorytent



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--ground-truth', '-g', metavar='GROUND_TRUTH', nargs='+', help='Filenames of ground truth masks')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--method', '-me', type=str, default='source', help='Method for adaptation')
    return parser.parse_args()

diceloss = []
def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_folder = args.input[0]  # Assuming the first argument is the input folder
    in_mask_folder = args.ground_truth[0] if args.ground_truth else None  # Ground truth mask folder, if specified
    out_folder = args.output[0] if args.output else None  # Output folder, if specified

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    model.to(device=device)


    ####测试我们的

    state_dict = torch.load(r"D:\python\UNet-TTA\checkpoints_RITE_ttt\checkpoint_epoch30.pth", map_location=device)
    print(state_dict)
    mask_values = [0, 1]
    model.load_state_dict(state_dict['net'], strict=False)




    # # ####原有的
    # state_dict = torch.load(r"D:\python\UNet-TTA\checkpoints_RITE\checkpoint_epoch20.pth", map_location=device)
    # print(state_dict)
    # mask_values = state_dict.pop('mask_values', [0, 1])
    # model.load_state_dict(state_dict,strict=False)

    if args.method == 'source':
        model = setup_source(model)
    if args.method == 'tent':
        model = setup_tent(model)
    if args.method == 'memorytent':
        print("memorytent")
        model = setup_memorybank(model)
    if args.method == 'ourmemorytent':
        print("ourtent")
        model = setup_ourmemorybank(model)
    logging.info('Model loaded!')


    # model.reset()
    # print("resetting model")


    in_files = get_image_files(in_folder)
    in_mask_files = get_image_files(in_mask_folder) if in_mask_folder else None

    start = time.time()
    for i, filename in enumerate(in_files):

        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        gt_mask = Image.open(in_mask_files[i]) if in_mask_files else None

        mask = predict_img(model=model,
                           full_img=img,
                           mask_img=gt_mask,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = os.path.join(out_folder, f'{os.path.basename(filename).split(".")[0]}_OUT.png') \
                if out_folder else f'{os.path.splitext(filename)[0]}_OUT.png'

            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

    print(f"average dice score: {np.mean(diceloss)}")
    logging.info(f'Inference done! Time elapsed: {time.time() - start:.2f} seconds')

def get_image_files(folder_path):
    jpg = glob(os.path.join(folder_path, '*.png'))
    png = glob(os.path.join(folder_path, '*.jpg'))
    return  jpg + png


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))

def dice_score(pred, target):
    smooth = 0.5
    num = pred.size(0)
    # 将灰度标签转换为二值标签
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    dice_coeff = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return dice_coeff



def predict_img(model, full_img, device, scale_factor=1, out_threshold=0.5, mask_img = None):
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    groundtruth = torch.from_numpy(BasicDataset.preprocess([0,255], mask_img, scale_factor, is_mask=True))
    groundtruth = groundtruth.unsqueeze(0)
    groundtruth = groundtruth.to(device=device, dtype=torch.float32)



    mask_img = torch.tensor(np.array(mask_img), dtype=torch.float32).unsqueeze(0)



    model.groundtruth = groundtruth
    output = model(img).cpu()
    output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
    # if net.n_classes > 1:
    #     print(">1")
    #     mask = output.argmax(dim=1)
    # else:
    #     print("<1")
    #     mask = torch.sigmoid(output) > out_threshold
    mask = output.argmax(dim=1)
    masks_pred = mask[0].long().float().unsqueeze(0)

    # 自写
    dice_loss_mask_img = dice_score(mask_img, masks_pred)
    print("dice_loss_mask_img: ", dice_loss_mask_img)
    diceloss.append(dice_loss_mask_img)


    return mask[0].long().squeeze().numpy()

#tent---------------------------------------------------------------
def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # print(f"model for evaluation: %s", model)
    return model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model,optimizer,
                           steps=1,
                           episodic=False)
    return tent_model


def setup_memorybank(model):
    model = memorytent.configure_model(model)
    params, param_names = memorytent.collect_params(model)
    optimizer = setup_optimizer(params)
    mbtt_model = memorytent.Tent(model, optimizer,
                           steps=1,
                           episodic=False)
    return mbtt_model

def setup_ourmemorybank(model):
    model = ourmemorytent.configure_model(model)
    params, param_names = ourmemorytent.collect_params(model)
    optimizer = setup_optimizer(params)
    our_mbtt_model = ourmemorytent.Tent(model, optimizer,
                           steps=1,
                           episodic=False)
    return our_mbtt_model

def setup_optimizer(params, optimizer_method='Adam', lr=0.001, beta=0.9, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if optimizer_method == 'Adam':
        return optim.Adam(params,
                    lr=lr,
                    betas=(beta, 0.999),
                    weight_decay=0.0)
    else:
        raise NotImplementedError

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    seed = 42  # 你可以选择任何你喜欢的整数作为种子值
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # (可选) 如果你的代码会用到CUDA，你也可以设置CUDA的随机种子
    torch.cuda.manual_seed(seed)
    # (可选) 设置NumPy的随机种子
    np.random.seed(seed)
    main()