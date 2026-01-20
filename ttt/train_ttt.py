import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate_ssh
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

from utils.prepare_dataset_ttt import *
from misc_ttt import *
from utils.test_helpers_ttt import *
from rotation_ttt import *



# # #CHASE
# dir_img = Path(r'D:\tempdataset\TTADataset\CHASE\train\images512')
# dir_mask = Path(r'D:\tempdataset\TTADataset\CHASE\train\masks')
# dir_checkpoint = Path('checkpoints_CHASE_ttt/')

##########RITE
# dir_img = Path(r'D:\tempdataset\TTADataset\RITE\train\images5122')
# dir_mask = Path(r'D:\tempdataset\TTADataset\RITE\train\masks5122')
# dir_checkpoint = Path('checkpoints_RITE_ttt2/')

#HRF
# dir_img = Path(r'D:\tempdataset\TTADataset\HRF\train\images512')
# dir_mask = Path(r'D:\tempdataset\TTADataset\HRF\train\masks51222')
# dir_checkpoint = Path('checkpoints_HRF/')


# # # Retina
# dir_img = Path(r"D:\tempdataset\TTADataset\Retina\train\images")
# dir_mask = Path(r"D:\tempdataset\TTADataset\Retina\train\masks")
# dir_checkpoint = Path('checkpoints_test/')

#retina_Hrf
# dir_img = Path(r"D:\tempdataset\tooth_aug\train\images")
# dir_mask = Path(r"D:\tempdataset\tooth_aug\train\masks")
# dir_checkpoint = Path('checkpoints_toothaug/')


# # CHASE_RITE
# dir_img = Path(r'D:\tempdataset\TTADataset\CHASE_RITE\images')
# dir_mask = Path(r'D:\tempdataset\TTADataset\CHASE_RITE\masks')
# dir_checkpoint = Path('checkpoints_CHASE_RITE/')

# #CHASE_HRF
# dir_img = Path(r'D:\tempdataset\TTADataset\CHASE_HRF\images')
# dir_mask = Path(r'D:\tempdataset\TTADataset\CHASE_HRF\masks')
# dir_checkpoint = Path('checkpoints_CHASE_HRF40/')
#
#RITE_HRF
# dir_img = Path(r"D:\tempdataset\TTADataset\RITE_HRF\images")
# dir_mask = Path(r"D:\tempdataset\TTADataset\RITE_HRF\masks")
# dir_checkpoint = Path('checkpoints_RITE_HRF/')

# # teeth_unaug
# dir_img = Path(r"D:\tempdataset\tooth_unaug\train\image")
# dir_mask = Path(r'D:\tempdataset\tooth_unaug\train\mask2')
# dir_checkpoint = Path('checkpoints_teeth_unaug/')

# # # teeth_aug
# dir_img = Path(r"D:\tempdataset\tooth_aug\train\image")
# dir_mask = Path(r'D:\tempdataset\tooth_aug\train\mask2')
# dir_checkpoint = Path('checkpoints_teeth_aug/')

# MoNuSeg
dir_img = Path(r"D:\tempdataset\TTADataset\MoNuSeg\train\images")
dir_mask = Path(r'D:\tempdataset\TTADataset\MoNuSeg\train\masks')
dir_checkpoint = Path('checkpoints_MoNuSeg_ttt/')

#Kumar
dir_img = Path(r"D:\tempdataset\TTADataset\TNBC\train\images")
dir_mask = Path(r'D:\tempdataset\TTADataset\TNBC\train\masks')
dir_checkpoint = Path('checkpoints_TNBC_ttt/')



#CPM







os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.test_helpers_ttt import *
from rotation_ttt import *

import os
from pathlib import Path
from misc_ttt import *
import multiprocessing
import torch.optim as optim
from prepare_dataset_ttt import *

def train_model(
        model,
        model2,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-6,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))


    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        ssh.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask'] # image [1,3,256,256] mask [1,256,256]

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long) #1,256,256

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images) #1,2,256,256
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        print("here2")
                        loss = criterion(masks_pred, true_masks)
                        diceloss = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        loss += diceloss

                    #这边是训练ssh的了
                    inputs_ssh, labels_ssh = rotate_batch(images, args.rotation_type)
                    inputs_ssh, labels_ssh = inputs_ssh.to(device), labels_ssh.to(device)
                    outputs_ssh = ssh(inputs_ssh)
                    loss_ssh = criterion(outputs_ssh, labels_ssh)
                    loss += loss_ssh


                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate_ssh(model,ssh, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state = {'mask_values':dataset.mask_values,
                     'net': model.state_dict(),
                     'ssh': ssh.state_dict()}

            torch.save(state, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    parser.add_argument('--shared', default="layer2")
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--group_norm', default=0, type=int)

    parser.add_argument('--milestone_1', default=50, type=int)
    parser.add_argument('--milestone_2', default=65, type=int)
    parser.add_argument('--rotation_type', default='rand')
    parser.add_argument('--outf', default='.')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available(),"cuda")
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model, ext, head, ssh = build_model(args)

    model = model.to(memory_format=torch.channels_last)
    ssh = ssh.to(memory_format=torch.channels_last)



    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    ssh.to(device=device)

    try:
        train_model(
            model=model,
            model2= ssh,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        ssh.use_checkpointing()
        train_model(
            model=model,
            model2=ssh,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )


