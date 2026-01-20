import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from utils.data_loading import BasicDataset, CarvanaDataset
from pathlib import Path
from torch.utils.data import DataLoader, random_split

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(*NORM)])
mnist_transforms = transforms.Compose([transforms.Resize((32, 32)),
										transforms.ToTensor(),
										transforms.Normalize((0.1307,), (0.3081,))])




def prepare_data(dir_img,dir_mask):
	# try:
	print("CarvanaDataset")
	dataset = CarvanaDataset(dir_img, dir_mask, 0.5)
	# except (AssertionError, RuntimeError, IndexError):
	# print("BasicDataset")
	# dataset = BasicDataset(dir_img, dir_mask, 0.5)

	# 2. Split into train / validation partitions
	# n_train = len(dataset)
	# data_set, _ = random_split(dataset, [n_train], generator=torch.Generator().manual_seed(0))
	loader_args = dict(batch_size=1, num_workers=2, pin_memory=True)
	data_loader = DataLoader(dataset, shuffle=True, **loader_args)

	return dataset,data_loader


	