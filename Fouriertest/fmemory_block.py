import numpy as np
from PIL import Image
from Fourier_Tans import *
import scipy.misc
import torchvision.transforms as transforms
import imageio
from numpy import amin, amax, ravel, asarray, cast, arange, \
     ones, newaxis, transpose, mgrid, iscomplexobj, sum, zeros, uint8, \
     issubdtype, array
import matplotlib.pyplot as plt
# 这个是测试我们自己的傅里叶的
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

im_src = Image.open(r"D:\python\FDA-master\demo_images\source.png").convert('RGB')
im_trg = Image.open(r"D:\python\FDA-master\demo_images\target.png").convert('RGB')

# im_src = Image.open(r"D:\tempdataset\TTADataset\CHASE\test\images\Image_01L.png").convert('RGB')
# im_trg = Image.open(r"D:\tempdataset\TTADataset\RITE\test\images\05_test.png").convert('RGB')


im_src = im_src.resize( (512,512), Image.BICUBIC)
im_trg = im_trg.resize( (512,512), Image.BICUBIC)

# im_src = np.asarray(im_src, np.float32)
# im_trg = np.asarray(im_trg, np.float32)

transform = transforms.ToTensor()
# 将图像转换为张量
im_src = transform(im_src) #3,512,512
im_trg = transform(im_trg)

# fft_src =torch.fft.fftn(im_src, dim=(-2, -1))
# fft_trg =torch.fft.fftn(im_trg, dim=(-2, -1))

im_src = im_src.unsqueeze(0)
im_trg = im_trg.unsqueeze(0)

src_m, src_p = FDA_get_amp_pha_tensor(im_src) # 1, 3,512,512
trg_m, trg_p = FDA_get_amp_pha_tensor(im_trg) # 1, 3,512,512


src_in_trg = arc_add_amp(src_m, trg_m,  src_p,0.01)
# src_in_trg = FDA_target_to_source( im_src, im_trg, L=0.01 )



# tensor转图像！
# 定义转换
transform = transforms.ToPILImage()
# 将张量转换为图像
image2 = transform(src_in_trg.squeeze(0))
# 显示图像
plt.imshow(image2)
plt.axis('off')  # 不显示坐标轴
plt.show()