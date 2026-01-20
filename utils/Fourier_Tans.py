import torch
import numpy as np

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha


def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    amp_src = torch.fft.fftshift ( amp_src, dim=(-2, -1))
    amp_trg = torch.fft.fftshift(amp_trg, dim=(-2, -1))
    _, _, h, w = amp_src.size()
    b = int(min(h, w) * L)  # get b 中间那一块的
    c_h = h // 2
    c_w = w // 2
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1
    amp_src[:, :,h1:h2,w1:w2] = amp_trg[:, :,h1:h2,w1:w2]
    amp_src = torch.fft.ifftshift(amp_src, dim=(-2, -1))
    return amp_src


def low_freq_mutate2( amp_src, amp_trg, L=0.1 ):
    amp_src = torch.fft.fftshift ( amp_src, dim=(-2, -1))
    amp_trg = torch.fft.fftshift(amp_trg, dim=(-2, -1))
    _, h, w = amp_src.size()
    b = int(min(h, w) * L)  # get b 中间那一块的
    c_h = h // 2
    c_w = w // 2
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1
    amp_src[:,h1:h2,w1:w2] = amp_trg[:,h1:h2,w1:w2]
    amp_src = torch.fft.ifftshift(amp_src, dim=(-2, -1))
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )
    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1
    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_target_to_source(src_img, trg_img, L=0.01):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src =torch.fft.fftn(src_img, dim=(-2, -1))  # 在图像的最后两个维度上执行傅里叶变换
    fft_trg =torch.fft.fftn(trg_img, dim=(-2, -1))  # 在图像的最后两个维度上执行傅里叶变换

    # extract amplitude and phase of both ffts
    amp_src = torch.abs(fft_src)  # 直接从FFT结果获取幅度谱
    pha_src = torch.angle(fft_src)  # 直接从FFT结果获取相位谱
    amp_trg = torch.abs(fft_trg)
    pha_trg = torch.angle(fft_trg)

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )  # 使用low_freq_mutate处理4维输入

    # # recompose fft of source
    constant = amp_src_
    fre_ = constant * torch.exp(1j * pha_src)  # 把幅度谱和相位谱再合并为复数形式的频域图数据
    src_in_trg = torch.abs(torch.fft.ifftn(fre_, dim=(-2, -1)))  # 还原为空间域图像

    return src_in_trg

def FDA_target_to_source_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img  #.cpu().numpy()
    trg_img_np = trg_img  #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


def FDA_get_amp_pha_np(trg_img):
    # exchange magnitude
    # input: src_img, trg_img

    trg_img_np = trg_img  #.cpu().numpy()
    # get fft of both source and target
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )
    # extract amplitude and phase of both ffts
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_trg, pha_trg


def FDA_get_amp_pha_tensor(img):
    # 进行傅里叶变换
    fre = torch.fft.fftn(img, dim=(-2, -1))
    fre_m = torch.abs(fre)   # 幅度谱，求模得到
    fre_p = torch.angle(fre) # 相位谱，求相角得到
    return fre_m, fre_p


def arc_add_amp(amp_src,amp_trg,pha_src,L):
    # amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    fre_ = amp_src_ * torch.exp(1j * pha_src)  # 把幅度谱和相位谱再合并为复数形式的频域图数据
    src_in_trg = torch.abs(torch.fft.ifftn(fre_, dim=(-2, -1)))  # 还原为空间域图像


    return src_in_trg