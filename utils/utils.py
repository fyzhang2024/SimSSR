#from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from einops import rearrange
#import scipy.io as io
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import metrics

class TrainSetLoader(Dataset):
    def __init__(self, args):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = args.trainset_dir + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
                           str(args.upfactor) + 'x/'
        if args.data_name == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args.data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)
        self.angRes = args.angRes

    def __getitem__(self, index):
        # index = index + 1
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y')) # Lr_SAI_y
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y
            data, label = augmentation(Lr_SAI_y, Hr_SAI_y)
            # hu, wv = data.shape
            data = ToTensor()(data.copy())
            label = ToTensor()(label.copy())

            data = rearrange(data, 'c (u h ) (v w ) -> u v c h w',u=self.angRes,v=self.angRes)
            label = rearrange(label, 'c (u h) (v w) -> u v c h w', u=self.angRes, v=self.angRes)
        return data, label

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None
    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        dataset_dir = args.testset_dir + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
                      str(args.upfactor) + 'x/'
        data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))
    return data_list, test_Loaders, length_of_tests


#
# class TestSetDataLoader(Dataset):
#     def __init__(self, args, data_name = 'ALL', Lr_Info=None):
#         super(TestSetDataLoader, self).__init__()
#         self.angRes = args.angRes
#         self.dataset_dir = args.testset_dir + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
#                       str(args.upfactor) + 'x/' + data_name
#
#         self.file_list = []
#         tmp_list = os.listdir(self.dataset_dir)
#         for index, _ in enumerate(tmp_list):
#             tmp_list[index] = tmp_list[index]
#
#         self.file_list.extend(tmp_list)
#
#         self.item_num = len(self.file_list)
#
#     def __getitem__(self, index):
#         file_name = self.dataset_dir + '/' + self.file_list[index]
#         with h5py.File(file_name, 'r') as hf:
#             data = np.array(hf.get('data'))
#             label = np.array(hf.get('label'))
#             data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
#             data, label = ToTensor()(data.copy()), ToTensor()(label.copy())
#
#         return data, label
#
#     def __len__(self):
#         return self.item_num

class TestSetDataLoader(Dataset):
    def __init__(self, cfg, data_name = 'ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes = cfg.angRes
        self.dataset_dir = cfg.testset_dir + 'SR_' +str(cfg.angRes) + 'x' + str(cfg.angRes) + '_' + str(cfg.upfactor) + 'x/'

        self.data_list = [data_name]
        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)
        self.Lr_Info = self.angRes

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            # Sr_SAI_cbcr = np.array(hf.get('Sr_SAI_cbcr'), dtype='single')
            Hr_SAI_ycbcr = np.array(hf.get('Hr_SAI_y'))
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_ycbcr = np.transpose(Hr_SAI_ycbcr, (1, 0))
            # Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (2, 1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_ycbcr = ToTensor()(Hr_SAI_ycbcr.copy())
        # Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        # return Lr_SAI_y, Hr_SAI_ycbcr, Sr_SAI_cbcr, self.Lr_Info
        return Lr_SAI_y, Hr_SAI_ycbcr

    def __len__(self):
        return self.item_num


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5: # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label

def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out


def LFdivide(lf, patch_size, stride):
    if lf.dim() == 4:
        U, V, H, W = lf.shape
        data = rearrange(lf, 'u v h w -> (u v) 1 h w')
    elif lf.dim() == 5:
        U, V, _, H, W = lf.shape
        data = rearrange(lf, 'u v c h w -> (u v) c h w')

    bdr = (patch_size - stride) // 2
    numU = (H + bdr * 2 - 1) // stride
    numV = (W + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(u v) (c h w) (n1 n2) -> n1 n2 u v c h w',
                      n1=numU, n2=numV, u=U, v=V, h=patch_size, w=patch_size)

    return subLF


def LFintegrate(subLFs, patch_size, stride):
    if subLFs.dim() == 6: # n1, n2, u, v, h, w
        subLFs = subLFs.unsqueeze(4) #n1, n2, u, v, c, h, w

    bdr = (patch_size - stride) // 2
    outLF = subLFs[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 u v c h w -> u v c (n1 h) (n2 w)')

    return outLF.squeeze()


def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np)

def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True,data_range=2.0)

def cal_metrics(img1, img2, angRes):
    if len(img1.size())==2:
        [H, W] = img1.size()
        img1 = img1.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)
    if len(img2.size())==2:
        [H, W] = img2.size()
        img2 = img2.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)

    [U, V, h, w] = img1.size()
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')

    for u in range(U):
        for v in range(V):
            PSNR[u, v] = cal_psnr(img1[u, v, :, :], img2[u, v, :, :])
            SSIM[u, v] = cal_ssim(img1[u, v, :, :], img2[u, v, :, :])
            pass
        pass

    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean



def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y