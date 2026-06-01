import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from utils import print_log
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import torch.nn as nn
from torch.autograd import Variable
from utils import print_log
from numpy import ndarray
import pandas as pd
from skimage import measure
from statistics import mean
from sklearn.metrics import auc
from scipy.ndimage import gaussian_filter

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def denormalization(x):
    # mean = np.array([0, 0, 0])
    # std = np.array([1, 1, 1])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x


def denormalization1(x):
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    #mean = np.array([0.1818, 0.1818, 0.1818])
    #std = np.array([0.1917, 0.1917, 0.1917])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x


def generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity):
    # 生成随机梯度向量
    # np.random.seed(seed)
    # out = torch.zeros(width, height)
    gradients = np.random.normal(size=(width, height, 2))

    # 生成坐标网格
    x = np.linspace(0, scale, num=width, endpoint=False)
    y = np.linspace(0, scale, num=height, endpoint=False)
    x_grid, y_grid = np.meshgrid(x, y)

    # 初始化Perlin噪声
    noise = np.zeros((width, height))
    frequency = 1
    amplitude = 1

    # 叠加不同频率的噪声
    for _ in range(octaves):
        # 计算当前频率上的Perlin噪声
        dx = x_grid / scale * frequency
        dy = y_grid / scale * frequency
        gx = np.floor(dx).astype(int)
        gy = np.floor(dy).astype(int)
        px = dx - gx
        py = dy - gy

        d00 = np.sum(gradients[gx % width, gy % height] * np.stack([dx, dy], axis=2), axis=2)
        d10 = np.sum(gradients[(gx + 1) % width, gy % height] * np.stack([dx - 1, dy], axis=2), axis=2)
        d01 = np.sum(gradients[gx % width, (gy + 1) % height] * np.stack([dx, dy - 1], axis=2), axis=2)
        d11 = np.sum(gradients[(gx + 1) % width, (gy + 1) % height] * np.stack([dx - 1, dy - 1], axis=2), axis=2)

        wx = (3 - 2 * px) * px ** 2
        wy = (3 - 2 * py) * py ** 2

        d0 = d00 * (1 - wx) + d10 * wx
        d1 = d01 * (1 - wx) + d11 * wx
        # b = np.sin(dx * frequency + dy * frequency) ** 2
        noise += (d0 * (1 - wy) + d1 * wy) * amplitude  # * b

        # 更新振幅和频率
        amplitude *= persistence
        frequency *= lacunarity

    # 归一化Perlin噪声到[0, 1]范围
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    #noise = torch.from_numpy(noise)
    return noise


def generate_mask(height, width, channels):
    mask = np.zeros((height, width, channels), dtype=int)
    start = width // 4
    end = 3 * width // 4
    mask[start:end, :, :] = 1
    return mask


def apply_perlin_noise(image, noise, threshold):
    # Apply threshold
    #perlin_thr = noise
    perlin_thr = np.where(noise > threshold, 1.0, 0.0)
    width = image.shape[0]
    # Apply mask to each channel of the image
    perlin_thr = np.stack([perlin_thr] * image.shape[2], axis=-1)  # 扩展为与图像通道数相同的形状
    mask = generate_mask(image.shape[0], image.shape[1], image.shape[2])
    mask = perlin_thr * mask
    mask[:width // 4, :, :] = 1
    mask[3 * width // 4:, :, :] = 1
    #kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size
    #mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    # Apply the mask to the image
    result_image = mask * image# + 255 * (1 - mask)
    mask_np = mask
    #print(mask_np.shape)


    return result_image.astype(np.uint8), mask_np.astype(np.uint8)


def torch_process(image):
    _, _, width, height = image.shape
    scale = 20  #10
    octaves = 3  #2
    persistence = 0.5 #1
    lacunarity = 8 #10
    threshold = 0.5
    rotated_list = []
    for i in range(image.size(0)):
        numpy_image = image[i].numpy()
        numpy_image = np.transpose(numpy_image, axes=(1, 2, 0))
        noise = generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity)
        noisy_image, _ = apply_perlin_noise(numpy_image, noise, threshold)
        noisy_image = torch.from_numpy(noisy_image)
        patch_img = noisy_image.permute(2, 0, 1)
        patch_img = torch.unsqueeze(patch_img, dim=0).float()
        rotated_list.append(patch_img)
    #
    patch_img = torch.cat(rotated_list, dim=0)  # 拼接为原来的张量
    return patch_img

def get_max(image):
    rotated_list = []
    n, c, h, w = image.shape
    for i in range(n):
        img = image[i]
        img = img.unsqueeze(0)
        max_values, max_indices = torch.max(img.view(-1, c, h * w), dim=2)
        max_channel_index = max_indices[0].argmax()
        max_channel_tensor = img[0, max_channel_index, :, :]
        max_channel_tensor = max_channel_tensor.unsqueeze(0)
        max_channel_tensor = max_channel_tensor.unsqueeze(0)
        rotated_list.append(max_channel_tensor)
    img_noise = torch.cat(rotated_list, dim=0)
    return img_noise

def get_KDmap(image1, image2, device):
    #H = 256
    n, c, h, w = image1.shape
    max_list = []
    min_list = []
    all =[]
    for i in range(n):
        a = image1[i].unsqueeze(dim=0)
        b = image2[i].unsqueeze(dim=0)
        anomaly_map1_kd = torch.ones(1, h, w).to(device) - F.cosine_similarity(a, b)
        anomaly_map1_kd = anomaly_map1_kd.unsqueeze(dim=0)
        all.append(anomaly_map1_kd)

    anomaly_map = torch.cat(all, dim=0)

    return anomaly_map
    #     anomaly_map = anomaly_map.unsqueeze(dim=0)
    #     #print(anomaly_map.shape)
    #     max_map = get_max(anomaly_map)
    #     mean_map = get_mean(anomaly_map)
    #     max_list.append(max_map)
    #     min_list.append(mean_map)
    #
    # max = torch.cat(max_list, dim=0)
    # mean = torch.cat(min_list, dim=0)
    #
    # return max, mean


def get_upsample(a1, a2, a3,):
    h = 64
    w = 64
    #a1 = F.interpolate(a1, size=(h, w), mode='bilinear', align_corners=True)
    a2 = F.interpolate(a2, size=(h, w), mode='bilinear', align_corners=True)
    a3 = F.interpolate(a3, size=(h, w), mode='bilinear', align_corners=True)
    # b1 = F.interpolate(b1, size=(h, w), mode='bilinear', align_corners=True)
    # b2 = F.interpolate(b2, size=(h, w), mode='bilinear', align_corners=True)
    # b3 = F.interpolate(b3, size=(h, w), mode='bilinear', align_corners=True)

    #a = (a1 + a2 + a3) / 3
    #b = (b1 + b2 + b3) / 3
    ano_map = torch.cat([a1, a2, a3], dim=1)
    return ano_map


class SegmentCrossEntropyLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, logit, gt_mask):
        #gt_mask = input["mask"]
        #logit = input["logit"]
        bsz, _, h, w = logit.size()
        logit = logit.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.criterion(logit, gt_mask)


def get_mean(image):
    rotated_list = []
    n, _, _, _ = image.shape
    for i in range(n):
        img = image[i]
        img = img.unsqueeze(0)
        output_tensor = img.mean(dim=1, keepdim=True)
        rotated_list.append(output_tensor)
    img_noise = torch.cat(rotated_list, dim=0)
    return img_noise


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, feature1, feature2):
        cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
        ano_map = torch.ones_like(cos) - cos
        loss = (ano_map.view(ano_map.shape[0], -1).mean(-1)).mean()
        return loss

class loss_normal(nn.Module):
    def __init__(self):
        super(loss_normal, self).__init__()

    def forward(self, feature_s, feature_t):
        loss_type = torch.nn.CosineSimilarity()
        loss = 0.0
        for i in range(len(feature_s)):
            cos = 1 - loss_type(feature_s[i], feature_t[i])
            loss_i = torch.mean(cos)
            loss += loss_i

        return loss


class loss_aug(nn.Module):
    def __init__(self):
        super(loss_aug, self).__init__()

    def forward(self, feature_s, feature_t, anomaly_mask, device, n):
        cos_loss = torch.nn.CosineSimilarity()
        loss_type = torch.nn.L1Loss(reduction='mean')
        #loss_type = torch.nn.MSELoss(reduction='mean')
        loss = 0.0
        for i in range(len(feature_s)):
            with torch.no_grad():
                anomaly_mask = F.interpolate(anomaly_mask, size=feature_s[i].shape[-2], mode='bilinear',
                                             align_corners=True)
                anomaly_mask = torch.where(
                    anomaly_mask < 0.5, torch.zeros_like(anomaly_mask), torch.ones_like(anomaly_mask)
                )

            # cos = cos_loss(
            #     feature_s[i],
            #     feature_t[i])
            anomaly = torch.ones(n, feature_s[i].shape[-1], feature_s[i].shape[-1]).to(device) - F.cosine_similarity(feature_s[i], feature_t[i])
            anomaly = anomaly.unsqueeze(1)
            #cos = torch.unsqueeze(1 - cos, dim=1)
            loss += loss_type(anomaly, anomaly_mask)

        return loss

class loss_distil(nn.Module):
    def __init__(self):
        super(loss_distil, self).__init__()

    def forward(self, feature_s, feature_t):
        loss_type = torch.nn.CosineSimilarity()
        loss = 0.0
        for i in range(len(feature_s)):
            loss_i = torch.mean(1 - loss_type(
                feature_s[i].view(feature_s[i].shape[0], -1),
                feature_t[i].view(feature_t[i].shape[0], -1)))
            loss += loss_i
        return loss

class loss_fucntion(nn.Module):
    def __init__(self):
        super(loss_fucntion, self).__init__()

    def forward(self, a, b):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))

        loss = loss / (len(a))
        return loss


class loss_shuffle(nn.Module):
    def __init__(self):
        super(loss_shuffle, self).__init__()
        self.cos_loss = torch.nn.CosineSimilarity()

    def forward(self, a, b):
        current_batchsize = a[0].shape[0]
        shuffle_index = torch.randperm(current_batchsize)
        total_loss = 0
        for i in range(len(a)):
            shuffle = a[i][shuffle_index]
            normal_proj = b[i]
            loss = torch.mean(1 - self.cos_loss(normal_proj.view(normal_proj.shape[0], -1), shuffle.view(shuffle.shape[0], -1)))
            total_loss += loss
        return total_loss



class single_loss(nn.Module):
    def __init__(self):
        super(single_loss, self).__init__()

    def forward(self, a, b):
        cos_loss = loss_fucntion()
        loss = cos_loss(a[0], b[0]) + cos_loss(a[1], b[1]) + cos_loss(a[2], b[2])
        return loss/3


class EarlyStop():
    """Used to early stop the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=True, delta=0, save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = 0


    def __call__(self, val_loss, encoder, bn, decoder, proj, log, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, encoder, bn, decoder, proj, log)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print_log((f'EarlyStopping counter: {self.counter} out of {self.patience}'), log)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.save_checkpoint(val_loss, encoder, bn, decoder, proj, log)
            self.counter = 0
            #self.alpha = learnable_alpha

        return self.early_stop

    def save_checkpoint(self, val_loss, encoder, bn, decoder, proj, log):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print_log((f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'),
                      log)
        state = {'encoder':encoder.state_dict(), 'bn': bn.state_dict(), 'decoder': decoder.state_dict(),
                 'proj':proj.state_dict(), 'best_epoch': self.best_epoch}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss


def cut(img, t, b):
    # h, w, c = img.shape
    x = np.random.randint(0, img.shape[1] - t)
    y = np.random.randint(0, img.shape[0] - b)
    if (x - t) % 2 == 1:
        t -= 1
    if (y - b) % 2 == 1:
        b -= 1

    roi = img[y:y + b, x:x + t]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # 将选中区域标记为 1
    mask[y:y + b, x:x + t] = 1
    return roi, mask

def paste_patch(img, patch):
    imgh, imgw, imgc = img.shape
    patchh, patchw, patchc = patch.shape
    #a = random.uniform(0, 1)
    patch_h_position = random.randrange(1, round(imgh) - round(patchh) - 1)
    patch_w_position = random.randrange(1, round(imgw) - round(patchw) - 1)
    pasteimg = np.copy(img)
    pasteimg[patch_h_position:patch_h_position + patchh,
    patch_w_position:patch_w_position + patchw, :] = patch #+ (1 - a) * img[patch_h_position:patch_h_position + patchh,
    #patch_w_position:patch_w_position + patchw, :]    #mvtec    patch

    return pasteimg

class Normalize(object):
    """
    Only normalize images
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image

def random_rotate(image, mask):
    """对图像进行随机旋转（90度、180度、-90度），但保持 mask 选取的中心部分不变"""
    angles = [1, 2, 3]  # np.rot90 旋转 90°、180°、-90°（对应 1, 2, 3 次）
    k = random.choice(angles)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    # 记录 mask 的中心区域
    center_region = image * mask  # 仅保留 patch 选取的中心部分

    # 旋转整张图像
    rotated_image = np.rot90(image, k=k, axes=(0, 1))  # 对 H, W 旋转
    rotated_mask = np.rot90(mask, k=k, axes=(0, 1))  # 旋转 mask

    # 恢复 mask 选中的中心区域
    rotated_image = rotated_image * (1 - rotated_mask) + center_region

    return rotated_image

def noise_generate(image_np, t):
    transform = transforms.Compose([
        Normalize(),
        ToTensor(),
    ])
    rotated_list = []
    for i in range(image_np.size(0)):
        np_img = image_np[i]
        #print(np_img.shape)
        # mask = binarize_image(np_img)
        # mask = mask.permute(1, 2, 0)
        # if np_img[0, 0, 0] > 0.5:  # 对于RGB图像 [H, W, 3]
        #     mask = 1 - mask
        # else:
        #     mask = mask

        patch_img, _ = cut(np_img, t, t)
        patch_img = paste_patch(np_img, patch_img)
        #image2 = random_rotate(np_img, mask)
        # patch_img = torch.tensor(patch_img)
        # patch_img = patch_img * mask + np_img * (1 - mask)
        # patch_img = patch_img.numpy()

        img_noise = transform(patch_img)
        img_noise = torch.unsqueeze(img_noise, dim=0)
        #img_noise2 = transform(image2)
        #img_noise2 = torch.unsqueeze(img_noise2, dim=0)

        # mask_tensor = torch.from_numpy(mask).float()
        # mask_tensor = torch.unsqueeze(mask_tensor, dim=0)
        # mask_tensor = torch.unsqueeze(mask_tensor, dim=1)

        rotated_list.append(img_noise)
        #rotated_list2.append(img_noise2)
        #mask_list.append(mask_tensor)
    img_noise = torch.cat(rotated_list, dim=0)
    #img_noise2 = torch.cat(rotated_list2, dim=0)
    #mask_ = torch.cat(mask_list, dim=0)
    return img_noise


def get_mask(data, feature):
    n, c, h, w = data.shape
    f0 = F.interpolate(feature[0], size=(h, w), mode='bilinear', align_corners=True)
    f1 = F.interpolate(feature[1], size=(h, w), mode='bilinear', align_corners=True)
    f2 = F.interpolate(feature[2], size=(h, w), mode='bilinear', align_corners=True)
    f = torch.cat([f0, f1, f2], dim=1)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    max = max_pool(f)
    avg = avg_pool(f)
    max_feature = F.pixel_shuffle(max, 2)
    avg_feature = F.pixel_shuffle(avg, 2)
    max_feature = torch.mean(max_feature, dim=1, keepdim=True)
    avg_feature = torch.mean(avg_feature, dim=1, keepdim=True)
    mask = (max_feature + avg_feature) / 2
    label = torch.max(max_feature)
    return mask, label

def otsu_thresholding(mask):
    n, _, h, w = mask.shape
    binary_images = torch.zeros_like(mask, dtype=torch.long)
    thresholds = []

    for i in range(n):
        single_map = mask[i, 0]
        hist, _ = torch.histogram(single_map, bins=256, range=(0, 1))
        hist = hist.float()
        total_pixels = torch.sum(hist)

        best_threshold = 0
        max_between_class_variance = 0

        for t in range(1, 256):
            w0 = torch.sum(hist[:t]) / total_pixels
            w1 = 1 - w0
            u0 = torch.sum(torch.arange(0, t, dtype=torch.float) * hist[:t]) / w0 / total_pixels if w0 > 0 else 0
            u1 = torch.sum(torch.arange(t, 256, dtype=torch.float) * hist[t:]) / w1 / total_pixels if w1 > 0 else 0
            between_class_variance = w0 * w1 * (u0 - u1) ** 2
            if between_class_variance > max_between_class_variance:
                max_between_class_variance = between_class_variance
                best_threshold = t / 255.0

        binary_images[i, 0] = torch.where(single_map > best_threshold, 1, 0)
        thresholds.append(best_threshold)

    return binary_images, torch.tensor(thresholds)

def generate_mask(shape):
    """
    此函数用于生成指定形状的掩码
    :param shape: 掩码的形状
    :return: 生成的掩码
    """
    batch_size, channels, height, width = shape
    mask = torch.zeros(shape)
    # 将最外的3行和3列置为1
    mask[:, :, :1, :] = 1
    mask[:, :, -1:, :] = 1
    mask[:, :, :, :1] = 1
    mask[:, :, :, -1:] = 1
    return mask

def binarize_image(image):
    # 将图像转换为灰度图
    if isinstance(image, torch.Tensor):
        # 转换为numpy数组并确保维度正确
        image_np = image.cpu().numpy()
        # 检查是否需要调整维度（如果是CHW格式则转换为HWC）
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
            image_np = np.transpose(image_np, (1, 2, 0))
    else:
        image_np = np.array(image)

    # 确保数据在[0,255]范围内并转为uint8
    if image_np.dtype in [np.float32, np.float64]:
        # 检查数据范围
        if image_np.max() <= 1.0 and image_np.min() >= 0.0:
            # 范围在[0,1]的浮点数，乘以255
            image_np = (image_np * 255).astype(np.uint8)
        else:
            # 范围不在[0,1]的浮点数，先归一化再乘以255
            image_np = ((image_np - image_np.min()) /
                        (image_np.max() - image_np.min()) * 255).astype(np.uint8)

    # 转换为灰度图
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_np
    # 进行二值化操作
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    binary_image_tensor = torch.from_numpy(binary_image).unsqueeze(0).float() / 255.0
    return binary_image_tensor

def weighted_decision_mechanism(num, output_list, alpha, beta, out_size=256):
    """
    num: The number of test samples
    output_list: List consists of features generated by minimizing the outputs of different layers of models
    alpha and beta: Hyperparameters for controlling upper and lower limit
    return: anomaly score for anomaly detection, anomaly map for anomaly segmentation
    """

    total_weights_list = list()
    for i in range(num):
        low_similarity_list = list()
        for j in range(len(output_list)):
            low_similarity_list.append(torch.max(output_list[j][i]).cpu())
        probs = F.softmax(torch.tensor(low_similarity_list), 0)
        weight_list = list()  # set P consists of L high probability values, where L ranges from n-1 to n+1
        for idx, prob in enumerate(probs):
            weight_list.append(low_similarity_list[idx].numpy()) if prob > torch.mean(probs) else None
        weight = np.max([np.mean(weight_list) * alpha, beta])
        total_weights_list.append(weight)

    assert len(total_weights_list) == num, "the number of weights dose not match that of samples!"

    am_lists = [list() for _ in output_list]
    for l, output in enumerate(output_list):
        output = torch.cat(output, dim=0)
        a_map = torch.unsqueeze(output, dim=1)  # B*1*h*w
        am_lists[l] = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)[:, 0, :, :]  # B*256*256

    anomaly_map = sum(am_lists)
    # anomaly_map_ = anomaly_map - anomaly_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]  # B*256*256
    # anomaly_maps_exp = torch.exp(anomaly_map_)
    # anomaly_score_exp = anomaly_maps_exp.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - anomaly_maps_exp

    anomaly_score_exp = anomaly_map
    batch = anomaly_score_exp.shape[0]
    anomaly_score = list()  # anomaly scores for all test samples
    for b in range(batch):
        top_k = int(out_size * out_size * total_weights_list[b])
        assert top_k >= 1 / (out_size * out_size), "weight can not be smaller than 1 / (H * W)!"

        single_anomaly_score_exp = anomaly_score_exp[b]
        single_anomaly_score_exp = torch.tensor(gaussian_filter(
            single_anomaly_score_exp.detach().cpu().numpy(), sigma=4)
        )
        assert single_anomaly_score_exp.reshape(1, -1).shape[-1] == out_size * out_size, \
            "something wrong with the last dimension of reshaped map!"

        single_map = single_anomaly_score_exp.reshape(1, -1)
        single_anomaly_score = np.mean(single_map.topk(top_k, dim=-1)[0].detach().cpu().numpy(), axis=1)
        anomaly_score.append(single_anomaly_score)

    return anomaly_score


def Uninet_losses(b, a, T, margin, λ=0.7, mask=None, stop_gradient=False):
    """
    b: List of teacher features
    a: List of student features
    mask: Binary mask, where 0 for normal and 1 for abnormal
    T: Temperature coefficient
    margin: Hyperparameter for controlling the boundary
    λ: Hyperparameter for balancing loss
    """

    loss = 0.0
    margin_loss_n = 0.0
    margin_loss_a = 0.0
    contra_loss = 0.0
    for i in range(len(a)):
        s_ = a[i]
        t_ = b[i].detach() if stop_gradient else b[i]

        n, c, h, w = s_.shape

        s = s_.view(n, c, -1).transpose(1, 2)  # (N, H*W, C)
        t = t_.view(n, c, -1).transpose(1, 2)  # (N, H*W, C)

        s_norm = F.normalize(s, p=2, dim=2)
        t_norm = F.normalize(t, p=2, dim=2)

        cos_loss = 1 - F.cosine_similarity(s_norm, t_norm, dim=2)
        cos_loss = cos_loss.mean()

        simi = torch.matmul(s_norm, t_norm.transpose(1, 2)) / T
        simi = torch.exp(simi)
        simi_sum = simi.sum(dim=2, keepdim=True)
        simi = simi / (simi_sum + 1e-8)
        diag_sim = torch.diagonal(simi, dim1=1, dim2=2)

        # unsupervised and only normal (or abnormal)
        if mask is None:
            contra_loss = -torch.log(diag_sim + 1e-8).mean()
            margin_loss_n = F.relu(margin - diag_sim).mean()

        margin_loss = margin_loss_n + margin_loss_a

        loss += cos_loss * λ + contra_loss * (1 - λ) + margin_loss

    return loss

class EarlyStop_V1:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, encoder, bn, decoder, proj, log, epoch):
        """
        参数:
            score: 当前 epoch 的评估得分（越小越好）
            encoder, bn, decoder, proj: 当前模型参数
            log: 日志文件句柄
            epoch: 当前 epoch
        返回值:
            是否应该停止训练
        """
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print_log(f'Epoch {epoch}: +++, Best Score ({self.best_score})', log)
        else:
            self.counter += 1
            print_log(f'Epoch {epoch}: --- ({self.counter}/{self.patience})', log)
            if self.counter >= self.patience:
                print_log('Early stopping', log)
                self.early_stop = True
                return True
        return False


def multi_scale_cos_margin_loss(normal_student, abnormal_student, label, output_noise_, margin_pos=1.0, margin_neg=0,
                                beta=1.0):
    loss_pos_total = 0
    loss_neg_total = 0
    n_scales = len(normal_student)

    for i in range(n_scales):
        ns = normal_student[i]  # (B, C, H, W)
        ab = abnormal_student[i]
        lb = label[i]
        on = output_noise_[i]

        # reshape to (B, C, H*W)
        B, C, H, W = ns.shape
        ns = ns.view(B, C, -1)  # (B, C, N)
        lb = lb.view(B, C, -1)
        ab = ab.view(B, C, -1)
        on = on.view(B, C, -1)

        # normalize features (L2)
        ns = F.normalize(ns, dim=1)
        lb = F.normalize(lb, dim=1)
        ab = F.normalize(ab, dim=1)
        on = F.normalize(on, dim=1)

        # cosine similarity (B, N)
        sim_pos = (ns * lb).sum(dim=1)  # shape: (B, N)
        sim_neg = (ab * on).sum(dim=1)

        # mean over all spatial positions
        sim_pos_mean = sim_pos.mean(dim=1)  # (B,)
        sim_neg_mean = sim_neg.mean(dim=1)

        # margin loss
        loss_pos = (margin_pos - sim_pos_mean).mean()
        loss_neg = (sim_neg_mean - margin_neg).mean()

        loss_pos_total += loss_pos
        loss_neg_total += loss_neg

    total_loss = loss_pos_total / n_scales + beta * (loss_neg_total / n_scales)
    return total_loss
