from PIL import Image
# import urllib.request
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import os
# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]

class Normalize(object):
    """
    Only normalize images
    """
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
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


class MVTecDataset(Dataset):
    def __init__(self,
                 dataset_path='../data/mvtec_anomaly_detection',
                 class_name='bottle',
                 is_train=True,
                 resize=256,  #256
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = transforms.Compose([
            Normalize(),
            ToTensor()
            # transforms.Resize(resize, Image.ANTIALIAS),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose(
            [transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
             transforms.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        # x = Image.open(x).convert('RGB')
        # x = self.transform_x(x)
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (256, 256))
        ## Normal
        img_normal = self.transform_x(img)

        if y == 0:
            mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return img_normal, img, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)




# import matplotlib
# matplotlib.use('TkAgg')  # 或 'Agg' (无界面环境)、'Qt5Agg' 等
# import matplotlib.pyplot as plt
# import cv2
# import os
#
#
# def compute_train_mean_and_mse(dataset):
#     """
#     计算训练集所有图像的均值和每个图像与均值的MSE
#
#     参数:
#     dataset: MVTecDataset实例（必须是训练集）
#
#     返回:
#     mean_image: 均值图像 (H, W, C)
#     mse_values: 每个图像的MSE值列表
#     """
#     if not dataset.is_train:
#         raise ValueError("仅支持训练集计算均值和MSE")
#
#     print("开始计算训练集均值和MSE...")
#     images = []
#
#     # 加载所有训练图像
#     for img_path in dataset.x:
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img/255., (dataset.resize, dataset.resize))
#         images.append(img)
#
#     # 转换为NumPy数组
#     images_np = np.array(images)
#
#     # 计算均值图像
#     mean_image = np.mean(images_np, axis=0)
#
#     # 计算每个图像与均值的MSE
#     mse_values = []
#     h, w, c = mean_image.shape
#     pixel_count = h * w * c  # 总像素数（包含通道）
#
#     for img in images_np:
#         # 计算MSE：平方误差总和 / 总像素数
#         mse = np.sum((img - mean_image) ** 2) * 255 * 255 / pixel_count
#         mse_values.append(mse)
#
#     print(f"计算完成: 共{len(images)}张图像，平均MSE = {np.mean(mse_values):.6f}")
#     return mean_image, mse_values
#
#
# def plot_mse_results(mse_values, mean_image, class_name, save_dir='./results'):
#     """
#     可视化MSE结果：绘制均值图像和MSE分布
#
#     参数:
#     mse_values: MSE值列表
#     mean_image: 均值图像
#     class_name: 类别名称
#     save_dir: 结果保存目录
#     """
#     # 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 1. 绘制均值图像
#     plt.figure(figsize=(8, 8))
#     plt.imshow(mean_image)
#     plt.title(f'Mean Image - {class_name} Training Set')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f'{class_name}_mean_image.png'), dpi=300)
#     plt.close()
#
#     # 2. 绘制MSE分布
#     plt.figure(figsize=(12, 5))
#
#     # MSE值折线图
#     plt.subplot(1, 2, 1)
#     plt.plot(range(len(mse_values)), mse_values, 'o-', color='royalblue', alpha=0.6)
#     plt.axhline(np.mean(mse_values), color='red', linestyle='--', label=f'Avg: {np.mean(mse_values):.6f}')
#     plt.title(f'MSE per Image - {class_name}')
#     plt.xlabel('Image Index')
#     plt.ylabel('MSE')
#     plt.legend()
#     plt.grid(alpha=0.3)
#
#     # MSE分布直方图
#     plt.subplot(1, 2, 2)
#     plt.hist(mse_values, bins=20, color='lightgreen', alpha=0.7)
#     plt.axvline(np.mean(mse_values), color='red', linestyle='--', label=f'Avg: {np.mean(mse_values):.6f}')
#     plt.title(f'MSE Distribution - {class_name}')
#     plt.xlabel('MSE Value')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.grid(alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f'{class_name}_mse_analysis.png'), dpi=300)
#     plt.close()
#
#     # 输出统计信息
#     stats = {
#         'mean_mse': np.mean(mse_values),
#         'std_mse': np.std(mse_values),
#         'max_mse': np.max(mse_values),
#         'min_mse': np.min(mse_values),
#         'max_idx': np.argmax(mse_values),
#         'min_idx': np.argmin(mse_values)
#     }
#
#     print("\nMSE统计信息:")
#     print(f"平均值: {stats['mean_mse']:.6f}")
#     print(f"标准差: {stats['std_mse']:.6f}")
#     print(f"最大值: {stats['max_mse']:.6f} (图像索引: {stats['max_idx']})")
#     print(f"最小值: {stats['min_mse']:.6f} (图像索引: {stats['min_idx']})")
#
#     return stats
#
#
# if __name__ == "__main__":
#    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#                'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#                'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
#
# # 1. 创建训练集
#    class_name = 'screw'
#    train_dataset = MVTecDataset(
#     dataset_path='E:\\URD-main\\URD-main\\data\\mvtec_anomaly_detection',
#     class_name=class_name,
#     is_train=True,
#     resize=256
#    )
#
#    # 2. 计算均值和MSE
#    mean_img, mse_list = compute_train_mean_and_mse(train_dataset)
#
#   # 3. 可视化结果
#    stats = plot_mse_results(mse_list, mean_img, class_name)

