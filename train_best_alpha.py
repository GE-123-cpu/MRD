import argparse
import time
from tqdm import tqdm
from mvtec import *
from fun import *
from utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
import numpy as np
import random
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score
import math


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main(object):
    parser = argparse.ArgumentParser(description='anomaly detection')
    parser.add_argument('--obj', type=str, default=object)
    parser.add_argument('--data_type', type=str, default='alpha')
    parser.add_argument('--data_path', type=str, default='E:\mvtec_anomaly_detection')
    parser.add_argument('--epochs', type=int, default=5, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--proj_lr', default=0.005, type=float)
    parser.add_argument('--encoder_lr', default=0.0001, type=float)
    parser.add_argument('--distill_lr', default=0.005, type=float)
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    args = parser.parse_args()

    args.prefix = time_file_str()
    args.save_dir = './' + args.data_type + '/' + args.obj

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'model_training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    expert, _ = wide_resnet50_2(pretrained=True)
    expert = expert.to(device)
    expert.eval()

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size)

    val_ratio = 0.2
    train_size = int((1 - val_ratio) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # 固定随机种子，保证每次划分一致
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
    )

    # DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        **kwargs
    )

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)

    optimizer_encoder = torch.optim.Adam(list(encoder.parameters()), lr=args.encoder_lr, betas=(0.5, 0.999))

        # 初始化模型和优化器
    start_time = time.time()
    epoch_time = AverageMeter()
    best_alpha_records = []
    for epoch in range(1, args.epochs + 1):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(f' {epoch:3d}/{args.epochs:3d} ----- [{time_string()}] {need_time}', log)
        train(expert, epoch, train_loader, optimizer_encoder, log, encoder)
        best_alpha = test_distance(val_loader, encoder, expert, epoch, log)
        best_alpha_records.append(best_alpha)  # 保存
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    mean_alpha = np.mean(best_alpha_records)

    final_best_alpha = math.ceil(mean_alpha * 10) / 10

    final_best_alpha = round(final_best_alpha, 1)

    print_log(
        f"Mean alpha = {mean_alpha:.6f}, Final alpha = {final_best_alpha:.1f}, obj = {args.obj}",
        log
    )
    log.close()


def train(expert, epoch, train_loader, optimizer_encoder, log, encoder):
    N1_losses = AverageMeter()
    losses = AverageMeter()
    l2 = loss_aug()
    l_single = loss_fucntion()
    for (data, img_np, _, _) in tqdm(train_loader):
        encoder.train()
        n, c, h, w = data.shape

        large_value = random.randint(150, 200)
        normal_value = random.randint(64, 100)
        small_value = random.randint(10, 32)

        weights = [0.2, 0.3, 0.5]
        values = [large_value, normal_value, small_value]
        t = random.choices(values, weights, k=1)[0]
        img_noise = noise_generate(img_np, t)

        data = data.to(device)
        noise = img_noise.to(device)
        mask = torch.ones(n, h, w).to(device) - F.cosine_similarity(noise, data)
        mask = mask.unsqueeze(dim=1)
        mask[mask > 0] = 1
        mask[mask <= 0] = 0

        with torch.no_grad():
            output = expert(data)

        normal = encoder(data)
        abnormal = encoder(noise)

        N1loss = l_single(normal[0], output[0]) + l_single(normal[1], output[1]) +  l_single(normal[2], output[2])
        N2loss = l2(abnormal, output, mask, device, n)

        loss = 0.001 * (N1loss + N2loss)

        N1_losses.update(N1loss.item(), data.size(0))
        losses.update(loss.item(), data.size(0))

        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()


    print_log(('Train Epoch: {} Loss: {:.6f} '.format(epoch, losses.avg)), log)


def get_anomap(output, Dn):
    anomaly_map1_kd = torch.ones(1, 64, 64).to(device) - F.cosine_similarity(output[0], Dn[0])
    anomaly_map1_kd = anomaly_map1_kd.unsqueeze(1)
    anomaly_map2_kd = torch.ones(1, 32, 32).to(device) - F.cosine_similarity(output[1], Dn[1])
    anomaly_map2_kd = anomaly_map2_kd.unsqueeze(1)
    anomaly_map3_kd = torch.ones(1, 16, 16).to(device) - F.cosine_similarity(output[2], Dn[2])
    anomaly_map3_kd = anomaly_map3_kd.unsqueeze(1)
    return [anomaly_map1_kd, anomaly_map2_kd, anomaly_map3_kd]



def test_distance(val_loader, encoder, expert, epoch, log):
    encoder.eval()
    expert.eval()

    alpha_list = [i / 10 for i in range(10)]

    alpha_preds = {alpha: [] for alpha in alpha_list}
    alpha_targets = {alpha: [] for alpha in alpha_list}

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    with torch.no_grad():
        for data, img_np, _, _ in tqdm(val_loader):
            n, c, h, w = data.shape

            data = data.to(device)

            # 生成伪异常
            large_value = random.randint(150, 200)
            normal_value = random.randint(64, 100)
            small_value = random.randint(10, 32)


            img_noise = noise_generate(img_np, small_value)
            noise = img_noise.to(device)

            # synthetic mask
            mask = 1 - F.cosine_similarity(noise, data, dim=1)
            mask = mask.unsqueeze(1)
            mask = (mask > 0).float()

            # feature extraction
            expert_normal = expert(data)
            encoder_normal = encoder(data)
            encoder_noise = encoder(noise)
            expert_noise = expert(noise)

            for alpha in alpha_list:
                # 只融合 normal reference feature
                label = [
                    alpha * e + (1 - alpha) * n
                    for e, n in zip(expert_normal, encoder_normal)
                ]
                ab = [
                    alpha * e + (1 - alpha) * n
                    for e, n in zip(expert_noise, encoder_noise)
                ]


                # abnormal branch 只使用 encoder(noise)
                am = get_anomap(ab, label)

                am0 = F.interpolate(am[0], size=(h, w), mode='bilinear', align_corners=True)
                am1 = F.interpolate(am[1], size=(h, w), mode='bilinear', align_corners=True)
                am2 = F.interpolate(am[2], size=(h, w), mode='bilinear', align_corners=True)

                anomaly_map = (am0 + am1 + am2) / 3

                pred = anomaly_map.reshape(-1).cpu().numpy()
                target = mask.reshape(-1).cpu().numpy()

                alpha_preds[alpha].extend(pred)
                alpha_targets[alpha].extend(target)

    alpha_score_dict = {}

    for alpha in alpha_list:
        pred = np.array(alpha_preds[alpha])
        target = np.array(alpha_targets[alpha])

        try:
            pixel_auc = roc_auc_score(target, pred)
        except ValueError:
            pixel_auc = 0.0

        try:
            pixel_ap = average_precision_score(target, pred)
        except ValueError:
            pixel_ap = 0.0

        score = 0.5 * pixel_auc + 0.5 * pixel_ap
        alpha_score_dict[alpha] = score

        print_log(
            f'Alpha {alpha:.1f} | Pixel AUROC: {pixel_auc:.6f} | Pixel AP: {pixel_ap:.6f} | Score: {score:.6f}',
            log
        )

    best_alpha = max(alpha_score_dict, key=alpha_score_dict.get)
    best_score = alpha_score_dict[best_alpha]

    print_log(
        f'[Epoch {epoch}] Selected Alpha: {best_alpha:.1f} | Best Score: {best_score:.6f}\n',
        log
    )

    return best_alpha

if __name__ == '__main__':
    item_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper']
    for i in item_list:
        main(i)