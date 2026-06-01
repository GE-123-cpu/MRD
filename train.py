import argparse
import time
from tqdm import tqdm
from mvtec import *
from fun import *
from utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.ndimage import gaussian_filter
import random
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from model import MultiProjectionLayer


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main(object, alpha):
    parser = argparse.ArgumentParser(description='anomaly detection')
    parser.add_argument('--obj', type=str, default=object)
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='E:\mvtec_anomaly_detection')
    parser.add_argument('--epochs', type=int, default=200, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--proj_lr', default=0.005, type=float)
    parser.add_argument('--encoder_lr', default=0.00001, type=float)
    parser.add_argument('--distill_lr', default=0.005, type=float)
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    #parser.add_argument('--alpha', type=float, default=0.8, help='sensitivity')
    args = parser.parse_args()
    if args.obj == 'screw':
        epoch_threshold = 10
    else:
        epoch_threshold = 100

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.prefix = time_file_str()
    args.save_dir = './' + args.data_type + '/' + args.obj + '/seed_{}_model_/'.format(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'model_training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}  # {args.obj}
    print_log(state, log)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    expert, _ = wide_resnet50_2(pretrained=True)
    expert = expert.to(device)
    expert.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    proj = MultiProjectionLayer(base=64).to(device)

    optimizer_encoder = torch.optim.Adam(list(encoder.parameters()), lr=args.encoder_lr, betas=(0.5, 0.999))
    optimizer_proj = torch.optim.Adam(list(proj.parameters()), lr=args.proj_lr, betas=(0.5, 0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=args.distill_lr,
                                         betas=(0.5, 0.999))


    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # start training
    save_name = os.path.join(args.save_dir, '{}_{}_model.pth'.format(args.obj, args.prefix))
    early_stop = EarlyStop(patience=30, save_name=save_name)
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(1, args.epochs + 1):
        # adjust_learning_rate(args, optimizer, epoch)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)
        train1(expert, epoch, train_loader, optimizer_encoder, optimizer_proj, optimizer_distill, log, encoder, bn, decoder, proj, alpha, epoch_threshold)

        if epoch > 0:
           scores, test_imgs, recon_imgs, gt_list, gt_mask_list, i_scores = test_(bn, decoder, expert, test_loader, encoder, proj, epoch, alpha, epoch_threshold)
           scores = np.asarray(scores)
           max_anomaly_score = scores.max()
           min_anomaly_score = scores.min()
           scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

           img_scores = i_scores
           gt_list = np.asarray(gt_list)
           fpr, tpr, _ = roc_curve(gt_list, img_scores)
           img_roc_auc = roc_auc_score(gt_list, img_scores)
           gt_mask = np.asarray(gt_mask_list)
           per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
           best_score = - (per_pixel_rocauc + img_roc_auc) * 100
           print_log(('epoch: {} image: {:.3f} pixel: {:.3f}'.format(epoch, img_roc_auc * 100, per_pixel_rocauc * 100)), log)

           if (early_stop(best_score, encoder, bn, decoder, proj, log, epoch)):
              break


        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()

def train1(expert, epoch, train_loader, optimizer_encoder, optimizer_proj, optimizer_distill, log, encoder, bn, decoder, proj, alpha, epoch_threshold):
    N1_losses = AverageMeter()
    N2_losses = AverageMeter()
    N3_losses = AverageMeter()
    N4_losses = AverageMeter()
    N5_losses = AverageMeter()
    N6_losses = AverageMeter()


    losses = AverageMeter()
    l_rec = loss_normal()
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

        with torch.no_grad():
            output = expert(data)

        normal = encoder(data)
        abnormal = encoder(noise)
        N1loss = l_rec(normal, output) + l_single(normal[0], output[0]) + l_single(normal[1], output[1]) +  l_single(normal[2], output[2])
        N2loss = l2(abnormal, output, mask, device, n)

        loss = 0.01 * (N1loss + N2loss)

        N1_losses.update(N1loss.item(), data.size(0))
        N2_losses.update(N2loss.item(), data.size(0))
        losses.update(loss.item(), data.size(0))

        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()

        bn.train()
        decoder.train()
        proj.train()
        encoder.eval()

        t = random.choices(values, weights, k=1)[0]
        img_noise_m_ = noise_generate(img_np, t)

        noise_ = img_noise_m_.to(device)

        with torch.no_grad():
            true_out = expert(data)
            output_ = encoder(data)
            output_noise_ = encoder(noise_)


        label = [alpha * o + (1 - alpha) * t for o, t in zip(true_out, output_)]
        proj_feature = proj(output_noise_)

        normal_student = decoder(bn(output_noise_), output_noise_, epoch, epoch_threshold)
        abnormal_student = decoder(bn(proj_feature), label, epoch, epoch_threshold)

        N4_loss = l_rec(normal_student, label)+ l_single(normal_student[0], label[0]) + \
                  l_single(normal_student[1], label[1]) + l_single(normal_student[2], label[2])

        N5_loss = l_rec(abnormal_student, label) + l_single(abnormal_student[0], label[0]) + \
                  l_single(abnormal_student[1], label[1]) + l_single(abnormal_student[2], label[2])

        N6_loss = l_rec(proj_feature, label) + l_single(proj_feature[0], label[0]) + \
                  l_single(proj_feature[1], label[1]) + l_single(proj_feature[2], label[2])

        loss2 = N4_loss + N5_loss + N6_loss

        N3_losses.update(loss2.item(), data.size(0))
        N4_losses.update(N4_loss.item(), data.size(0))
        N5_losses.update(N5_loss.item(), data.size(0))
        N6_losses.update(N6_loss.item(), data.size(0))

        optimizer_proj.zero_grad()
        optimizer_distill.zero_grad()
        loss2.backward()
        optimizer_proj.step()
        optimizer_distill.step()

    print_log(('Train Epoch: {} N3_Loss: {:.6f}, N4_Loss: {:.6f} N5_Loss: {:.6f}, N6_Loss: {:.6f}, Loss: {:.6f} '.format(epoch,
            N3_losses.avg, N4_losses.avg, N5_losses.avg, N6_losses.avg, losses.avg)), log)


def get_anomap_(output, Dn, b):
    anomaly_map1_kd = torch.ones(b, 64, 64).to(device) - F.cosine_similarity(output[0], Dn[0])
    anomaly_map1_kd = anomaly_map1_kd.unsqueeze(1)
    anomaly_map2_kd = torch.ones(b, 32, 32).to(device) - F.cosine_similarity(output[1], Dn[1])
    anomaly_map2_kd = anomaly_map2_kd.unsqueeze(1)
    anomaly_map3_kd = torch.ones(b, 16, 16).to(device) - F.cosine_similarity(output[2], Dn[2])
    anomaly_map3_kd = anomaly_map3_kd.unsqueeze(1)
    return [anomaly_map1_kd, anomaly_map2_kd, anomaly_map3_kd]


def get_anomap(output, Dn):
    anomaly_map1_kd = torch.ones(1, 64, 64).to(device) - F.cosine_similarity(output[0], Dn[0])
    anomaly_map1_kd = anomaly_map1_kd.unsqueeze(1)
    anomaly_map2_kd = torch.ones(1, 32, 32).to(device) - F.cosine_similarity(output[1], Dn[1])
    anomaly_map2_kd = anomaly_map2_kd.unsqueeze(1)
    anomaly_map3_kd = torch.ones(1, 16, 16).to(device) - F.cosine_similarity(output[2], Dn[2])
    anomaly_map3_kd = anomaly_map3_kd.unsqueeze(1)
    return [anomaly_map1_kd, anomaly_map2_kd, anomaly_map3_kd]

def test_(bn, decoder, expert, test_loader, encoder, proj, epoch, alpha, epoch_threshold):
    encoder.eval()
    bn.eval()
    proj.eval()
    decoder.eval()
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    image_scores = []
    for (data, _, label, mask) in tqdm(test_loader):
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        test_imgs.extend(data.cpu().numpy())
        gt_list.extend(label.cpu().numpy().astype(int))
        gt_mask_list.extend(mask.cpu().numpy())
        with torch.no_grad():
            n, c, h, w = data.shape
            data = data.to(device)
            a = expert(data)
            output = encoder(data)

            true_label = [alpha * o + (1 - alpha) * t for o, t in zip(a, output)]
            low_sen_label = a

            out_proj = proj(true_label)
            out = decoder(bn(out_proj), true_label, epoch, epoch_threshold)

            out_proj_low_sen = proj(low_sen_label)
            out_low_sen = decoder(bn(out_proj_low_sen), low_sen_label, epoch, epoch_threshold)

            #
            am = get_anomap(true_label, out)
            bm = get_anomap(low_sen_label, out_low_sen)
            #
            am0 = F.interpolate(am[0], size=(h, w), mode='bilinear', align_corners=True)
            am1 = F.interpolate(am[1], size=(h, w), mode='bilinear', align_corners=True)
            am2 = F.interpolate(am[2], size=(h, w), mode='bilinear', align_corners=True)
            anomaly_map = (am0 + am1 + am2) / 3
            #
            bm0 = F.interpolate(bm[0], size=(h, w), mode='bilinear', align_corners=True)
            bm1 = F.interpolate(bm[1], size=(h, w), mode='bilinear', align_corners=True)
            bm2 = F.interpolate(bm[2], size=(h, w), mode='bilinear', align_corners=True)
            anomaly_map_score = (bm0 + bm1 + bm2) / 3

            am_proj = get_anomap(true_label, out_proj)
            amp_proj = sum(F.interpolate(item, size=(h, w), mode='bilinear', align_corners=True) for item in am_proj) / 3

            if epoch % 2 != 0:
              pixel_score = anomaly_map_score * (1 + amp_proj)
            else:
              pixel_score = anomaly_map * (1 + amp_proj)

        score1 = anomaly_map_score.squeeze(0).cpu().numpy()
        score1 = gaussian_filter(score1, sigma=4)
        score2 = anomaly_map.squeeze(0).cpu().numpy()
        score2 = gaussian_filter(score2, sigma=4)
        score_1 = np.max(score1)
        score_2 = np.max(score2)
        s = max(score_1, score_2)

        pixel_score = pixel_score.squeeze(0).cpu().numpy()
        pixel_score = gaussian_filter(pixel_score, sigma=4)
        scores.append(pixel_score)
        recon_imgs.extend(data.cpu().numpy())
        image_scores.append(s)
    return scores, test_imgs, recon_imgs, gt_list, gt_mask_list, image_scores



if __name__ == '__main__':
    item_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper']

    threshold = {
        "transistor": {"threshold": 0.2},
        "cable": {"threshold": 0.2},
        "zipper": {"threshold": 0.6},
        "pill": {"threshold": 0.2},
        "bottle": {"threshold": 0.4},
        "leather": {"threshold": 0.2},
        "toothbrush": {"threshold": 0.7},
        "screw": {"threshold": 0.2},
        "hazelnut": {"threshold": 0.1},
        "grid": {"threshold": 0.1},
        "tile": {"threshold": 0.1},
        "wood": {"threshold": 0.1},
        "carpet": {"threshold": 0.2},
        "metal_nut": {"threshold": 0.6},
        "capsule": {"threshold": 0.4}
    }


    def get_params(obj):
        """根据输入的类别名返回对应的阈值和seed"""
        return threshold.get(obj, {"threshold": None})

    for i in item_list:
        params = get_params(i)
        alpha = params["threshold"]
        main(i, alpha)