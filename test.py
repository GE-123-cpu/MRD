import argparse
import matplotlib
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage import morphology, measure
from scipy.ndimage import gaussian_filter
from fun import denormalization1, denormalization
from mvtec import *
from fun import *
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from model import MultiProjectionLayer


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
plt.switch_backend('agg')


def main(object, path, seed):
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--obj', type=str, default=object)
    parser.add_argument('--data_type', type=str, default='Mvtec')
    parser.add_argument('--data_path', type=str, default=r'E:\mvtec_anomaly_detection')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default=path)
    parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--alpha', type=float, default=0.3, help='sensitivity')

    args = parser.parse_args()
    args.save_dir = './' + args.data_type + '/' + args.obj + '/seed_{}_model/'.format(args.seed) + '/AM'
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    epoch_threshold = 10 if args.obj == 'screw' else 100
    args.input_channel = 1 if args.grayscale else 3

    expert, _ = wide_resnet50_2(pretrained=True)
    expert = expert.to(device)
    expert.eval()

    encoder, bn = wide_resnet50_2(pretrained=False)
    encoder = encoder.to(device)
    bn = bn.to(device)

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    proj = MultiProjectionLayer(base=64).to(device)

    checkpoint = torch.load(args.checkpoint_dir, weights_only=False)

    encoder.load_state_dict(checkpoint['encoder'])
    bn.load_state_dict(checkpoint['bn'])
    decoder.load_state_dict(checkpoint['decoder'])
    proj.load_state_dict(checkpoint['proj'])
    best_epoch = checkpoint['best_epoch']
    print(best_epoch)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    scores, test_imgs, recon_imgs, gt_list, gt_mask_list, i_scores, aupro = test_(bn, decoder, expert, test_loader,
                                                                                  encoder, proj, args.alpha,
                                                                                  best_epoch - 1, epoch_threshold)

    scores = np.asarray(scores)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    img_scores = np.asarray(i_scores)
    gt_list = np.asarray(gt_list)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print(img_roc_auc)

    img_ap = average_precision_score(gt_list, img_scores)
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    pix_ap = average_precision_score(gt_mask.flatten(), scores.flatten())
    ap_rocauc = np.mean(aupro)
    plot_fig_all(args, test_imgs, recon_imgs, scores, gt_mask_list, threshold, args.save_dir)
    print('image ROCAUC: %.3f' % (img_roc_auc * 100), 'image AP: %.3f' % (img_ap * 100), 'Pixel ROCAUC: %.3f' % (per_pixel_rocauc * 100),
          'Pixel AP: %.3f' % (pix_ap * 100), 'PRO: %.3f' % (ap_rocauc * 100))



def plot_fig_all(args, test_img, recon_imgs, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    scores = np.squeeze(scores, axis=1)
    gts = np.squeeze(gts, axis=1)
    #gts = np.squeeze(gts, axis=1)
    for i in range(num):
        img = test_img[i]
        # img = denorm1(img)
        img = denormalization(img)
        recon_img = recon_imgs[i]
        recon_img = denormalization(recon_img)
        gt = gts[i].squeeze()
        #gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('Predicted mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }

        fig_img.savefig(os.path.join(save_dir, '_{}_png'.format(i)), dpi=100)
        plt.close()


def get_anomap(output, Dn):
    anomaly_map1_kd = torch.ones(1, 64, 64).to(device) - F.cosine_similarity(output[0], Dn[0])
    anomaly_map1_kd = anomaly_map1_kd.unsqueeze(1)
    anomaly_map2_kd = torch.ones(1, 32, 32).to(device) - F.cosine_similarity(output[1], Dn[1])
    anomaly_map2_kd = anomaly_map2_kd.unsqueeze(1)
    anomaly_map3_kd = torch.ones(1, 16, 16).to(device) - F.cosine_similarity(output[2], Dn[2])
    anomaly_map3_kd = anomaly_map3_kd.unsqueeze(1)
    return [anomaly_map1_kd, anomaly_map2_kd, anomaly_map3_kd]

def test_(bn, decoder, expert, test_loader, encoder, proj, alpha, epoch, epoch_threshold):
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
    aupro_list = []
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
        if label.item() != 0:
            aupro_list.append(compute_pro(mask.squeeze(0).cpu().numpy().astype(int), pixel_score))

    return scores, test_imgs, recon_imgs, gt_list, gt_mask_list, image_scores, aupro_list


if __name__ == '__main__':
    main('zipper', './XXX.pth', '3313')

