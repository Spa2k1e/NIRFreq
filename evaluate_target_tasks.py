import argparse
import os
import cv2
import torch
import numpy as np
from scipy import io
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models.nir_freq_model import NIRFreqModel
from src.models.ssamrn_model import SSAMRNModel
from src.models.lgteun_model import LGTEUNModel
from src.utils.common import check_and_make

DEFAULT_CKPT_PATHS = {
    'nirfreq': r"log_m=NIRFreqModel_s=GF_l=9_d=64_GCFM=True_HFA=True_CorrMap=True/nirfreq_ep=329_PSNR=42.4740.ckpt",
    'lgteun': r"log_m=LGTEUN_s=GF/lgteun_ep=269_PSNR=40.9971.ckpt",
    'ssamrn': r"log_m=SSAMRN_s=GF/ssamrn_ep=319_PSNR=41.0020.ckpt"
}


def get_model_class(model_name):
    model_classes = {'nirfreq': NIRFreqModel, 'ssamrn': SSAMRNModel, 'lgteun': LGTEUNModel}
    return model_classes[model_name]


def load_sample_data(data_dir, sample_filename, sensor):
    ms_path = os.path.join(data_dir, "MS_256", sample_filename)
    pan_path = os.path.join(data_dir, "PAN_1024", sample_filename)
    if not os.path.exists(ms_path) or not os.path.exists(pan_path):
        raise FileNotFoundError(f"未找到数据文件: {sample_filename}")

    max_val = 1023.0 if sensor.lower() in ['gf', 'gf1'] else 2047.0
    mat_ms = io.loadmat(ms_path)
    mat_pan = io.loadmat(pan_path)

    key_ms = "imgMS" if "imgMS" in mat_ms else "I_MS"
    key_pan = "imgPAN" if "imgPAN" in mat_pan else "I_PAN"
    if key_pan not in mat_pan: key_pan = "block"

    ms_hr = torch.from_numpy((mat_ms[key_ms] / max_val).astype(np.float32)).permute(2, 0, 1)
    pan_hr = torch.from_numpy((mat_pan[key_pan] / max_val).astype(np.float32))
    if pan_hr.ndim == 2: pan_hr = pan_hr.unsqueeze(0)

    gt = ms_hr.clone()
    from kornia.filters import gaussian_blur2d
    ms_lr = gaussian_blur2d(gt.unsqueeze(0), (3, 3), (1.5, 1.5))
    ms_lr = F.interpolate(ms_lr, scale_factor=0.25, mode="bicubic", align_corners=False).squeeze(0)
    pan_lr = F.interpolate(pan_hr.unsqueeze(0), scale_factor=0.25, mode="bicubic", align_corners=False).squeeze(0)
    up_ms = F.interpolate(ms_lr.unsqueeze(0), size=(pan_lr.shape[-2], pan_lr.shape[-1]), mode="bicubic",
                          align_corners=False).squeeze(0)

    return {"gt": gt.unsqueeze(0), "ms": ms_lr.unsqueeze(0), "pan": pan_lr.unsqueeze(0), "up_ms": up_ms.unsqueeze(0)}


def tensor_to_rgb(tensor, rgb_indices=[2, 1, 0], enhance=True):
    img_tensor = tensor.squeeze(0).cpu()
    if img_tensor.shape[0] > max(rgb_indices):
        img_tensor = img_tensor[rgb_indices, :, :]

    img_np = img_tensor.numpy().transpose(1, 2, 0)
    if enhance:
        img_np_enhanced = np.zeros_like(img_np)
        for i in range(img_np.shape[2]):
            p2, p98 = np.percentile(img_np[:, :, i], (2, 98))
            img_np_enhanced[:, :, i] = np.clip((img_np[:, :, i] - p2) / (p98 - p2 + 1e-8), 0, 1)
        img_np = img_np_enhanced
    return (img_np * 255).astype(np.uint8)


def calculate_metrics(pred_mask, gt_mask):
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()

    union = tp + fp + fn
    if union == 0: return 0.0, 0.0
    iou = tp / union
    f1 = 2 * tp / (2 * tp + fp + fn)
    return iou, f1


def get_pure_mask_img(mask, color):
    display = np.zeros((*mask.shape, 3), dtype=np.uint8)
    display[mask] = color
    return display


def get_diff_map_img(pred_mask, gt_mask):
    display = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    tp = np.logical_and(pred_mask, gt_mask)
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
    display[tp] = [0, 255, 0]  # 正确提取 (Green)
    display[fp] = [255, 0, 0]  # 误报 (Red)
    display[fn] = [0, 150, 255]  # 漏报 (Blue)
    return display


def generate_task_report(sample, pred_tensor, model_name, target_task, output_dir, sample_name, rgb_indices=[2, 1, 0]):
    gt_tensor = sample['gt']
    gt_rgb = tensor_to_rgb(gt_tensor, rgb_indices)
    pred_rgb = tensor_to_rgb(pred_tensor, rgb_indices)

    rgb_diff = np.mean(np.abs(pred_rgb.astype(np.float32) - gt_rgb.astype(np.float32)), axis=2)
    p_99 = np.percentile(rgb_diff, 99.5)
    rgb_diff_scaled = np.clip(rgb_diff / (p_99 + 1e-8), 0, 1)

    def calc_index(img_tensor, index_type='ndvi'):
        img_np = img_tensor.squeeze(0).cpu().numpy()
        # GF-1: 0=B, 1=G, 2=R, 3=NIR
        g = img_np[1, :, :]
        r = img_np[2, :, :]
        nir = img_np[3, :, :]

        if index_type == 'ndvi':
            return (nir - r) / (nir + r + 1e-8)
        elif index_type == 'ndwi':
            # 标准 NDWI 计算
            return (g - nir) / (g + nir + 1e-8)

    if target_task == 'water':
        gt_index = calc_index(gt_tensor, 'ndwi')
        pred_index = calc_index(pred_tensor, 'ndwi')
        # 调试信息
        print(f"[{model_name}] NDWI Range: min={gt_index.min():.4f}, max={gt_index.max():.4f}")
        # 调整阈值：如果全图蓝色，说明 0.1 太低，尝试调高到 0.2
        threshold = 0.35
        gt_mask = gt_index > threshold
        pred_mask = pred_index > threshold
        mask_color = [34, 139, 34]
        target_name = "Water"

    elif target_task == 'forest':
        gt_index = calc_index(gt_tensor, 'ndvi')
        pred_index = calc_index(pred_tensor, 'ndvi')
        threshold = 0.35
        gt_mask = gt_index > threshold
        pred_mask = pred_index > threshold
        mask_color = [34, 139, 34]
        target_name = "Forest"

    iou, f1 = calculate_metrics(pred_mask, gt_mask)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"[{target_task.upper()} Task] Model: {model_name.upper()} | Sample: {sample_name}", fontsize=16,
                 y=0.98)

    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title('Ground Truth (RGB)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_rgb)
    axes[0, 1].set_title('Prediction (RGB)')
    axes[0, 1].axis('off')

    im_diff = axes[0, 2].imshow(rgb_diff_scaled, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('RGB Error Map')
    axes[0, 2].axis('off')
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    gt_mask_img = get_pure_mask_img(gt_mask, mask_color)
    pred_mask_img = get_pure_mask_img(pred_mask, mask_color)
    diff_map_img = get_diff_map_img(pred_mask, gt_mask)

    axes[1, 0].imshow(gt_mask_img)
    axes[1, 0].set_title(f'GT {target_name} Mask')
    axes[1, 0].axis('off')

    pred_title = f"Pred {target_name} Mask\nIoU: {iou:.4f} | F1: {f1:.4f}"
    axes[1, 1].imshow(pred_mask_img)
    axes[1, 1].set_title(pred_title)
    axes[1, 1].axis('off')

    diff_title = f'{target_name} Difference Map\n(Green:TP | Red:FP | Blue:FN)'
    axes[1, 2].imshow(diff_map_img)
    axes[1, 2].set_title(diff_title)
    axes[1, 2].axis('off')

    plt.tight_layout()
    report_path = os.path.join(output_dir, f"{sample_name}_{model_name}_{target_task}_report.png")
    fig.savefig(report_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    indiv_dir = os.path.join(output_dir, f"{model_name}_details")
    check_and_make(indiv_dir)

    Image.fromarray(gt_rgb).save(os.path.join(indiv_dir, f"{sample_name}_01_GT_RGB.png"))
    Image.fromarray(pred_rgb).save(os.path.join(indiv_dir, f"{sample_name}_02_Pred_RGB.png"))

    fig_err, ax_err = plt.subplots(figsize=(6, 6))
    im_err = ax_err.imshow(rgb_diff_scaled, cmap='hot', vmin=0, vmax=1)
    ax_err.axis('off')
    cbar = fig_err.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
    # cbar.set_label('Error Scale') 

    err_map_path = os.path.join(indiv_dir, f"{sample_name}_03_RGB_ErrorMap.png")
    fig_err.savefig(err_map_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_err)

    Image.fromarray(gt_mask_img).save(os.path.join(indiv_dir, f"{sample_name}_04_GT_{target_name}_Mask.png"))
    Image.fromarray(pred_mask_img).save(os.path.join(indiv_dir, f"{sample_name}_05_Pred_{target_name}_Mask.png"))
    Image.fromarray(diff_map_img).save(os.path.join(indiv_dir, f"{sample_name}_06_{target_name}_Difference_Map.png"))

    print(f"[{model_name}] 任务分析完成。")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = load_sample_data(args.data_dir, args.sample_filename, args.sensor)
    for key in sample:
        if isinstance(sample[key], torch.Tensor):
            sample[key] = sample[key].to(device)

    models_to_run = ['nirfreq', 'lgteun', 'ssamrn']
    sample_name = os.path.splitext(args.sample_filename)[0]
    output_folder = os.path.join(args.output_dir, f"{sample_name}_{args.target_task}")
    check_and_make(output_folder)
    rgb_indices = [int(c) for c in args.rgb_c.split(',')]

    for model_name in models_to_run:
        print(f"\n--- 执行模型: {model_name.upper()} ---")
        ckpt_path = DEFAULT_CKPT_PATHS[model_name]
        if not os.path.exists(ckpt_path): continue

        try:
            if model_name == 'nirfreq':
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                hparams = checkpoint.get('hyper_parameters', {})
                model = NIRFreqModel(
                    lr=0, epochs=0, ms_chans=4, rgb_c=args.rgb_c, sensor=args.sensor,
                    embed_dim=hparams.get('embed_dim', 64),
                    num_layers=hparams.get('num_layers', 9),
                    enable_GCFM=True, enable_HFA=True
                ).to(device)
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                ModelClass = get_model_class(model_name)
                model = ModelClass.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False, map_location=device)

            model.eval()
            with torch.no_grad():
                pred_out = model(sample['up_ms'], sample['pan']) if model_name == 'nirfreq' else model(sample['ms'],
                                                                                                       sample['pan'])
                pred = pred_out[0] if isinstance(pred_out, tuple) else (
                    pred_out['pred'] if isinstance(pred_out, dict) else pred_out)
                pred = torch.clamp(pred, 0.0, 1.0)

            generate_task_report(sample, pred, model_name, args.target_task, output_folder, sample_name, rgb_indices)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_filename', type=str, required=True)
    parser.add_argument('--target_task', type=str, required=True, choices=['water', 'forest'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--rgb_c', type=str, default='2,1,0')
    parser.add_argument('--sensor', type=str, default='GF')
    parser.add_argument('--output_dir', type=str, default='outputs/target_visuals')
    args = parser.parse_args()
    main(args)
