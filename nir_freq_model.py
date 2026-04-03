import os
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from thop import profile, clever_format
from torch.optim.lr_scheduler import MultiStepLR

from src.losses.combined_loss import combined_loss
from src.metrics.evaluation import cross_correlation
from src.models.network import NIRFreq
from src.utils.common import check_and_make, regularize_inputs


def _q_index(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    E_a = torch.mean(a, dim=(1, 2))
    E_b = torch.mean(b, dim=(1, 2))
    E_a2 = torch.mean(a * a, dim=(1, 2))
    E_b2 = torch.mean(b * b, dim=(1, 2))
    E_ab = torch.mean(a * b, dim=(1, 2))
    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b
    numerator = 4 * cov_ab * E_a * E_b
    denominator = (var_a + var_b) * (E_a.pow(2) + E_b.pow(2))
    return torch.mean(numerator / (denominator + eps))


def _d_lambda_torch(l_ms: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    ps_downsampled = F.interpolate(ps, size=(l_ms.shape[-2], l_ms.shape[-1]), mode="bicubic", align_corners=False)
    b, l, h, w = ps_downsampled.shape
    indices = torch.triu_indices(l, l, offset=1, device=ps.device)
    q_ps = _q_index(ps_downsampled[:, indices[0]], ps_downsampled[:, indices[1]])
    q_lms = _q_index(l_ms[:, indices[0]], l_ms[:, indices[1]])
    return torch.mean(torch.abs(q_ps - q_lms))


def _d_s_torch(l_ms: torch.Tensor, pan: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    b, l, h, w = ps.shape
    l_ms_upsampled = F.interpolate(l_ms, size=(h, w), mode="bicubic", align_corners=False)
    l_pan = F.interpolate(pan, size=(h, w), mode="bicubic", align_corners=False)
    q_ps_pan = _q_index(ps, l_pan.expand(-1, l, -1, -1))
    q_lms_lpan = _q_index(l_ms_upsampled, l_pan.expand(-1, l, -1, -1))
    return torch.mean(torch.abs(q_ps_pan - q_lms_lpan))


class NIRFreqModel(pl.LightningModule):
    def __init__(
            self,
            lr: float,
            epochs: int,
            ms_chans: int,
            rgb_c: str,
            sensor: str,
            embed_dim: int,
            num_layers: int,
            enable_GCFM: bool = True,
            enable_HFA: bool = True,
            enable_corr_map: bool = True,
            test_freq: int = 10,
            l1_weight: float = 0.9,
            mse_weight: float = 0.1,
            ssim_weight: float = 0,
            test_mode: str = 'reduced',
            output_dir: str = 'test_results',
            fixed_alpha: Optional[float] = None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.rgb_c = [int(c) for c in self.hparams.rgb_c.split(",")]
        self.model = NIRFreq(
            bands=self.hparams.ms_chans,
            embed_dim=self.hparams.embed_dim,
            num_layers=self.hparams.num_layers,
            sensor=self.hparams.sensor,
            enable_GCFM=self.hparams.enable_GCFM,
            enable_HFA=self.hparams.enable_HFA,
            enable_corr_map=self.hparams.enable_corr_map,
            fixed_alpha=self.hparams.fixed_alpha
        )

        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

        self.reset_val_metrics()
        self.epoch_gcfm_stats = []  # 用于缓存每一轮的 Alpha 统计数据

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        lr_scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=[80, 160, 240, 320, 400], gamma=0.6),
            'interval': 'epoch', 'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    def forward(self, ms: torch.Tensor, pan: torch.Tensor) -> Dict[str, Any]:
        pred, stats = self.model(ms, pan)
        return {"pred": pred, "stats": stats}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pan, ms, gt, up_ms = batch["pan"], batch["ms"], batch["gt"], batch["up_ms"]
        batch_size = pan.size(0)

        forward_out = self(up_ms, pan)
        pred = forward_out["pred"]
        stats = forward_out["stats"]

        opt = self.optimizers()
        opt.zero_grad()

        total_loss, log_dict = combined_loss(
            pred, gt, l1_weight=self.l1_weight, mse_weight=self.mse_weight, ssim_weight=self.ssim_weight
        )

        self.manual_backward(total_loss)
        opt.step()

        lr = opt.param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=batch_size)
        for key, value in log_dict.items():
            self.log(f"train/{key}", value, prog_bar=(key == "total_loss"), logger=True, on_step=False, on_epoch=True,
                     batch_size=batch_size)

        # 缓存当前 batch 的 stats，准备在 epoch 结束时单独写入文件
        if stats:
            self.epoch_gcfm_stats.append({k: v.item() for k, v in stats.items()})

    def on_train_epoch_end(self) -> None:
        # 单独处理和保存 Alpha 等统计数据到独立文件
        if self.epoch_gcfm_stats:
            # 计算该 Epoch 的平均值
            avg_stats = {k: np.mean([d[k] for d in self.epoch_gcfm_stats]) for k in self.epoch_gcfm_stats[0].keys()}
            avg_stats['epoch'] = self.current_epoch

            # 找到日志目录
            log_dir = self.trainer.log_dir if self.trainer.log_dir else "."
            csv_path = os.path.join(log_dir, "gcfm_alpha_stats.csv")

            # 保存到单独的 CSV 文件
            df = pd.DataFrame([avg_stats])
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, mode='a', header=False, index=False)

            # 清空缓存供下一个 Epoch 使用
            self.epoch_gcfm_stats = []

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pan, ms, gt, up_ms = batch["pan"], batch["ms"], batch["gt"], batch["up_ms"]
        batch_size = pan.size(0)

        forward_out = self(up_ms, pan)
        pred = forward_out["pred"]

        pred, gt, up_ms = regularize_inputs(pred, gt, up_ms)

        _, log_dict = combined_loss(
            pred, gt, l1_weight=self.l1_weight, mse_weight=self.mse_weight, ssim_weight=self.ssim_weight
        )
        for key, value in log_dict.items():
            self.log(f"val/{key}", value, prog_bar=(key in ["total_loss", "ssim_value"]), logger=True,
                     batch_size=batch_size)

        self.save_val_ref_metrics(pred, gt)

    def on_validation_epoch_end(self) -> None:
        for metric_key in self.val_metric_keys:
            full_metric_name = f"val/{metric_key}"
            if self.val_metrics_all[full_metric_name]:
                mean = np.mean(self.val_metrics_all[full_metric_name])
                std = np.std(self.val_metrics_all[full_metric_name])
                self.log(f"{full_metric_name}_mean", round(mean, 4), prog_bar=(metric_key in ["PSNR", "SAM"]))
                self.log(f"{full_metric_name}_std", round(std, 4))

        self.reset_val_metrics()

    def reset_val_metrics(self) -> None:
        self.val_metric_keys = ['PSNR', 'SSIM', 'SAM', 'ERGAS', 'CC', 'RMSE', 'MAE', 'RASE', 'UQI']
        self.val_metrics_all = {f"val/{key}": [] for key in self.val_metric_keys}

    def record_val_metric(self, k: str, v: torch.Tensor) -> None:
        metric_name = f'val/{k}'
        if torch.isfinite(v):
            self.val_metrics_all[metric_name].append(v.item())

    def save_val_ref_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        data_range = 1.0
        self.record_val_metric('MAE', F.l1_loss(pred, gt))
        self.record_val_metric('SSIM', MF.structural_similarity_index_measure(pred, gt, data_range=data_range))
        self.record_val_metric('RMSE', torch.sqrt(F.mse_loss(pred, gt)))
        self.record_val_metric('ERGAS', self.calculate_ergas_corrected(pred, gt, ratio=4.0))
        self.record_val_metric('SAM', MF.spectral_angle_mapper(pred, gt))
        self.record_val_metric('RASE', MF.relative_average_spectral_error(pred, gt))
        self.record_val_metric('PSNR', MF.peak_signal_noise_ratio(pred, gt, data_range=data_range))
        self.record_val_metric('UQI', MF.universal_image_quality_index(pred, gt))
        self.record_val_metric('CC', cross_correlation(pred, gt))

    def on_test_epoch_start(self) -> None:
        self.results = []
        check_and_make(self.hparams.output_dir)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        filename = batch.get("filename", [f"sample_{batch_idx}"])[0]
        pan, ms, up_ms = batch["pan"], batch["ms"], batch["up_ms"]

        start_time = time.time()
        forward_out = self(up_ms, pan)
        pred = forward_out["pred"]
        end_time = time.time()

        sample_metrics = {"filename": filename}

        if self.hparams.test_mode == 'reduced':
            gt = batch["gt"]
            pred_reg, gt_reg = regularize_inputs(pred, gt)
            sample_metrics.update(self.calculate_ref_metrics(pred_reg, gt_reg))
        else:
            pred_reg, = regularize_inputs(pred)
            noref_metrics = self.calculate_noref_metrics(ms, pan, pred_reg)

            d_lambda_val = noref_metrics.get('D_lambda')
            if d_lambda_val is None or not np.isfinite(d_lambda_val) or d_lambda_val > 0.02:
                return

            for metric_name, metric_value in noref_metrics.items():
                if metric_name == 'D_lambda':
                    continue
                if not np.isfinite(metric_value):
                    return

            sample_metrics.update(noref_metrics)

        sample_metrics['Time'] = end_time - start_time
        self.results.append(sample_metrics)

    def on_test_epoch_end(self) -> None:
        if not self.results:
            print("No results to save.")
            return

        result_dir = os.path.join(self.hparams.output_dir, self.hparams.sensor.upper(), self.hparams.test_mode)
        check_and_make(result_dir)

        df = pd.DataFrame(self.results).set_index("filename")
        suffix = f"_alpha_{self.hparams.fixed_alpha}" if self.hparams.fixed_alpha is not None else ""

        csv_filename = f"metrics_detailed{suffix}.csv"
        csv_path = os.path.join(result_dir, csv_filename)
        df.to_csv(csv_path)

        summary_stats = {col: {'mean': df[col].mean(), 'std': df[col].std()} for col in df.columns}
        summary_filename = f"metrics_summary{suffix}.txt"
        summary_path = os.path.join(result_dir, summary_filename)

        summary_lines = []
        summary_lines.append(
            f"{self.hparams.test_mode.capitalize()} resolution summary (Alpha: {self.hparams.fixed_alpha if self.hparams.fixed_alpha is not None else 'Learnable'}):")
        for metric, stats in summary_stats.items():
            summary_lines.append(f"  {metric}:  {stats['mean']:.4f} ± {stats['std']:.4f}")

        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))

        print("\n--- Test Summary ---")
        for line in summary_lines:
            print(line)
        print("--------------------")

    def calculate_ref_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
        data_range = 1.0
        metrics = {
            'PSNR': MF.peak_signal_noise_ratio(pred, gt, data_range=data_range).item(),
            'SSIM': MF.structural_similarity_index_measure(pred, gt, data_range=data_range).item(),
            'SAM': MF.spectral_angle_mapper(pred, gt).item(),
            'ERGAS': self.calculate_ergas_corrected(pred, gt, ratio=4.0).item(),
            'CC': cross_correlation(pred, gt).item(),
            'RMSE': torch.sqrt(F.mse_loss(pred, gt)).item(),
            'UQI': MF.universal_image_quality_index(pred, gt).item(),
        }
        return metrics

    def calculate_noref_metrics(self, lrms: torch.Tensor, pan: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
        d_lambda = _d_lambda_torch(lrms, pred).item()
        d_s = _d_s_torch(lrms, pan, pred).item()
        qnr = (1 - d_lambda) * (1 - d_s)
        metrics = {'D_lambda': d_lambda, 'D_s': d_s, 'QNR': qnr}
        return metrics

    def calculate_ergas_corrected(self, pred: torch.Tensor, gt: torch.Tensor, ratio: float,
                                  eps: float = 1e-8) -> torch.Tensor:
        DATA_MAX = 1023.0 if self.hparams.sensor.lower() == 'gf' else 2047.0
        pred_un, gt_un = pred * DATA_MAX, gt * DATA_MAX
        n_bands = pred.shape[1]
        sum_ratio_sq = 0.0
        for i in range(n_bands):
            rmse_i = torch.sqrt(F.mse_loss(pred_un[:, i, :, :], gt_un[:, i, :, :]) + eps)
            mu_i = torch.mean(gt_un[:, i, :, :])
            mu_i = torch.clamp(mu_i, min=eps)
            sum_ratio_sq += (rmse_i / mu_i) ** 2
        return 100.0 * torch.sqrt(sum_ratio_sq / n_bands)

    def count(self) -> None:
        up_ms = torch.rand(1, self.hparams.ms_chans, 256, 256)
        pan = torch.rand(1, 1, 256, 256)
        macs, params = profile(self.model, inputs=(up_ms, pan), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"Model MACs: {macs}, Params: {params}")