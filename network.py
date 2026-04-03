# src/models/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


class ShallowFeatureExtractor(nn.Module):
    """
    Extracts shallow features from MS image, separating BGR and NIR streams,
    and computes a correlation map between them for guidance in deeper layers.
    """

    def __init__(self, in_channels: int, out_channels: int, sensor: str):
        super().__init__()
        self.sensor = sensor
        # Define separate branches for BGR and NIR to maintain their distinct characteristics
        self.bgr_branch = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.nir_branch = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        # A simple convolution to create a combined MS feature representation
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=True)

    def forward(self, ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Robust band selection: guard against unexpected channel ordering by sensor string
        if self.sensor.lower() in ['wv2', 'wv3']:
            # keep original indexing if you are sure about dataset layout
            bgr = torch.cat([ms[:, 4:5, :, :], ms[:, 2:3, :, :], ms[:, 1:2, :, :]], dim=1)
            nir = ms[:, 6:7, :, :]
        else:
            bgr = ms[:, :3, :, :]
            nir = ms[:, 3:4, :, :]

        bgr_feat = self.bgr_branch(bgr)
        nir_feat = self.nir_branch(nir)

        # cosine similarity in [-1,1] -> map to [0,1] for multiplicative modulation
        corr_raw = F.cosine_similarity(bgr_feat, nir_feat, dim=1).unsqueeze(1)  # shape (B,1,H,W)
        corr_map = (corr_raw + 1.0) * 0.5  # now in [0,1]
        corr_map = corr_map.clamp(0.0, 1.0)

        ms_feat = self.fusion_conv(torch.cat([bgr_feat, nir_feat], dim=1))
        return ms_feat, corr_map, nir_feat


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class FrequencyAwareGCFM(nn.Module):
    class DynamicKernelPredictor(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
            super().__init__()
            self.kernel_size = kernel_size
            self.predictor = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                # predict per-output-channel kernels: out_channels * (k*k)
                nn.Conv2d(in_channels, out_channels * (kernel_size ** 2), 1, 1, 0)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # returns shape (B, out_channels * k^2, H, W)
            return self.predictor(x)

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 3, fixed_alpha: Optional[float] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.gate_ca = ChannelAttention(channels * 3, reduction)
        self.gate_sa = SpatialAttention()
        self.gate_final_conv = nn.Conv2d(channels * 3, channels * 2, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

        # Depthwise high-pass filter initialized with laplacian kernel and frozen
        self.high_pass_filter = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], dtype=torch.float32)
        # register buffer for reproducibility (not trainable)
        self.register_buffer("laplacian_kernel", laplacian_kernel.view(1, 1, 3, 3))
        # initialize depthwise filter weights and freeze them
        with torch.no_grad():
            self.high_pass_filter.weight.data.copy_(self.laplacian_kernel.repeat(channels, 1, 1, 1))
            self.high_pass_filter.weight.requires_grad = False

        self.kernel_predictor = self.DynamicKernelPredictor(in_channels=channels * 3, out_channels=channels,
                                                            kernel_size=kernel_size)

        # Enable fixed alpha for ablation studies
        self.fixed_alpha = fixed_alpha
        if fixed_alpha is not None:
            self.alpha = nn.Parameter(torch.tensor(float(fixed_alpha)), requires_grad=False)
        else:
            self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        self.final_conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, pan_feat: torch.Tensor, ms_feat: torch.Tensor, nir_feat: torch.Tensor, corr_map: torch.Tensor,
                enable_corr_map: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, c, h, w = pan_feat.size()
        pan_high_freq_static = self.high_pass_filter(pan_feat)  # depthwise laplacian

        combined_full = torch.cat([pan_feat, ms_feat, nir_feat], dim=1)  # B, 3C, H, W
        attn_ca = self.gate_ca(combined_full)
        gated_spatial = self.gate_sa(attn_ca)
        gated_features_res = gated_spatial + combined_full

        gates = self.gate_activation(self.gate_final_conv(gated_features_res))
        gate_pan, gate_ms = torch.chunk(gates, 2, dim=1)  # each B, C, H, W

        # dynamic kernels predicted per-output-channel
        dyn_kernels = self.kernel_predictor(gated_features_res)  # B, C*(k^2), H, W
        # reshape to (B*C, k^2, H, W)
        dyn_kernels = dyn_kernels.view(b, c, self.kernel_size ** 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
        # dyn_kernels shape -> (B, C, H, W, k^2)
        # apply to pan by unfolding
        pan_reshaped = pan_feat.view(b * c, 1, h, w)
        unfolded_pan = F.unfold(pan_reshaped, kernel_size=self.kernel_size, padding=self.padding)  # (B*C, k^2, H*W)
        # reshape dynamic kernels to (B*C, k^2, H*W)
        dyn_kernels_flat = dyn_kernels.view(b * c, h * w, self.kernel_size ** 2).permute(0, 2, 1).contiguous()
        # elementwise multiply & sum -> (B*C, H*W)
        pan_details = (unfolded_pan * dyn_kernels_flat).sum(dim=1).view(b, c, h, w)

        dynamic_feat = gate_pan * pan_details + gate_ms * ms_feat
        scaled_static_feat = self.alpha * pan_high_freq_static

        fused_feat = dynamic_feat + scaled_static_feat

        if enable_corr_map:
            # Ensure corr_map broadcastable: corr_map (B,1,Hc,Wc) -> resize to HxW
            if corr_map.shape[-2:] != fused_feat.shape[-2:]:
                corr_map_resized = F.interpolate(corr_map, size=fused_feat.shape[-2:], mode='bilinear',
                                                 align_corners=False)
            else:
                corr_map_resized = corr_map
            modulated = fused_feat * corr_map_resized  # multiplicative guidance
        else:
            modulated = fused_feat

        out = self.final_conv(modulated) + modulated

        # Calculate feature magnitudes for logging
        with torch.no_grad():
            dynamic_mag = dynamic_feat.abs().mean()
            static_mag = scaled_static_feat.abs().mean()
            stats = {
                'gcfm_alpha': self.alpha.detach().clone(),
                'dynamic_feature_mag': dynamic_mag,
                'static_feature_mag': static_mag,
                'static_ratio': static_mag / (static_mag + dynamic_mag + 1e-8)
            }

        return out, stats


class ReconstructionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, max(8, in_channels // 2), 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_channels // 2), out_channels, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GatedResidualModule(nn.Module):
    """
    Optimized Gated Residual Module (GRM) with LayerScale for training stability.
    """

    class InnerResidualBlock(nn.Module):
        def __init__(self, channels: int, init_value: float = 1e-6):
            super().__init__()
            self.norm = nn.LayerNorm(channels)
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1)
            )
            self.gamma = nn.Parameter(init_value * torch.ones(channels), requires_grad=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            normalized_x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return x + self.gamma.view(1, -1, 1, 1) * self.conv(normalized_x)

    def __init__(self, in_channels: int, out_channels: int, num_inner_blocks: int = 2):
        super().__init__()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.inner_blocks = nn.Sequential(*[self.InnerResidualBlock(out_channels) for _ in range(num_inner_blocks)])
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        out = self.inner_blocks(residual)
        return self.final_relu(out + residual)


class HierarchicalFeatureAggregation(nn.Module):
    """
    Upgraded Hierarchical Feature Aggregation module (HFA++) with enhanced semantic consistency modeling.
    """

    def __init__(self, channels: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        descriptor_channels = channels * 3

        self.corr_processor = nn.Sequential(
            nn.Conv1d(1, channels // 4, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, kernel_size=1)
        )
        self.weight_generator = nn.Sequential(
            nn.Conv1d(descriptor_channels * num_layers + channels, channels, kernel_size=1), nn.ReLU(inplace=True)
        )
        self.final_linear = nn.Linear(channels, num_layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_maps: List[torch.Tensor], corr_map: torch.Tensor, global_fused_descriptor: torch.Tensor,
                enable_corr_map: bool = True) -> torch.Tensor:
        b, c, _, _ = feature_maps[0].size()
        layer_descriptors = []
        for feat in feature_maps:
            feat_reshaped = feat.view(b, c, -1)
            mean = torch.mean(feat_reshaped, dim=2)
            std = torch.std(feat_reshaped, dim=2)
            descriptor = torch.cat([mean, std, global_fused_descriptor], dim=1)
            layer_descriptors.append(descriptor)

        if enable_corr_map:
            pooled_corr = F.adaptive_avg_pool2d(corr_map, (1, 1)).view(b, 1, 1)
        else:
            pooled_corr = torch.ones(b, 1, 1, device=corr_map.device)

        processed_corr = self.corr_processor(pooled_corr).squeeze(-1)

        combined_descriptors = torch.cat(layer_descriptors + [processed_corr], dim=1).unsqueeze(2)
        x = self.weight_generator(combined_descriptors).squeeze(2)
        weights = self.softmax(self.final_linear(x)).view(b, self.num_layers, 1, 1, 1)

        stacked_features = torch.stack(feature_maps, dim=1)
        fused_features = torch.sum(stacked_features * weights, dim=1)
        return fused_features


class PanFeatureExtractor(nn.Module):
    def __init__(self, pan_channels: int, feature_channels: int):
        super().__init__()
        self.extractor = nn.Sequential(nn.Conv2d(pan_channels, feature_channels, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True))

    def forward(self, pan: torch.Tensor) -> torch.Tensor:
        return self.extractor(pan)


class PansharpeningNetwork(nn.Module):
    """
    The main backbone network, integrating SAG, HFA++, and optimized MSRBs.
    """

    def __init__(self, ms_channels: int = 4, pan_channels: int = 1, out_channels: int = 4,
                 feature_channels: int = 64, num_blocks: int = 8, sensor: str = 'GF',
                 enable_GCFM: bool = True, enable_HFA: bool = True, enable_corr_map: bool = True,
                 fixed_alpha: Optional[float] = None):
        super().__init__()
        self.enable_GCFM = enable_GCFM
        self.enable_HFA = enable_HFA
        self.enable_corr_map = enable_corr_map
        self.ms_feature_extractor = ShallowFeatureExtractor(ms_channels, feature_channels, sensor)
        self.pan_feature_extractor = PanFeatureExtractor(pan_channels, feature_channels)

        if self.enable_GCFM:
            self.gcfm = FrequencyAwareGCFM(feature_channels, fixed_alpha=fixed_alpha)

        self.num_blocks = num_blocks
        blocks = []
        for i in range(num_blocks):
            input_dim = feature_channels if i == 0 else feature_channels * 2
            blocks.append(GatedResidualModule(in_channels=input_dim, out_channels=feature_channels))
        self.deep_refinement_blocks = nn.ModuleList(blocks)

        if self.enable_HFA:
            self.hfa = HierarchicalFeatureAggregation(feature_channels, num_blocks)

        self.reconstruction = ReconstructionModule(feature_channels, out_channels)

    def forward(self, ms: torch.Tensor, pan: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Now returns ms_feat, corr_map, and the crucial nir_feat for deep fusion
        ms_feat, corr_map, nir_feat = self.ms_feature_extractor(ms)
        pan_feat = self.pan_feature_extractor(pan)

        stats = {}
        # Pass all three features and the correlation map to GCFM for guided fusion
        if self.enable_GCFM:
            fused_feat, stats = self.gcfm(pan_feat, ms_feat, nir_feat, corr_map, self.enable_corr_map)
        else:
            # Fallback logic remains the same
            fused_feat = pan_feat + ms_feat

        global_fused_descriptor = F.adaptive_avg_pool2d(fused_feat, 1).view(fused_feat.size(0), -1)

        layer_outputs = []
        for i, block in enumerate(self.deep_refinement_blocks):
            # Corrected feature reuse: concatenate initial fused feature with the output of the *previous* block
            if i == 0:
                block_input = fused_feat
            else:
                block_input = torch.cat([fused_feat, layer_outputs[-1]], dim=1)

            current_output = block(block_input)
            layer_outputs.append(current_output)

        if self.enable_HFA:
            final_fused_feat = self.hfa(layer_outputs, corr_map, global_fused_descriptor, self.enable_corr_map)
        else:
            final_fused_feat = torch.mean(torch.stack(layer_outputs, dim=1), dim=1)

        residual = self.reconstruction(final_fused_feat)
        return ms + residual, stats


class NIRFreq(nn.Module):
    """
    Top-level wrapper for the PansharpeningNetwork.
    """

    def __init__(self, bands: int, embed_dim: int, sensor: str, num_layers: int = 8,
                 enable_GCFM: bool = True, enable_HFA: bool = True, enable_corr_map: bool = True,
                 fixed_alpha: Optional[float] = None):
        super().__init__()
        self.model = PansharpeningNetwork(
            ms_channels=bands,
            pan_channels=1,
            out_channels=bands,
            feature_channels=embed_dim,
            num_blocks=num_layers,
            sensor=sensor,
            enable_GCFM=enable_GCFM,
            enable_HFA=enable_HFA,
            enable_corr_map=enable_corr_map,
            fixed_alpha=fixed_alpha
        )

    def forward(self, up_ms: torch.Tensor, pan: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.model(up_ms, pan)