import torch.nn as nn
import torch
import torch.nn.functional as F
from models.mlp import MLP

class TimeSeriesEncoder(nn.Module):
    def __init__(self, sequence_length: int, in_channels: int, channels:list, channels_final_to_mlp:int, kernel_sizes:list, strides:list, latent_dim=128, activation=nn.ReLU, use_norm=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.use_norm = use_norm
        self.activation_cls = activation

        # --- Backbone (Conv/Pool layers) ---
        backbone_layers = []
        current_channels = in_channels
        for _, (out_ch, kernel_size, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            # Convolution
            backbone_layers.append(nn.Conv1d(in_channels=current_channels, out_channels=out_ch, 
                                             kernel_size=kernel_size, stride=stride, 
                                             padding=kernel_size//2, groups=in_channels))
            # Normalization
            if use_norm:
                backbone_layers.append(nn.BatchNorm1d(out_ch))
            # Activation
            if activation:
                backbone_layers.append(activation(inplace=True))
            # Pooling
            backbone_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            current_channels = out_ch
        self.backbone = nn.Sequential(*backbone_layers)

        # --- Dummy forward pass to find shape for MLP & record lengths ---
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, sequence_length)
            self.lengths = [sequence_length]  # L0 original
            tmp = dummy_input
            layer_idx = 0
            for layer in self.backbone:
                tmp = layer(tmp)
                # Record length right AFTER each pooling layer (stage boundary)
                if isinstance(layer, nn.MaxPool1d):
                    self.lengths.append(tmp.shape[2])
                layer_idx += 1
            self.reduced_seq_len = tmp.shape[2]
            self.channels_final_to_mlp = channels_final_to_mlp

        # --- Head (Projection layers) ---
        self.head = nn.Sequential(
            nn.Conv1d(in_channels=channels[-1], out_channels=channels_final_to_mlp, kernel_size=1, groups=in_channels),
            nn.Flatten(),
            nn.Linear(channels_final_to_mlp * self.reduced_seq_len, latent_dim),
            # MLP(channels_final_to_mlp * self.reduced_seq_len, [4 * latent_dim], latent_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.backbone(x)
        z = self.head(x) # (B, latent)
        return z

class TimeSeriesDecoder(nn.Module):
    """Decoder that exactly reconstructs the original sequence length using stored encoder lengths.

    Pass the encoder.lengths list so that each upsampling step targets a concrete length
    instead of relying on heuristic scale factors. This avoids trimming/padding hacks.
    """
    def __init__(self, sequence_length: int, reduced_seq_len: int, in_channels: int,
                 channels: list, channels_final_to_mlp: int, kernel_sizes: list,
                 strides: list, latent_dim=128, activation=nn.ReLU, use_norm=True,
                 lengths: list | None = None):
        super().__init__()
        self.sequence_length = sequence_length
        self.reduced_seq_len = reduced_seq_len
        self.channels_final_to_mlp = channels_final_to_mlp
        self.activation_cls  = activation
        self.use_norm        = use_norm
        self.in_channels     = in_channels

        if lengths is None:
            # Fallback assumption (may be slightly off if padding effects differ)
            raise ValueError("TimeSeriesDecoder requires the encoder 'lengths' list for exact reconstruction.")
        self.lengths = lengths  # [L0, L1, ..., Lm] where Lm == reduced_seq_len
        assert self.lengths[-1] == reduced_seq_len, "Provided lengths inconsistent with reduced_seq_len"

        # --- Head (MLP inverse of encoder head) ---
        self.head_mlp = nn.Linear(latent_dim,channels_final_to_mlp * reduced_seq_len)
        # self.head_mlp = MLP(latent_dim, [4 * latent_dim], channels_final_to_mlp * reduced_seq_len)
        self.head_conv1x1 = nn.Conv1d(in_channels=channels_final_to_mlp, out_channels=channels[-1], kernel_size=1, groups=in_channels)

        # Build per-stage conv blocks (reverse of encoder). We will interpolate explicitly to target lengths.
        reversed_channels = list(reversed(channels))  # [cm, ..., c1]
        reversed_kernel_sizes = list(reversed(kernel_sizes))
        reversed_strides = list(reversed(strides))

        # For each stage we go from channels_i to channels_{i-1}
        conv_blocks = []
        for idx in range(len(reversed_channels) - 1):
            in_ch  = reversed_channels[idx]
            out_ch = reversed_channels[idx + 1]
            k      = reversed_kernel_sizes[idx]
            stri   = reversed_strides[idx]
            conv   = [nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, padding=k//2, stride=stri, groups=in_channels)]
            if use_norm:
                conv.append(nn.BatchNorm1d(out_ch))
            if activation:
                conv.append(activation())
            conv_blocks.append(nn.Sequential(*conv))
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # Final projection (after last conv block) to original input channels
        self.final_proj = nn.Conv1d(in_channels=channels[0], out_channels=in_channels,
                                    kernel_size=reversed_kernel_sizes[-1], padding=reversed_kernel_sizes[-1]//2)

    def forward(self, z):
        # z: (B, latent_dim)
        # Head
        x = self.head_mlp(z)
        x = x.view(-1, self.channels_final_to_mlp, self.reduced_seq_len)
        x = self.head_conv1x1(x)  # (B, channels[-1], Lm)

        # lengths: [L0(original), L1, ..., Lm(current)]
        # We traverse conv_blocks; for block i we upscale to target length lengths[-(i+2)] (the previous stage length)
        # reversed order lengths_rev = [Lm, ..., L0]
        lengths_rev = list(reversed(self.lengths))  # [Lm, L_{m-1}, ..., L0]

        for i, block in enumerate(self.conv_blocks):
            target_len = lengths_rev[i + 1]  # length we need after this upsampling (previous stage length)
            # First upsample to target length (linear interpolation)
            if x.shape[2] != target_len:
                x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
            # Then apply conv block to adjust channels
            x = block(x)

        # Final ensure size matches original (L0)
        if x.shape[2] != self.lengths[0]:
            x = F.interpolate(x, size=self.lengths[0], mode='linear', align_corners=False)
        x = self.final_proj(x)
        assert x.shape[2] == self.sequence_length, "Decoder reconstruction length mismatch"
        # Return (B, seq_len, features)
        return x.permute(0, 2, 1)

class SqueezeExcite(nn.Module):
    """Simple Squeeze-and-Excitation block.
    Reduces channel dimension, applies activation, then expands back"""
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Squeeze: Global information
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid())  # Excitation: Get channel weights
    def forward(self, x):
        return x * self.se(x)

class ResidualBlock(nn.Module):
    """An improved convolutional block with:
    1. Residual connection.
    2. Bottleneck design (1x1 -> Depthwise -> 1x1).
    3. Squeeze-and-Excitation.
    4. Optional downsampling via strided convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=4, use_se=False):
        super().__init__()
        self.stride  = stride
        mid_channels = in_channels * expansion_factor
        
        # self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.conv = nn.Sequential(
            # 1. Pointwise Conv (Expansion)
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU(inplace=True),
            # 2. Depthwise Conv (Spatial Filtering)
            nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, groups=mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU(inplace=True),
            # 3. Squeeze-and-Excitation
            SqueezeExcite(mid_channels) if use_se else nn.Identity(),
            # 4. Pointwise Conv (Projection)
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels))
        
        self.shortcut = nn.Identity()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        return self.shortcut(x) + self.conv(x)

class TimeSeriesEncoder_v2(nn.Module):
    """A Time Series Encoder using a backbone of ResidualBlocks.
    This version uses strided convolutions for downsampling and learns cross-feature
    interactions via a bottleneck design"""
    def __init__(self, sequence_length: int, in_channels: int, channels:list, latent_dim=128, 
                 block_kernels:list | None = None, block_strides:list | None = None, 
                 expansion_factor=4, use_se=True):
        super().__init__()
        self.latent_dim      = latent_dim
        self.in_channels     = in_channels
        self.sequence_length = sequence_length
        self.lengths         = [sequence_length]

        if block_kernels is None:
            block_kernels = [3] * len(channels)
        if block_strides is None:
            block_strides = [1] * len(channels)

        # --- Backbone with ResidualBlocks ---
        # 1. Initial "Stem" convolution
        stem_channels = channels[0]
        backbone = [nn.Conv1d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(stem_channels),
                    nn.SiLU(inplace=True)]
        
        # 2. Stack of ResidualBlocks
        current_channels = stem_channels
        for _, (out_ch, kernel, stride) in enumerate(zip(channels, block_kernels, block_strides)):
            backbone.append(
                ResidualBlock(current_channels, out_ch, kernel_size=kernel, stride=stride, 
                              expansion_factor=expansion_factor, use_se=use_se))
            current_channels = out_ch
        self.backbone = nn.Sequential(*backbone)

        # --- Dummy forward pass to find shape for MLP & record lengths ---
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, sequence_length)
            
            # Pass through layer by layer to record lengths after strided convs
            tmp = dummy_input
            for layer in self.backbone:
                tmp = layer(tmp)
                # Find ResidualBlocks with downsampling to record new length
                if isinstance(layer, ResidualBlock) and layer.stride > 1:
                    self.lengths.append(tmp.shape[2])

            self.reduced_seq_len    = tmp.shape[2]
            final_backbone_channels = channels[-1]

        # --- Head (Projection layers) ---
        # Global average pooling + MLP is a common and effective head
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  nn.Flatten(),
                                  nn.Linear(final_backbone_channels, latent_dim))

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.backbone(x)
        z = self.head(x) # (batch, latent)
        return z
    
class TimeSeriesDecoder_v2(nn.Module):
    """A decoder designed to mirror the TimeSeriesEncoder_v2.
    It uses learned upsampling (ConvTranspose1d) and ResidualBlocks to reconstruct the time series."""
    def __init__(self, sequence_length: int, out_channels: int, channels: list, latent_dim=128,
                 block_kernels: list | None = None, block_strides: list | None = None,
                 expansion_factor=4, use_se=True, reduced_seq_len=None, initial_channels=None):
        super().__init__()
        self.sequence_length = sequence_length
        self.latent_dim      = latent_dim
        
        if initial_channels is None:
            initial_channels = channels[-1] # Last channel count from encoder
        if reduced_seq_len is None:
            # Calculate based on strides if not provided
            total_stride = 1
            for s in block_strides: total_stride *= s
            reduced_seq_len = sequence_length // total_stride

        # --- Head: Project latent vector to a small, deep feature map ---
        self.head = nn.Sequential(
            nn.Linear(latent_dim, initial_channels * reduced_seq_len),
            nn.Unflatten(1, (initial_channels, reduced_seq_len)))

        # --- Upsampling Backbone ---
        backbone = []
        reversed_channels = list(reversed(channels))
        reversed_strides  = list(reversed(block_strides))
        reversed_kernels  = list(reversed(block_kernels))
        current_channels  = initial_channels

        for _, (out_ch, stride, kernel) in enumerate(zip(reversed_channels, reversed_strides, reversed_kernels)):
            # 1. Learned Upsampling if stride > 1
            if stride > 1:
                backbone.append(nn.ConvTranspose1d(current_channels, out_ch, kernel_size=stride, stride=stride))
            
            # 2.a Refine with a simple convolution (no expansion)
            # if stride == 1:
            #     backbone.append(nn.Conv1d(current_channels, out_ch, kernel_size=kernel, padding=1))
            # backbone.append(nn.BatchNorm1d(out_ch))
            # backbone.append(nn.SiLU(inplace=True))
            # current_channels = out_ch

            # 2.b Refine with a Residual Block
            if stride == 1:
                backbone.append(
                    ResidualBlock(out_ch if stride > 1 else current_channels, out_ch, kernel_size=kernel, 
                                expansion_factor=expansion_factor, use_se=use_se))
            current_channels = out_ch
        self.backbone = nn.Sequential(*backbone)

        # --- Tail: Project back to original number of channels ---
        self.tail = nn.Conv1d(current_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.head(z)
        x = self.backbone(x)
        x = self.tail(x)

        # Final interpolation to guarantee exact sequence length
        if x.shape[2] != self.sequence_length:
            x = F.interpolate(x, size=self.sequence_length, mode='linear', align_corners=False)
        return x.permute(0, 2, 1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# V3 Models: InceptionTime-inspired Multi-Scale Convolutions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class InceptionModule(nn.Module):
    """An Inception-style module for time series.
    It applies multiple parallel convolutions with different kernel sizes to the input.
    A bottleneck layer is used to reduce computational cost."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 11], use_bottleneck=True):
        super().__init__()
        
        # Ensure out_channels is divisible by the number of parallel paths
        # if out_channels % (len(kernel_sizes) + 1) != 0:
        #     raise ValueError(f"out_channels ({out_channels}) must be divisible by number of kernels + 1 ({len(kernel_sizes) + 1})")
        
        self.out_channels_per_path = out_channels // (len(kernel_sizes) + 1)
        self.remainder_channels    = out_channels - self.out_channels_per_path * (len(kernel_sizes) + 1)
        self.use_bottleneck        = use_bottleneck
        bottleneck_channels        = in_channels // len(kernel_sizes) if use_bottleneck else in_channels

        # --- Parallel Convolutional Paths ---
        # 1. Bottleneck -> Conv branches
        self.conv_branches = nn.ModuleList()
        for k in kernel_sizes:
            layers = []
            if use_bottleneck:
                layers.append(nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False))
            
            layers.append(nn.Conv1d(
                bottleneck_channels if use_bottleneck else in_channels,
                self.out_channels_per_path,
                kernel_size=k,
                padding=k//2,
                bias=False))
            self.conv_branches.append(nn.Sequential(*layers))

        # 2. MaxPool -> Conv branch
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, self.out_channels_per_path + self.remainder_channels, kernel_size=1, bias=False))

        # --- Batch Normalization and Activation ---
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        # Apply each parallel branch
        outputs = [self.conv_branches[0](x)]
        target_len = outputs[0].shape[2]

        for i in range(1, len(self.conv_branches)):
            branch_out = self.conv_branches[i](x)
            if branch_out.shape[2] != target_len:
                branch_out = F.interpolate(branch_out, size=target_len, mode='linear', align_corners=False)
            outputs.append(branch_out)

        # Handle the pool branch separately
        pool_out = self.pool_branch(x)
        if pool_out.shape[2] != target_len:
            pool_out = F.interpolate(pool_out, size=target_len, mode='linear', align_corners=False)
        outputs.append(pool_out)
        
        # Concatenate outputs along the channel dimension
        x = torch.cat(outputs, 1)
        
        # Apply batch norm and activation
        x = self.bn(x)
        x = self.activation(x)
        return x

class InceptionBlock(nn.Module):
    """A block that wraps the InceptionModule with a residual connection.
    Handles downsampling via strided convolution in the shortcut connection."""
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1, use_bottleneck=True):
        super().__init__()
        self.stride = stride

        # Inception Module
        self.inception_module = InceptionModule(in_channels, out_channels, kernel_sizes, use_bottleneck)
        
        # Downsampling layer for the main path, if needed.
        # Must match the downsampling of the shortcut connection.
        self.downsample = nn.Identity()
        if stride > 1:
            self.downsample = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=stride, groups=in_channels, bias=False)

        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        # The shortcut path defines the target output shape (length and channels)
        shortcut_out = self.shortcut(x)

        # The main path operates on the original input
        inception_out = self.inception_module(x)

        # If the main path's length differs from the shortcut's, resize it.
        # This handles any off-by-one errors from different downsampling calculations.
        if inception_out.shape[2] != shortcut_out.shape[2]:
            inception_out = F.interpolate(inception_out, size=shortcut_out.shape[2], mode='linear', align_corners=False)
        
        # Add residual and apply final activation
        return self.activation(inception_out + shortcut_out)

class TimeSeriesEncoder_v3(nn.Module):
    """A TimeSeriesEncoder based on a stack of InceptionBlocks."""
    def __init__(self, sequence_length: int, in_channels: int, channels: list, latent_dim=128, 
                 block_strides: list | None = None, 
                 kernel_sizes_per_block: list | None = None, use_bottleneck=True):
        super().__init__()
        self.latent_dim      = latent_dim
        self.in_channels     = in_channels
        self.sequence_length = sequence_length
        self.lengths         = [sequence_length]

        if block_strides is None: block_strides = [1] * len(channels)
        if kernel_sizes_per_block is None: kernel_sizes_per_block = [[3, 5, 7]] * len(channels)

        # --- Backbone with InceptionBlocks ---
        stem_channels = channels[0]
        backbone = [nn.Conv1d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(stem_channels),
                    nn.SiLU(inplace=True)]
        current_channels = stem_channels
        for _, (out_ch, stride, kernels) in enumerate(zip(channels, block_strides, kernel_sizes_per_block)):
            backbone.append(
                InceptionBlock(current_channels, out_ch, kernel_sizes=kernels, stride=stride, use_bottleneck=use_bottleneck))
            current_channels = out_ch
        self.backbone = nn.Sequential(*backbone)

        # --- Dummy forward pass to find shape for MLP & record lengths ---
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, sequence_length)
            tmp = self.backbone(dummy_input)
            self.reduced_seq_len = tmp.shape[2]
            final_backbone_channels = tmp.shape[1]

        # --- Head (Projection layers) ---
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_backbone_channels, latent_dim))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.backbone(x)
        z = self.head(x)
        return z

class TimeSeriesDecoder_v3(nn.Module):
    """A decoder designed to mirror the TimeSeriesEncoder_v3."""
    def __init__(self, sequence_length: int, out_channels: int, channels: list, latent_dim=128,
                 block_strides: list | None = None, kernel_sizes_per_block: list | None = None,
                 use_bottleneck=True, reduced_seq_len=None, initial_channels=None):
        super().__init__()
        self.sequence_length = sequence_length
        
        if initial_channels is None: initial_channels = channels[-1]
        if reduced_seq_len is None:
            total_stride = 1
            for s in block_strides: total_stride *= s
            reduced_seq_len = sequence_length // total_stride

        # --- Head: Project latent vector to a small, deep feature map ---
        self.head = nn.Sequential(
            nn.Linear(latent_dim, initial_channels * reduced_seq_len),
            nn.Unflatten(1, (initial_channels, reduced_seq_len)))

        # --- Upsampling Backbone ---
        backbone = []
        reversed_channels = list(reversed(channels))
        reversed_strides  = list(reversed(block_strides))
        reversed_kernels  = list(reversed(kernel_sizes_per_block))

        current_channels = initial_channels
        for _, (out_ch, stride, kernels) in enumerate(zip(reversed_channels, reversed_strides, reversed_kernels)):
            # 1. Learned Upsampling if stride > 1
            if stride > 1:
                backbone.append(
                    nn.ConvTranspose1d(current_channels, out_ch, kernel_size=stride, stride=stride))
                current_channels = out_ch

            # 2. Refine with an Inception Block
            backbone.append(
                InceptionBlock(current_channels, out_ch, kernel_sizes=kernels, stride=1, use_bottleneck=use_bottleneck))
            current_channels = out_ch
        
        self.backbone = nn.Sequential(*backbone)

        # --- Tail: Project back to original number of channels ---
        self.tail = nn.Conv1d(current_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.head(z)
        x = self.backbone(x)
        x = self.tail(x)

        if x.shape[2] != self.sequence_length:
            x = F.interpolate(x, size=self.sequence_length, mode='linear', align_corners=False)
        
        return x.permute(0, 2, 1)
