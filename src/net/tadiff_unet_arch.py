from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


from src.net.utils import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    count_flops_attn,
    FourierFeatures,
)


def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    """
    Numerically stable softmax variant with +1 in denominator.
    
    Improves stability by:
    1. Subtracting max for numerical stability
    2. Adding 1 to denominator to mitigate tiny value effects
    
    Args:
        x: Input tensor
        dim: Dimension to compute softmax over
        _stacklevel: Internal parameter for warning traces
        dtype: Optional output dtype
        
    Returns:
        Tensor with softmax_one applied along specified dimension
    
    Note:
        Adapted from: https://github.com/kyegomez/AttentionIsOFFByOne/blob/main/softmax_one/softmax_one.py
    """
    x = x - x.max(dim=dim, keepdim=True).values  # Subtract max for stability
    exp_x = th.exp(x)  # Compute exponentials
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))  # Softmax with +1 denominator


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


class AttentionPool2d(nn.Module):
    """
    2D Attention Pooling layer adapted from CLIP architecture.
    
    Performs spatial attention pooling by:
    1. Flattening spatial dimensions
    2. Adding mean-pooled global feature
    3. Applying learned positional embeddings
    4. Computing attention-weighted features
    
    Args:
        spacial_dim: Input spatial dimension (assumed square)
        embed_dim: Input feature dimension
        num_heads_channels: Channels per attention head
        output_dim: Optional output dimension (defaults to embed_dim)
        
    Forward Input:
        x: [batch_size, channels, height, width] feature map
        
    Forward Output:
        [batch_size, output_dim] pooled features
        
    Note:
        Original implementation from:
        https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]  # Return pooled features


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    N-dimensional self-attention block for diffusion models.
    
    Implements multi-head attention with optional checkpointing and supports:
    - Both legacy and optimized attention computation orders
    - Flexible head configuration via channels or num_heads
    - Memory-efficient gradient checkpointing
    
    Args:
        channels: Input feature channels
        num_heads: Number of attention heads
        num_head_channels: Channels per head (alternative to num_heads)
        use_checkpoint: Enable gradient checkpointing
        use_new_attention_order: Use optimized attention computation order
        
    Forward Input:
        x: [batch_size, channels, *spatial_dims] feature tensor
        
    Forward Output:
        [batch_size, channels, *spatial_dims] attended features
        
    Note:
        Adapted from:
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)  # Split qkv before heads
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)  # Split heads first

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # Flatten spatial dimensions
        qkv = self.qkv(self.norm(x))  # Compute queries, keys, values
        h = self.attention(qkv)  # Apply attention
        h = self.proj_out(h)  # Project back
        return (x + h).reshape(b, c, *spatial)  # Residual connection



class QKVAttentionLegacy(nn.Module):
    """
    Legacy QKV attention implementation that splits heads before splitting QKV.
    
    Performs multi-head attention with:
    - Split heads first architecture
    - Scaled dot-product attention
    - Softmax_one for numerical stability
    - Efficient einsum operations
    
    Args:
        n_heads: Number of attention heads
        
    Forward Input:
        qkv: [batch_size, (heads * 3 * channels), seq_len] concatenated Q,K,V
        
    Forward Output:
        [batch_size, (heads * channels), seq_len] attention output
        
    Note:
        Uses softmax_one instead of standard softmax for better numerical stability
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        
        # Split into heads first, then QKV
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        
        # Scaled dot-product attention
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        
        # Attention weights with softmax_one
        weight = softmax_one(weight.float(), dim=-1).type(weight.dtype)
        
        # Apply attention to values
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)  # Merge heads

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    Optimized QKV attention that splits QKV before splitting heads.
    
    Performs multi-head attention with:
    - Split QKV first architecture (more efficient)
    - Scaled dot-product attention
    - Softmax_one for numerical stability
    - Optimized memory layout
    
    Args:
        n_heads: Number of attention heads
        
    Forward Input:
        qkv: [batch_size, (3 * heads * channels), seq_len] concatenated Q,K,V
        
    Forward Output:
        [batch_size, (heads * channels), seq_len] attention output
        
    Note:
        More efficient than legacy version due to better memory access patterns
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        
        # Split QKV first, then heads
        q, k, v = qkv.chunk(3, dim=1)
        
        # Scaled dot-product attention
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        
        # Attention weights with softmax_one
        weight = softmax_one(weight.float(), dim=-1).type(weight.dtype)
        
        # Apply attention to values
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)  # Merge heads

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


def expand_to_planes(input, shape):
    """
    Expand 1D or 2D input to match target feature plane dimensions.
    
    Args:
        input: [batch_size, channels] or [batch_size, channels, 1, 1] tensor
        shape: Target shape tuple to match spatial dimensions
        
    Returns:
        [batch_size, channels, height, width] tensor with input repeated spatially
    """
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])



class TaDiff_Net(nn.Module):
    """
    A specialized UNet model for treatment-aware diffusion processes with temporal conditioning.

    This model extends standard diffusion UNet architecture with:
    - Treatment code conditioning (treat_code)
    - Temporal interval conditioning between reference/target exams (intv_t)
    - Target index specification (i_tg)

    Key Features:
    - Three parallel embedding streams: time, days, and treatments
    - Relative treatment-day difference computation
    - Multi-scale feature conditioning
    - Custom softmax_one attention for numerical stability

    Architecture Parameters:
    :param in_channels: channels in the input Tensor
    :param model_channels: base channel count for the model
    :param out_channels: channels in the output Tensor
    :param num_res_blocks: number of residual blocks per downsample
    :param attention_resolutions: collection of downsample rates for attention layers
    :param dropout: dropout probability
    :param channel_mult: channel multiplier for each UNet level
    :param conv_resample: use learned convolutions for up/downsampling if True
    :param dims: dimensionality of input (1D, 2D, or 3D)
    :param num_classes: optional class conditioning (int)
    :param use_checkpoint: enable gradient checkpointing to reduce memory
    :param num_heads: number of attention heads
    :param num_head_channels: fixed channel width per attention head (overrides num_heads)
    :param use_scale_shift_norm: use FiLM-like conditioning
    :param resblock_updown: use residual blocks for up/downsampling
    :param use_new_attention_order: use optimized attention pattern

    Treatment/Temporal Parameters:
    :param image_size: input image dimensions
    :param use_fp16: use float16 precision if True
    :param num_heads_upsample: separate head count for upsampling (deprecated)

    Forward Pass Inputs:
    :param x: input tensor [N x C x ...]
    :param timesteps: diffusion timesteps [N]
    :param intv_t: list of interval days between reference/target exams
    :param treat_code: encoded treatment labels [N]
    :param i_tg: target indices [N] (default: -1)
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        # num_intv_time=3, # number of intervals of treatment days against the target days
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        
        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels * 2),
            nn.SiLU(),
            linear(model_channels * 2, model_channels),
        )
        
        
        self.days_embed = nn.Sequential(
                # FourierFeatures(1, model_channels),
                linear(model_channels, model_channels * 2),
                # nn.LayerNorm(intv_embed_dim),
                nn.SiLU(), 
                linear(model_channels * 2, model_channels),
            )
        
        self.treats_embed = nn.Sequential(
                FourierFeatures(1, model_channels),
                linear(model_channels, model_channels * 2),
                # nn.LayerNorm(intv_embed_dim),
                nn.SiLU(), 
                linear(model_channels * 2, model_channels),
            )
        
        # self.ff_emb = FourierFeatures(1, model_channels)
        
        # update the time_embed_dim with concatenated num of intv_embed_dim
        all_time_day_dim = model_channels * 4
        
        # self.layer_norm = nn.LayerNorm(all_time_day_dim)
        
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        all_time_day_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            all_time_day_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

            
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                model_channels,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                model_channels,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        all_time_day_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            all_time_day_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """Convert the torso of the model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the torso of the model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, intv_t=None, treat_code=None, i_tg = None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param intv_t: a batch of list containing inteval days of the refrence exam wrt the target exam.
        :param treat_code: an [N] Tensor of encoded treatment labels, if treatments-conditional, 
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        hs = []
        b = x.shape[0]
        
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if emb.shape[0] == 1 and b > 1:
            emb = emb.repeat(b, 1).contiguous()
            
        if i_tg is None:
            i_tg = -th.ones(size=(b,), dtype=th.int8, device=x.device)  # -1
            
        d_embs = [timestep_embedding(days, self.model_channels) for days in intv_t] 
        days_feat = [self.days_embed(d) for d in d_embs]

        # treat_embs = [timestep_embedding((t+1)*10, self.model_channels) for t in treat_code]
        treat_feat = [self.treats_embed((t[:, None] + 1)*10) for t in treat_code]
        
        treat_day_sum = th.cat([(t + d).unsqueeze(1)  for t, d in zip(treat_feat, days_feat)], dim=1) # b, s, dim
        
        # target =  th.cat([treat_day_sum[[i], j, :] for i, j in zip(range(b), i_tg)], dim=0) # b, dim
        
        if b == 1:
            target = treat_day_sum[:, i_tg.long(), :]
        else:
            target =  th.cat([treat_day_sum[[i], j, :] for i, j in zip(range(b), i_tg)], dim=0) # b, dim
        
        treat_day_diff = treat_day_sum - target[:, None, :]
        
        middle_emb = emb + target
        
        for i, j in zip(range(b), i_tg):
            treat_day_diff[i, j, :] = treat_day_diff[i, j, :] + middle_emb[i]
    
        treat_day_diff = treat_day_diff.view(b, -1)
        
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, treat_day_diff)
            hs.append(h)
        # h_treat = expand_to_planes(self.middle_treat_enc(treat_stack), h.shape)
        # h = th.cat([h, h_treat], dim=1)
        h = self.middle_block(h, middle_emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, treat_day_diff)
        h = h.type(x.dtype)
        return self.out(h)

