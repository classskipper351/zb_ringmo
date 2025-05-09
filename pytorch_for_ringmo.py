from types import NoneType
import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.checkpoint as checkpoint
#from timm.models.layers import DropPath, to_2tuple, 
from torch.nn.init import trunc_normal_
import collections.abc
from itertools import repeat
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from megatron.core import mpu

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)

_activation = {
    'softmax': nn.Softmax,
    'logsoftmax': nn.LogSoftmax,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'prelu': nn.PReLU,
    'leakyrelu': nn.LeakyReLU,
    'logsigmoid': nn.LogSigmoid,
    'softshrink': nn.Softshrink,
}



    
class RelativePositionBiasForSwin(nn.Module):
    """relative position bias for swin"""
    def __init__(self, window_size, num_heads):
        super(RelativePositionBiasForSwin, self).__init__()
        self.window_size = window_size
        # cls to token & token to cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias
'''
class WindowAttention(nn.Module):

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 compute_dtype=torch.float16,
                 
                 softmax_compute_type=torch.float32
                 ):
        super(WindowAttention, self).__init__()
        if isinstance(dim, tuple) and len(dim) == 1:
            dim = dim[0]

        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        #import pdb;pdb.set_trace()
        self.scale_factor = torch.tensor(qk_scale or head_dim ** -0.5,dtype=torch.float32)
        self.relative_position_bias = RelativePositionBiasForSwin(self.window_size, num_heads)

        # get pair-wise relative position index for each token inside the window
        self.q = nn.Linear(
            in_features =dim, out_features=dim, bias=qkv_bias
            )
        

        self.k = nn.Linear(
            in_features=dim, out_features=dim, bias=qkv_bias
            )
        
        self.v = nn.Linear(
            in_features=dim, out_features=dim, bias=qkv_bias
            )
        
        self.attn_drop = nn.Dropout(p= attn_drop)

        self.proj = nn.Linear(
            in_features=dim, out_features=dim, bias=True
            )
      

        self.proj_drop = nn.Dropout(p= proj_drop)
        

        self.softmax = nn.Softmax()
        
        self.softmax_3d = nn.Softmax()
        
        self.reshape = torch.reshape
        
        self.compute_type = compute_dtype
        self.softmax_dtype = softmax_compute_type
        self.matmul = torch.mul
        self.mul = torch.mul
    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """
        attention_probs = self.softmax(attention_scores)
    
        return attention_probs

    def forward(self, x, mask = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, seq, c = x.shape
        ori_type = x.dtype
        x.to(self.compute_type)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        #q = self.transpose(self.reshape(q, (b, seq, self.num_heads, c // self.num_heads)), (0, 2, 1, 3))
        q=self.reshape(q, (b, seq, self.num_heads, c // self.num_heads))
        q.permute((0, 2, 1, 3))
        #k = self.transpose(self.reshape(k, (b, seq, self.num_heads, c // self.num_heads)), (0, 2, 3, 1))
        k=self.reshape(k, (b, seq, self.num_heads, c // self.num_heads))
        k.permute((0, 2, 3, 1))
        #v = self.transpose(self.reshape(v, (b, seq, self.num_heads, c // self.num_heads)), (0, 2, 1, 3))
        v=self.reshape(v, (b, seq, self.num_heads, c // self.num_heads))
        v.permute((0, 2, 1, 3))

        self.scale_factor.to(self.compute_type)
        q = self.mul(q, self.scale_factor)
        attn = self.matmul(q, k)

        #attn = self.cast(attn, ori_type)
        attn.to(ori_type)
        import pdb;pdb.set_trace()
        attn = torch.add(attn, self.relative_position_bias())

        if mask is not None:
            nw, ws2, _ = mask.shape
            # mask: [64, 49, 49] ==> [1, 64, 1, 49, 49]
            mask = self.reshape(mask, (1, nw, 1, ws2, ws2))
            attn = self.reshape(attn, (b // nw, nw, self.num_heads, seq, seq))
            attn = torch.add(attn, mask)
            attn = self.reshape(attn, (-1, self.num_heads, seq, seq))
            attn = self._softmax(attn)
        else:
            attn = self._softmax(attn)
        attn.to(self.compute_type)
        attn = self.attn_drop(attn)

        x = self.matmul(attn, v)
        x.permute(0, 2, 1, 3)
        x = self.reshape(x, (b, seq, c))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
'''

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    #import pdb;pdb.set_trace()
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        #import pdb;pdb.set_trace()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            #self.attn_mask = nn.Parameter(attn_mask, requires_grad=False)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        #import pdb;pdb.set_trace()

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        #import pdb;pdb.set_trace()
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:

            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                #import pdb;pdb.set_trace()
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 192.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=192, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        #import pdb;pdb.set_trace()
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

class SwinTransformer(MegatronModule):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 192
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """


    def __init__(self,pre_process ,post_process, config:TransformerConfig,img_size=192, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,total_layers=4, **kwargs):
        super().__init__(config)
        self.pre_process = pre_process
        self.post_process = post_process
        layers_per_GPU =total_layers// mpu.get_pipeline_model_parallel_world_size()
        num_completed_layers = mpu.get_pipeline_model_parallel_rank() * layers_per_GPU
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (4- 1))
        self.mlp_ratio = mlp_ratio
        self.in_chans = in_chans
        self.patch_size = patch_size
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution


        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.final_seq = num_patches // 4**num_completed_layers  # downsample seq_length
         #to be imporved
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** (i_layer+num_completed_layers)),
                               input_resolution=(patches_resolution[0] // (2 **(i_layer+num_completed_layers)),
                                                 patches_resolution[1] // (2 ** (i_layer+num_completed_layers))),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) or ( not post_process )else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            if i_layer < self.num_layers - 1:
                self.final_seq = self.final_seq // 4
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    


class SwinTransformerForRingMo(SwinTransformer):

    """swim transformer for ringmo"""
    def __init__(self, 
                 pre_process,
                 post_process,
                 **kwargs):
        
        self.config =  TransformerConfig(
            pipeline_model_parallel_size=kwargs['pipeline_model_parallel_size'],
            pipeline_dtype = kwargs['pipeline_dtype'],
            num_attention_heads=1,
            num_layers=1,
            hidden_size=1,
            variable_seq_lengths= True, # therefore three params above is useless
            batch_p2p_sync=False ,
            deallocate_pipeline_outputs=True,
            batch_p2p_comm=False
            )
        
        super(SwinTransformerForRingMo, self).__init__(pre_process,post_process,self.config,**kwargs)
        assert self.num_classes == 0
        
        


        self.use_lbp = None
        self.pre_process = pre_process
        self.post_process = post_process
        self.reshape = torch.reshape
        #self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.add_pos = torch.add
        self.sub = torch.sub
        self.multi = torch.mul
        self.hw = int(self.final_seq ** 0.5)
        self.encoder_stride = 32
        self.input_tensor = None
        self.x_in = None
        self.share_embeddings_and_output_weights = False


    def set_input_tensor(self, input_tensor : torch.Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def ringmo_loss(self, x, x_rec, lbp=None, lbp_rec=None, mask=None):
        """ringmo loss"""
        # 计算原始重建损失
        loss_ori_recon = self.abs(self.sub(x, x_rec))  # [B,C,H,W]
        loss_ori_mask = self._masked_mean(loss_ori_recon, mask)
        
        # 计算LBP重建损失（如果启用）
        loss_lbp_mask = 0.
        if self.use_lbp and lbp is not None:
            loss_lbp_recon = self.abs(self.sub(lbp, lbp_rec))
            loss_lbp_mask = self._masked_mean(loss_lbp_recon, mask)
            
        return loss_ori_mask + loss_lbp_mask

    def _masked_mean(self, loss, mask):
        """带掩码的加权平均"""
        # loss: [B,C,H,W], mask: [B,1,H,W]
        weighted_loss = self.mul(loss, mask)
        sum_loss = self.sum(weighted_loss)
        sum_mask = self.sum(mask) + 1e-5  # 防止除零
        return sum_loss / sum_mask / self.in_chans  # 按通道数归一化

    def _check_input(self, inputs):
        #import pdb ;pdb.set_trace()
        if not self.use_lbp:
            return inputs[0], None, inputs[1]

        return inputs[0], inputs[1], inputs[2]

    def forward(self, inputs:Union[tuple , NoneType]):
        """construct of SwinTransformerForRingMo"""

        #import pdb;pdb.set_trace()
         # 从流水线前级获取的隐藏状态
        x_rec = None 
        x_in = None
        if self.pre_process:
            assert not isinstance(inputs,NoneType) 
            x_in, lbp_in, mask_in = self._check_input(inputs)

            x = self.multi(x_in, self.sub(1, mask_in))
            x = self.patch_embed(x)

            if self.ape:
                x = self.add_pos(x, self.absolute_pos_embed)

            x = self.pos_drop(x)
        if not self.pre_process and self.input_tensor is not None:
            x = self.input_tensor[0] if isinstance(self.input_tensor,List) else self.input_tensor# x_intermediate
            #import pdb;pdb.set_trace()
        for layer in self.layers:
            x = layer(x)

        if self.post_process:

            x = self.norm(x)
            #import pdb;pdb.set_trace()
            x=x.permute(0, 2, 1)
            #import pdb;pdb.set_trace()
            z = self.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))
            x_rec = self.decoder(z)
            x_rec = self.pixelshuffle(x_rec)
            '''
            lbp_rec = None # to be improved
            if lbp_in is not None:
                lbp_rec = self.decoder_lbp(z)
                lbp_rec = self.pixelshuffle(lbp_rec)
            '''
        
        return x_rec if self.post_process else x


    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}           


class L1Loss(nn.Module): # why do loss use module?  abandoned since 2025.3.17
    """L1 loss"""
    def __init__(self, reduction='mean', parallel_config=None):
        super(L1Loss, self).__init__()



        self.abs = torch.abs
        self.sub = torch.sub

        self.mul = torch.mul
        self.reduce_mean = torch.mean
        self.reduce_sum = torch.sum
        #self.cast = P.Cast()

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False


    def get_axis(self, x):
        shape = x.shape
        length = len(shape)
        perm = range(length)
        return perm

    def get_loss(self, x, weights=1.0):
        """get loss of L1loss"""
        input_dtype = x.dtype
        x.to(torch.float32)
        #weights.to(torch.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x.to(input_dtype)
        return x

    def forward(self, logits, labels):
        """construct of L1loss"""
        x_sub = self.sub(logits, labels)
        x = self.abs(x_sub)
        x = self.get_loss(x)
        return x
    
class RingMo(nn.Module):
    """RingMo"""
    def __init__(self, encoder, encoder_stride, use_lbp=False, parallel_config=None):
        super(RingMo, self).__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.use_lbp = use_lbp

        self.decoder = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 3,
            kernel_size=1, bias=True
        )

        # encoder output -> [B,C,H,W]
        self.decoder_lbp = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 3,
            kernel_size=1, bias=True
        )

        # encoder output -> [B,C,H,W]


        self.pixelshuffle = nn.PixelShuffle(self.encoder_stride)
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        #self.l1_loss = L1Loss(reduction='none',)

        self.expand_dim = torch.unsqueeze
        
        self.div = torch.div
        self.multi = torch.mul

        self.sum = torch.sum
        self.add = torch.add

    def ringmo_loss(self, x, x_rec, lbp=None, lbp_rec=None, mask=None):
        """ringmo loss"""
        # 计算原始重建损失
        loss_ori_recon = self.abs(self.sub(x, x_rec))  # [B,C,H,W]
        loss_ori_mask = self._masked_mean(loss_ori_recon, mask)
        
        # 计算LBP重建损失（如果启用）
        loss_lbp_mask = 0.
        if self.use_lbp and lbp is not None:
            loss_lbp_recon = self.abs(self.sub(lbp, lbp_rec))
            loss_lbp_mask = self._masked_mean(loss_lbp_recon, mask)
            
        return loss_ori_mask + loss_lbp_mask

    def _masked_mean(self, loss, mask):
        """带掩码的加权平均"""
        # loss: [B,C,H,W], mask: [B,1,H,W]
        weighted_loss = self.mul(loss, mask)
        sum_loss = self.sum(weighted_loss)
        sum_mask = self.sum(mask) + 1e-5  # 防止除零
        return sum_loss / sum_mask / self.in_chans  # 按通道数归一化

    def _check_input(self, inputs):
        #import pdb ;pdb.set_trace()
        if not self.use_lbp:
            return inputs[0], None, inputs[1]

        return inputs[0], inputs[1], inputs[2]

    def forward(self, *inputs):
        """construct of RingMo"""
        #import pdb;pdb.set_trace()
        x_in, lbp_in, mask_in = self._check_input(inputs)

        # x -> [B,L,C]
        z = self.encoder(x_in, mask_in)
        # z -> [B,C,H,W]
        x_rec = self.decoder(z)
        # self.summary_4d("decoder_conv2d", self.decoder.weight)
        # z -> [B,C,H,W]
        x_rec = self.pixelshuffle(x_rec)

        lbp_rec = None
        if lbp_in is not None:
            lbp_rec = self.decoder_lbp(z)
            lbp_rec = self.pixelshuffle(lbp_rec)

        sim_loss = self.ringmo_loss(x_in, x_rec, lbp_in, lbp_rec, mask_in)

        return sim_loss

    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_ringmo(config):
    """build ringmo"""
    model_type = config.model.backbone
    if model_type == 'swin':
        encoder = SwinTransformerForRingMo(
            parallel_config=config.parallel_config,
            moe_config=config.moe_config,
            batch_size=config.train_config.batch_size * config.device_num
            if config.parallel.parallel_mode == "semi_auto_parallel" else config.train_config.batch_size,
            img_size=config.train_config.image_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.in_chans,
            num_classes=0,
            embed_dim=config.model.embed_dim,
            depths=config.model.depth,
            num_heads=config.model.num_heads,
            window_size=config.model.window_size,
            mlp_ratio=config.model.mlp_ratio,
            qkv_bias=config.model.qkv_bias,
            qk_scale=config.model.qk_scale,
            drop_rate=config.model.drop_rate,
            drop_path_rate=config.model.drop_path_rate,
            ape=config.model.ape,
            patch_norm=config.model.patch_norm,
            patch_type=config.model.patch_type)
        encoder_stride = 32
    
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = RingMo(encoder=encoder, encoder_stride=encoder_stride, 
                   use_lbp=config.model.use_lbp)
    #import pdb; pdb.set_trace()
    return model

