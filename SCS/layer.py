from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sigplus(x: torch.Tensor) -> torch.Tensor:
    return F.sigmoid(x) * F.softplus(x)


class SCS(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel: torch.Size,
        epsilon: float = 1e-12,
        *args,
        **kwargs
    ):
        super(SCS, self).__init__(*args, **kwargs)
        self.kernel_shape = kernel
        self.stride = 1
        self.dilation = 1
        self.epsilon = epsilon
        self.q = nn.Parameter(torch.Tensor(1))
        self.p = nn.Parameter(torch.Tensor(1))
        self.w = nn.Parameter(torch.Tensor(kernel.numel() * channels_in, channels_out))
        self.b = nn.Parameter(torch.Tensor(channels_out))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.p, 2)
        nn.init.uniform_(self.q)
        nn.init.uniform_(self.b)
        nn.init.xavier_uniform_(self.w)

    def sharpened_cosine_sim(self, s: torch.Tensor) -> torch.Tensor:
        """Function to perform sharpened cosine similarity as per https://twitter.com/_brohrer_/status/1232460132455305218

        Args:
            s (torch.Tensor): Signal Tensor. Must be 2D, with the size of the second dimension equal to the number of elements
                in a given patch

        Returns:
            torch.Tensor: Computed similarity score
        """
        q = self.q.exp()
        s_dot_k = torch.einsum("bx,xc->bc", s, self.w)
        norm_s_q = torch.clamp(torch.norm(s, dim=1), min=self.epsilon) + q
        norm_k_q = torch.clamp(torch.norm(self.w), min=self.epsilon) + q
        norm_base = 1 / (norm_s_q * norm_k_q)
        sim_frac = torch.einsum("bc,b->bc", s_dot_k, norm_base)
        sign = torch.sign(sim_frac)
        sim_frac_abs = sim_frac.abs() + 1e-12
        sim_p = torch.pow(sim_frac_abs + sigplus(self.b), sigplus(self.p))
        res = sign * sim_p

        return res

    def extract_image_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches from an N-dimensional tensor.
        Adapted from https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/6

        Args:
            x (torch.Tensor): Source tensor. Must contain at least a batch, channel and one other dimension.

        Returns:
            torch.Tensor: Tensor with patches in the first dimension
        """
        # Do TF 'SAME' Padding
        pad_vals = []
        for d, k in zip(x.shape[2:], self.kernel_shape):
            l = math.ceil(d / self.stride)
            pad_len = (l - 1) * self.stride + (k - 1) * self.dilation + 1 - d
            pad_vals = [pad_len // 2, pad_len - pad_len // 2] + pad_vals
        patches = F.pad(x, pad_vals)

        # Extract patches
        for d, k in zip(range(2, x.dim()), self.kernel_shape):
            patches = patches.unfold(d, k, self.stride)

        # Merge all patches into the channel dimension and move that
        # to the end of the tensor. Then flatten the patches and channel
        # dimension together.
        patches = (
            patches.moveaxis(1, -1)
            .flatten(-1 * (len(self.kernel_shape) + 1))
            .contiguous()
        )

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Extract patches
        patches = self.extract_image_patches(x)

        # Push all dimensions besides patch into the batch dimension.
        # Keep track of what they were before flattening, though.
        patch_dims = patches.shape
        flat_patches = torch.flatten(patches, end_dim=-2)

        # Calculate the sharpened cosine similarity
        sim = self.sharpened_cosine_sim(flat_patches) + self.b

        # Retrieve squashed dimensions
        res = sim.reshape(*patch_dims[:-1], -1).moveaxis(-1, 1)

        return res


class AbsPool(nn.Module):
    def __init__(self, pooling_module, *args, **kwargs):
        super(AbsPool, self).__init__()
        self.pooling_layer = pooling_module(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_pool = self.pooling_layer(x)
        neg_pool = self.pooling_layer(-x)
        abs_pool = torch.where(pos_pool >= neg_pool, pos_pool, -neg_pool)
        return abs_pool


MaxAbsPool1d = partial(AbsPool, pooling_module=nn.MaxPool1d)
MaxAbsPool2d = partial(AbsPool, pooling_module=nn.MaxPool2d)
MaxAbsPool3d = partial(AbsPool, pooling_module=nn.MaxPool3d)
