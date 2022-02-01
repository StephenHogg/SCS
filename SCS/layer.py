import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SCS(nn.Module):
    def __init__(
        self, channels_in: int, channels_out: int, kernel: torch.Size, *args, **kwargs
    ):
        super(SCS, self).__init__(*args, **kwargs)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_shape = kernel
        self.stride = 1
        self.dilation = 1
        self.q = nn.Parameter(torch.Tensor(1))
        self.p = nn.Parameter(torch.Tensor(1))
        self.w = nn.Parameter(torch.Tensor(kernel.numel() * channels_in, channels_out))
        self.b = nn.Parameter(torch.Tensor(channels_out))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.p)
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.b)

    def sharpened_cosine_sim(self, s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Function to perform sharpened cosine similarity as per https://twitter.com/_brohrer_/status/1232460132455305218

        Args:
            s (torch.Tensor): Signal Tensor. Must be 2D, with the size of the second dimension equal to the number of elements
                in a given patch
            k (torch.Tensor): Kernel Tensor. Must be 2D, with the first dimension size equal to the number of elements in a patch from
                the source array and the second dimension size equal to the number of desired output channels

        Returns:
            torch.Tensor: Computed similarity score
        """
        q = self.q.exp()
        p = self.p.exp()
        s_dot_k = torch.einsum("bx,xc->bc", s, k)
        norm_s_q = torch.norm(s, dim=1) + q
        norm_k_q = torch.norm(k) + q
        norm_base = 1 / (norm_s_q * norm_k_q)
        sim_frac = torch.einsum("bc,b->bc", s_dot_k, norm_base)
        sim_p = sim_frac**p
        return torch.sign(s_dot_k) * sim_p

    def extract_image_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches from an N-dimensional tensor.
        Adapted from https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/6

        Args:
            x (torch.Tensor): Source tensor

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

        # Merge all patches into the channel dimension
        patches = (
            patches.moveaxis(1, -1)
            .flatten(-1 * (len(self.kernel_shape) + 1))
            .moveaxis(-1, 1)
            .contiguous()
        )

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad the array

        # Extract patches

        # Push all other dimensions into the batch dimension

        # Calculate the sharpened cosine similarity

        # Retrieve squashed dimensions
        return x
