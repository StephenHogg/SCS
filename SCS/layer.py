import torch
import torch.nn as nn


class SCS(nn.Module):
    def __init__(self, kernel_size=2, units=3, *args, **kwargs):
        super(SCS, self).__init__(*args, **kwargs)
        self.q = nn.Parameter(torch.Tensor(1))
        self.p = nn.Parameter(torch.Tensor(1))
        self.w = nn.Parameter(torch.Tensor(1, kernel_size, units))
        self.b = nn.Parameter(torch.Tensor(1, kernel_size, units))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.p)

    def sharpened_cosine_sim(self, s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Function to perform sharpened cosine similarity as per https://twitter.com/_brohrer_/status/1232460132455305218

        Args:
            s (torch.Tensor): Signal Tensor
            k (torch.Tensor): Kernel Tensor

        Returns:
            torch.Tensor: Computed similarity score
        """
        q = self.q.exp()
        p = self.p.exp()
        s_dot_k = torch.dot(s, k)
        norm_s_q = torch.norm(s) + q
        norm_k_q = torch.norm(k) + q
        sim_p = ((s_dot_k) / (norm_s_q * norm_k_q)) ** p
        return torch.sign(s_dot_k) * sim_p
