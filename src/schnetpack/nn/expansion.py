from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from schnetpack.nn import Dense

__all__ = ["SphcBasisExpansion"]

class SphcBasisExpansion(nn.Module):
    
    def __init__(self, sphc_basis_expansion_fn: nn.Module, sphc_cutoff_fn: nn.Module, n_rbf: int):
        super(SphcBasisExpansion, self).__init__()
        self.sphc_basis_expansion_fn = sphc_basis_expansion_fn
        self.sphc_cutoff_fn = sphc_cutoff_fn
        self.reduce_fn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(8 * (n_rbf // 4), 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, inputs: torch.Tensor):
        m_cut_ij = self.sphc_cutoff_fn(inputs)
        output = torch.zeros_like(inputs)
        for col_idx in range(inputs.shape[1]):
            # get the corresponding column of chi
            chi_l = inputs[:, col_idx]
            m_cut_l = m_cut_ij[:, col_idx].unsqueeze(1)
            exp_chi_l = self.sphc_basis_expansion_fn(chi_l) # (n, #rbfs) 
            exp_chi_l = torch.where(m_cut_l != 0, exp_chi_l * m_cut_l, 0) # shape: (n_pairs,n_rbfs)
            exp_chi_l = exp_chi_l.unsqueeze(1) # 1 in channel
            reduced_chi_l = self.reduce_fn(exp_chi_l)
            output[:, col_idx] = reduced_chi_l.squeeze()
        return output