from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from schnetpack.nn import Dense

__all__ = ["SphcBasisExpansion"]

class SphcBasisExpansion(nn.Module):
    
    def __init__(self, sphc_basis_expansion_fn: nn.Module, sphc_cutoff_fn: nn.Module, n_rbf: int, num_features: Sequence[int]):
        super(SphcBasisExpansion, self).__init__()
        self.sphc_basis_expansion_fn = sphc_basis_expansion_fn
        self.sphc_cutoff_fn = sphc_cutoff_fn
        self.num_features = num_features
        l1 = Dense(in_features=n_rbf,activation=F.silu,out_features=self.num_features[0])
        l2 = Dense(in_features=self.num_features[0], activation=F.silu, out_features=self.num_features[1])
        l3 = Dense(in_features=self.num_features[1],activation=None,out_features=1)
        self.reduce_fn = nn.Sequential(*[l1,l2,l3])
        
        
    def forward(self, inputs: torch.Tensor):
        m_cut_ij = self.sphc_cutoff_fn(inputs)
        output = torch.zeros_like(inputs)
        for col_idx in range(inputs.shape[1]):
            # get the corresponding column of chi
            chi_l = inputs[:, col_idx]
            m_cut_l = m_cut_ij[:, col_idx].unsqueeze(1)
            exp_chi_l = self.sphc_basis_expansion_fn(chi_l) # (n, #rbfs) 
            exp_chi_l = torch.where(m_cut_l != 0, exp_chi_l * m_cut_l, 0) # shape: (n_pairs,n_rbfs)
            reduced_chi_l = self.reduce_fn(exp_chi_l)
            output[:, col_idx] = reduced_chi_l.squeeze()
        return output