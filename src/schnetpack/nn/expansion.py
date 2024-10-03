import torch
import torch.nn as nn

__all__ = ["SphcBasisExpansion"]

class SphcBasisExpansion(nn.Module):
    
    def __init__(self, sphc_basis_expansion_fn: nn.Module, sphc_cutoff_fn: nn.Module):
        super(SphcBasisExpansion, self).__init__()
        self.sphc_basis_expansion_fn = sphc_basis_expansion_fn
        self.sphc_cutoff_fn = sphc_cutoff_fn
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=8)  
        
        
    def forward(self, inputs: torch.Tensor):
        m_cut_ij = self.sphc_cutoff_fn(inputs)
        output = torch.zeros_like(inputs)
        for col_idx in range(inputs.shape[1]):
            # get the corresponding column of chi
            chi_l = inputs[:, col_idx]
            exp_chi_l = self.sphc_basis_expansion_fn(chi_l)
            m_cut_col = m_cut_ij[:, col_idx].unsqueeze(1)  # Shape: (n_mol, 1)
            exp_chi_l = torch.where(m_cut_col != 0, exp_chi_l * m_cut_col, torch.zeros_like(exp_chi_l))
        
            # Reshape to (n_mol, n_rbf, 1) for conv1d input
            exp_chi_l = exp_chi_l.unsqueeze(1)
            
            # Apply 1D convolution to reduce n_rbf to 1
            reduced_chi_l = self.conv1d(exp_chi_l)  # shape: (n_mol, 1, 1)
            reduced_chi_l = torch.sigmoid(reduced_chi_l) # apply non linearity
            reduced_chi_l = reduced_chi_l.squeeze()  # shape: (n_mol)
            
            output[:, col_idx] = reduced_chi_l
        return output
        