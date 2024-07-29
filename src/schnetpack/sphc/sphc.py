from typing import Dict, Sequence

import torch
from torch import nn

from src.schnetpack import properties
from src.schnetpack.nn.spherical import init_sph_fn


def _init_sphc_zeros(z, sph_ij, *args, **kwargs):
    return {'chi': torch.zeros((z.shape[-1], sph_ij.shape[-1]), dtype=sph_ij.dtype)}


def _init_sphc(z, sph_ij, phi_r_cut, idx_i, mp_normalization, *args, **kwargs):
    lm = lambda x: x * phi_r_cut[:, None]
    _sph_harms_ij = lm(sph_ij)  # shape: (n_pairs,m_tot)
    chi = torch.bincount(idx_i, weights=_sph_harms_ij, minlength=len(z))  # equivalent to segment_sum
    return {'chi': chi / mp_normalization}


class SphericalHarmonics(nn.Module):
    def __init__(self, prop_keys: Dict, atomic_types: Sequence[int], pairwise_distances: torch.Tensor,
                 module_name: str = 'spherical_harmonics',
                 cut_fn: nn.Module = None, cut_fn_args: Dict = None,
                 degrees: Sequence[int] = None,
                 sphc_normalization: float = None,
                 radial_basis_fn: nn.Module = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prop_keys = prop_keys
        self.atomic_types = atomic_types
        self.module_name = module_name
        self.atomic_type_key = self.prop_keys[properties.energy]
        self.cut_fn = cut_fn
        self.pairwise_distances = pairwise_distances  # shape: (n_pairs)
        self.degrees = degrees
        self.sph_fns = [init_sph_fn(y) for y in self.degrees]
        self.radial_basis_fn = radial_basis_fn
        self.sphc_normalization = sphc_normalization

    def forward(self, inputs: Dict, *args, **kwargs):
        idx_i = inputs['idx_i']  # shape: (n_pairs)
        # Calculate pairwise distances
        d_ij = torch.linalg.norm(self.pairwise_distances, axis=-1)  # shape : (n_pairs)

        # Gaussian basis expansion of distances
        rbf_ij = self.radial_basis_fn(d_ij[:, None])  # shape: (n_pairs,K)
        phi_r_cut = self.cut_fn(d_ij)  # shape: (n_pairs)

        # Normalized distance vectors
        fn = lambda y: y / d_ij[:, None]
        # apply lambda
        unit_r_ij = fn(self.pairwise_distances)  # shape: (n_pairs, 3)
        # Spherical harmonics
        sph_harms_ij = []
        for sph_fn in self.sph_fns:
            sph_ij = sph_fn(unit_r_ij)  # shape: (n_pairs,2l+1)
            sph_harms_ij += [sph_ij]  # len: |L| / shape: (n_pairs,2l+1)

        sph_harms_ij = torch.concatenate(sph_harms_ij, dim=-1) if len(self.degrees) > 0 else None
        # shape: (n_pairs,m_tot)

        geometric_data = {
            'sph_ij': sph_harms_ij,
        }

        # Spherical harmonic coordinates (SPHCs)
        if self.sphc:
            z = inputs[self.atomic_type_key]
            if self.sphc_normalization is None:
                # Initialize SPHCs to zero
                geometric_data.update(_init_sphc_zeros(z=z,
                                                       sph_ij=sph_harms_ij)
                                      )
            else:
                # Initialize SPHCs with a neighborhood dependent embedding
                geometric_data.update(_init_sphc(z=z,
                                                 sph_ij=sph_harms_ij,
                                                 phi_r_cut=phi_r_cut,
                                                 idx_i=idx_i,
                                                 mp_normalization=self._lambda)
                                      )
            # Solid harmonics (Spherical harmonics + radial part)

        if self.solid_harmonic:
            lm = lambda x: x * phi_r_cut[:, None]
            rbf_ij = lm(rbf_ij)  # shape: (n_pairs,K)
            g_ij = sph_harms_ij[:, :, None] * rbf_ij[:, None, :]  # shape: (n_pair,m_tot,K)
            geometric_data.update({'g_ij': g_ij})

        return geometric_data
