"""
Geometrical embeddings from the atomic positions and its neighbors.
"""
from typing import Dict, Sequence

import torch
from torch import nn

from src.schnetpack import properties


class OneHotEmbed(nn.Module):
    def __init__(self, prop_keys: Dict, atomic_types: Sequence[int], module_name: str = 'one_hot_embed',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prop_keys = prop_keys
        self.atomic_types = atomic_types
        self.module_name = module_name
        self.to_one_hot = lambda x: to_onehot(x, node_types=self.atomic_types)
        self.atomic_type_key = self.prop_keys[properties.energy]

    def forward(self, inputs: Dict, *args, **kwargs):
        z = inputs[self.atomic_type_key]
        return {'z_one_hot': self.to_one_hot(z)}

    def __dict_repr__(self):
        return {self.module_name: {'atomic_types': self.atomic_types,
                                   'prop_keys': self.prop_keys}
                }

    def reset_input_convention(self, input_convention: str) -> None:
        pass


def to_onehot(features: torch.Tensor, node_types: Sequence):
    r"""
    Create onehot encoded vectors from :data:`features`.

    Args:
        features (Array): type of the nodes, shape: (n)
        node_types (Sequence[int]): list of possible node types.
    """
    ones = []
    for i, e in enumerate(node_types):
        ones.append(torch.where(features == e, torch.ones(1), torch.zeros(1))[..., None])
    return torch.concatenate(ones, dim=-1)
