import logging

import hydra
import torch
from torch.autograd import grad
from typing import Dict, Optional, List

from schnetpack import structure
from schnetpack.model.base import AtomisticModel

import schnetpack as spk

log = logging.getLogger(__name__)

__all__ = ["PESModel"]


class PESModel(AtomisticModel):
    """
    AtomisticModel for potential energy surfaces
    """

    def build_model(self, datamodule: spk.data.AtomsDataModule):
        self.representation = hydra.utils.instantiate(self._representation_cfg)

        self.props = {}
        if "energy" in self._output_cfg:
            self.props["energy"] = self._output_cfg.energy.property

        self.predict_forces = "forces" in self._output_cfg
        if self.predict_forces:
            self.props["forces"] = self._output_cfg.forces.property

        self.predict_stress = "stress" in self._output_cfg
        if self.predict_stress:
            self.props["stress"] = self._output_cfg.stress.property

        self.output = hydra.utils.instantiate(self._output_cfg.module)

        self.losses = {}
        for prop in ["energy", "forces", "stress"]:
            try:
                loss_fn = hydra.utils.instantiate(self._output_cfg[prop].loss)
                loss_weight = self._output_cfg[prop].loss_weight
                self.losses[prop] = (loss_fn, loss_weight)
            except Exception as e:
                print(e)

        self._collect_metrics()

    def _collect_metrics(self):
        self.metrics = {}
        if "metrics" in self._output_cfg.energy:
            energy_metrics = {
                "energy_" + name: hydra.utils.instantiate(metric)
                for name, metric in self._output_cfg.energy.metrics.items()
            }
            self.energy_metrics = torch.nn.ModuleDict(energy_metrics)
            self.metrics["energy"] = energy_metrics

        if self.predict_forces:
            force_metrics = {
                "force_" + name: hydra.utils.instantiate(metric)
                for name, metric in self._output_cfg.forces.metrics.items()
            }
            self.force_metrics = torch.nn.ModuleDict(force_metrics)
            self.metrics["forces"] = force_metrics
        if self.predict_stress:
            stress_metrics = {
                "stress_" + name: hydra.utils.instantiate(metric)
                for name, metric in self._output_cfg.stress.metrics.items()
            }
            self.stress_metrics = torch.nn.ModuleDict(stress_metrics)
            self.metrics["stress"] = stress_metrics

    def forward(self, inputs: Dict[str, torch.Tensor]):
        R = inputs[structure.R]
        inputs[structure.Rij].requires_grad_()
        x = self.representation(inputs)
        inputs.update(x)
        Epred = self.output(inputs)
        results = {"energy": Epred}

        if self.predict_forces or self.predict_stress:
            go: List[Optional[torch.Tensor]] = [torch.ones_like(Epred)]
            dEdRij = grad(
                [Epred],
                [inputs[structure.Rij]],
                grad_outputs=go,
                create_graph=self.training,
            )[0]

            # TorchScript needs Tensor instead of Optional[Tensor]
            if dEdRij is None:
                dEdRij = torch.zeros_like(inputs[structure.Rij])

            if self.predict_forces and dEdRij is not None:
                Fpred = torch.zeros_like(R)
                Fpred = Fpred.index_add(0, inputs[structure.idx_i], dEdRij)
                Fpred = Fpred.index_add(0, inputs[structure.idx_j], -dEdRij)
                results["forces"] = Fpred

            if self.predict_stress:
                stress_i = torch.zeros(
                    (R.shape[0], 3, 3), dtype=R.dtype, device=R.device
                )

                # sum over j
                stress_i = stress_i.index_add(
                    0,
                    inputs[structure.idx_i],
                    dEdRij[:, None, :] * inputs[structure.Rij][:, :, None],
                )

                # sum over i
                idx_m = inputs[structure.idx_m]
                maxm = int(idx_m[-1]) + 1
                stress = torch.zeros(
                    (maxm, 3, 3), dtype=stress_i.dtype, device=stress_i.device
                )
                stress = stress.index_add(0, idx_m, stress_i)

                # TODO: remove reshapes?
                cell_33 = inputs[structure.cell].view(maxm, 3, 3)
                volume = (
                    torch.sum(
                        cell_33[:, 0, :]
                        * torch.cross(cell_33[:, 1, :], cell_33[:, 2, :], dim=1),
                        dim=1,
                        keepdim=True,
                    )
                    .expand(maxm, 3)
                    .reshape(maxm * 3, 1)
                )
                results["stress"] = stress.reshape(maxm * 3, 3) / volume

        results = self.postprocess(inputs, results)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for k, v in self.losses.items():
            fn, weight = v
            loss_p = weight * torch.mean((pred[k] - batch[self.props[k]]) ** 2)
            loss += loss_p
        return loss

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        for prop, pmetrics in self.metrics.items():
            for name, pmetric in pmetrics.items():
                self.log(
                    f"train_{name}",
                    pmetric(pred[prop], batch[self.props[prop]]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        for prop, pmetrics in self.metrics.items():
            for name, pmetric in pmetrics.items():
                self.log(
                    f"val_{name}",
                    pmetric(pred[prop], batch[self.props[prop]]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        pred = self(batch)
        loss = self.loss_fn(pred, batch)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        for prop, pmetrics in self.metrics.items():
            for name, pmetric in pmetrics.items():
                self.log(
                    f"test_{name}",
                    pmetric(pred[prop], batch[self.props[prop]]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self._optimizer_cfg, params=self.parameters()
        )
        schedule = hydra.utils.instantiate(
            self._schedule_cfg.scheduler, optimizer=optimizer
        )
        optimconf = {"scheduler": schedule, "name": "lr_schedule"}
        if self._schedule_cfg.monitor:
            optimconf["monitor"] = self._schedule_cfg.monitor
        return [optimizer], [optimconf]

    # TODO: add eval mode post-processing
