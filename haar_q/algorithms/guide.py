"""
This file implements a variational Monte Carlo method for solving quantum many-body problems with guide.
"""

import logging
import typing
import dataclasses
import omegaconf
import torch
import torch.utils.tensorboard
from ..utility.context import RuntimeContext
from ..utility.action_dict import action_dict
from .vmc import expand_seeds, sampled_energy


@dataclasses.dataclass
class GuideConfig:
    """
    The guided VMC optimization for solving quantum many-body problems.
    """

    # The sampling count
    sampling_count: int = 4000
    # The number of relative configurations to be used in energy calculation
    relative_count: int = 40000
    # The number of steps for the local optimizer
    local_step: int = 1000
    # The number of steps for the distribution optimizer
    dist_step: int = 100
    # Generate configuration uniquely
    unique: bool = True

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """
        The main function for the guided VMC optimization.
        """

        # Create Model
        model = context.create_model(runtime_config.model)

        # Create Main Network
        network = context.create_network(runtime_config.network, model, checkpoint_data.get("network"))

        # Create Sampling Network
        sampling = context.create_network(runtime_config.sampling, model, checkpoint_data.get("sampling"))

        data = checkpoint_data

        # Create Optimizers
        # We use the same optimizer configuration for both network and sampling
        optimizer_network = context.create_optimizer(
            runtime_config.optimizer, network.parameters(), checkpoint_data.get("optimizer")
        )
        optimizer_sampling = context.create_optimizer(
            runtime_config.optimizer, sampling.parameters(), checkpoint_data.get("optimizer_sampling")
        )

        logging.info(
            "Arguments Summary: Sampling Count: %d, Relative Count: %d, Local Steps: %d, Dist Steps: %d, Unique: %s",
            self.sampling_count,
            self.relative_count,
            self.local_step,
            self.dist_step,
            self.unique,
        )

        if "guide" not in data:
            data["guide"] = {"global": 0, "local": 0, "dist": 0}

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=context.folder())  # type: ignore[no-untyped-call]

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            if self.unique:
                configs_src_network, psi_src_network, _, _ = network.generate_unique(self.sampling_count)
                reweight_network = torch.ones_like(psi_src_network, dtype=torch.float64)
                configs_src_sampling, psi_src_sampling, _, _ = sampling.generate_unique(self.sampling_count)
                reweight_sampling = torch.ones_like(psi_src_sampling, dtype=torch.float64)
            else:
                configs_src_network, psi_src_network, count_network, _ = network.generate(self.sampling_count)
                reweight_network = count_network / psi_src_network.abs() ** 2
                configs_src_sampling, psi_src_sampling, count_sampling, _ = sampling.generate(self.sampling_count)
                reweight_sampling = count_sampling / psi_src_sampling.abs() ** 2

            logging.info("Calculating relative configurations")
            configs_dst_network = expand_seeds(model, configs_src_network, psi_src_network, self.relative_count)
            configs_dst_sampling = expand_seeds(model, configs_src_sampling, psi_src_sampling, self.relative_count)

            def energy_sampling() -> torch.Tensor:
                optimizer_network.zero_grad()
                psi_src = network(configs_src_sampling)
                with torch.no_grad():
                    psi_dst = network(configs_dst_sampling)
                    hpsi = model.apply_within(configs_dst_sampling, psi_dst, configs_src_sampling)
                energy = sampled_energy(psi_src, hpsi, reweight_sampling).real
                energy.backward()  # type: ignore[no-untyped-call]
                return energy

            @torch.no_grad()
            def evaluate(seeds: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                # expand_seeds retains the seed rows at the start of the support.
                psi = network(support)
                return psi[: len(seeds)], model.apply_within(support, psi, seeds)

            logging.info("Starting local optimization process")

            for i in range(self.local_step):
                sampling_energy: torch.Tensor = optimizer_network.step(energy_sampling)  # type: ignore[assignment,arg-type]
                diagnostic_psi, diagnostic_hpsi = evaluate(configs_src_network, configs_dst_network)
                network_energy = sampled_energy(diagnostic_psi, diagnostic_hpsi, reweight_network).real
                logging.info(
                    "Local optimization in progress, step: %d, energy from sampling: %.10f, energy from network: %.10f, ref energy: %.10f",
                    i,
                    sampling_energy.item(),
                    network_energy.item(),
                    model.ref_energy,
                )
                writer.add_scalar("guide/energy/sampling", sampling_energy, data["guide"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("guide/error/sampling", sampling_energy - model.ref_energy, data["guide"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("guide/energy/network", network_energy, data["guide"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("guide/error/network", network_energy - model.ref_energy, data["guide"]["local"])  # type: ignore[no-untyped-call]
                if self.dist_step > 0:
                    # The main network stays fixed throughout these sampler-only updates,
                    # including repeated LBFGS closure evaluations.
                    psi, hpsi = evaluate(configs_src_sampling, configs_dst_sampling)
                    energy = sampled_energy(psi, hpsi, reweight_sampling)
                    residual_weight = (hpsi - energy * psi).abs().square()
                    total_weight = residual_weight.sum()
                    if not torch.isfinite(total_weight):
                        raise ValueError("The guide sampler target must have a finite residual norm.")
                    if total_weight > 0:
                        target = residual_weight / total_weight

                        def distribution() -> torch.Tensor:
                            optimizer_sampling.zero_grad()
                            probability = sampling(configs_src_sampling).abs().square()
                            loss = (probability / probability.sum() - target).square().sum()
                            loss.backward()  # type: ignore[no-untyped-call]
                            return loss

                        for _ in range(self.dist_step):
                            dist_loss: torch.Tensor = optimizer_sampling.step(distribution)  # type: ignore[assignment,arg-type]
                            writer.add_scalar("guide/dist/loss", dist_loss, data["guide"]["dist"])  # type: ignore[no-untyped-call]
                            data["guide"]["dist"] += 1
                    else:
                        logging.info("Skipping guide sampler updates: the selected residual is zero.")
                data["guide"]["local"] += 1

            logging.info("Local optimization process completed")

            writer.flush()  # type: ignore[no-untyped-call]

            logging.info("Saving model checkpoint")
            data["guide"]["global"] += 1
            data["network"] = network.state_dict()
            data["sampling"] = sampling.state_dict()
            data["optimizer"] = optimizer_network.state_dict()
            data["optimizer_sampling"] = optimizer_sampling.state_dict()
            context.save(data, data["guide"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


action_dict["guide"] = GuideConfig
