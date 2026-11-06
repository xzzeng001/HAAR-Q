"""
This file implements a variational Monte Carlo method for solving quantum many-body problems.
"""

import logging
import typing
import dataclasses
import omegaconf
import torch
import torch.utils.tensorboard
from ..utility.context import RuntimeContext
from ..utility.action_dict import action_dict


def _projected_energy_loss(psi: torch.Tensor, hpsi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected-space energy and a surrogate with its exact gradient.

    The Hamiltonian must be Hermitian, and ``hpsi`` must be its action on
    ``psi`` in the same fixed configuration basis. Native Hamiltonian kernels
    need not support autograd: the detached residual supplies both terms of
    the Rayleigh-quotient derivative, including the factor of two.
    """
    with torch.no_grad():
        state = psi.detach()
        norm = torch.vdot(state, state).real
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("The wavefunction must have a finite, positive norm on the selected configurations.")
        energy = torch.vdot(state, hpsi.detach()).real / norm
        residual = (hpsi.detach() - energy * state) / norm
    loss = 2 * torch.vdot(psi, residual).real
    return energy, loss


@dataclasses.dataclass
class VmcConfig:
    """
    The VMC optimization for solving quantum many-body problems.
    """

    # The sampling count
    sampling_count: int = 4000
    # Target total number of configurations after Hamiltonian-guided expansion
    relative_count: int = 40000
    # The number of steps for the local optimizer
    local_step: int = 1000
    # Generate configuration uniquely
    unique: bool = True
    # Optimize on the entire expanded support; False restores the legacy objective
    hamiltonian_guided: bool = True

    def __post_init__(self) -> None:
        if self.sampling_count <= 0:
            raise ValueError("sampling_count must be positive.")
        if self.relative_count < 0:
            raise ValueError("relative_count must be nonnegative.")
        if self.local_step <= 0:
            raise ValueError("local_step must be positive.")

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """
        The main function for the VMC optimization.
        """

        # Create Model and Network
        model = context.create_model(runtime_config.model)
        network = context.create_network(runtime_config.network, model, checkpoint_data.get("network"))
        data = checkpoint_data

        # Create Optimizer
        optimizer = context.create_optimizer(
            runtime_config.optimizer, network.parameters(), checkpoint_data.get("optimizer")
        )

        logging.info(
            "Arguments Summary: Sampling Count: %d, Relative Count: %d, Local Steps: %d, Unique: %s, "
            "Hamiltonian Guided: %s",
            self.sampling_count,
            self.relative_count,
            self.local_step,
            self.unique,
            self.hamiltonian_guided,
        )

        if "vmc" not in data:
            data["vmc"] = {"global": 0, "local": 0}

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=context.folder())  # type: ignore[no-untyped-call]

        while True:
            logging.info("Starting a new optimization cycle")

            logging.info("Sampling configurations")
            reweight = None
            if self.unique:
                configs_i, psi_i, _, _ = network.generate_unique(self.sampling_count)
                if not self.hamiltonian_guided:
                    reweight = torch.ones_like(psi_i, dtype=torch.float64)
            else:
                configs_i, psi_i, count, _ = network.generate(self.sampling_count)
                if not self.hamiltonian_guided:
                    reweight = count / psi_i.abs() ** 2
            logging.info("Sampling completed, unique configurations count: %d", len(configs_i))

            logging.info("Calculating relative configurations")
            if self.relative_count <= len(configs_i):
                configs_dst = configs_i
            else:
                configs_dst = torch.cat(
                    [
                        configs_i,
                        model.find_relative(
                            configs_i, psi_i, self.relative_count - len(configs_i), configs_exclude=configs_i
                        ),
                    ]
                )
            if self.hamiltonian_guided:
                # Each selected configuration is one basis state, regardless of
                # sampling multiplicity. New neighbors participate in gradients.
                configs_dst = torch.unique(configs_dst, dim=0)
                configs_src = configs_dst
            else:
                configs_src = configs_i
            logging.info("Relative configurations calculated, count: %d", len(configs_dst))
            writer.add_scalar("vmc/configs/sampled", len(configs_i), data["vmc"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("vmc/configs/expanded", len(configs_dst), data["vmc"]["global"])  # type: ignore[no-untyped-call]

            @torch.enable_grad()
            def closure() -> torch.Tensor:
                # Optimizing energy
                optimizer.zero_grad()
                psi_src = network(configs_src)
                if self.hamiltonian_guided:
                    with torch.no_grad():
                        hpsi = model.apply_within(configs_src, psi_src.detach(), configs_src)
                    energy, loss = _projected_energy_loss(psi_src, hpsi)
                    loss.backward()  # type: ignore[no-untyped-call]
                    return energy

                assert reweight is not None
                with torch.no_grad():
                    psi_dst = network(configs_dst)
                    hamiltonian_psi_dst = model.apply_within(configs_dst, psi_dst, configs_src)
                psi_src_reweight = psi_src.conj() * reweight
                num = psi_src_reweight @ hamiltonian_psi_dst
                den = psi_src_reweight @ psi_src.detach()
                energy = num / den
                energy = energy.real
                energy.backward()  # type: ignore[no-untyped-call]
                return energy

            logging.info("Starting local optimization process")

            for i in range(self.local_step):
                energy: torch.Tensor = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                logging.info(
                    "Local optimization in progress, step: %d, energy: %.10f, ref energy: %.10f",
                    i,
                    energy.item(),
                    model.ref_energy,
                )
                writer.add_scalar("vmc/energy", energy, data["vmc"]["local"])  # type: ignore[no-untyped-call]
                writer.add_scalar("vmc/error", energy - model.ref_energy, data["vmc"]["local"])  # type: ignore[no-untyped-call]
                data["vmc"]["local"] += 1

            logging.info("Local optimization process completed")

            writer.flush()  # type: ignore[no-untyped-call]

            logging.info("Saving model checkpoint")
            data["vmc"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            context.save(data, data["vmc"]["global"])
            logging.info("Checkpoint successfully saved")

            logging.info("Current optimization cycle completed")


action_dict["vmc"] = VmcConfig
