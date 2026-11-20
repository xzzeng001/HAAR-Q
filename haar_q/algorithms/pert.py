"""
This file implements a perturbation estimator from haar.
"""

import logging
import typing
import dataclasses
import omegaconf
import torch
from .orbit import _load_orbital_model
from ..utility.sci_checkpoint import checkpoint_state
from ..utility.context import RuntimeContext
from ..utility.action_dict import action_dict


@dataclasses.dataclass
class PerturbationConfig:
    """
    The perturbation estimator from haar.
    """

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        """Estimate the Epstein-Nesbet correction on one selected external batch."""
        model = _load_orbital_model(context, runtime_config.model, checkpoint_data)
        state = checkpoint_data.get("haar", checkpoint_data.get("imag", {}))
        configs, psi = checkpoint_state(state)
        configs = configs.to(context.device)
        psi = psi.to(context.device)
        norm = torch.vdot(psi, psi).real
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("Perturbation theory requires a finite, positive CI norm.")
        energy0 = (torch.vdot(psi, model.apply_within(configs, psi, configs)).real / norm).item()
        logging.info("Current energy is %.8f", energy0)
        logging.info("Reference energy is %.8f", model.ref_energy)

        other_configs = model.find_relative(configs, psi, len(configs), configs)
        logging.info("Found %d external configurations.", len(other_configs))
        correction = 0.0
        if len(other_configs):
            coupling = model.apply_within(configs, psi, other_configs)
            # A zero coupling contributes nothing, even at a degenerate diagonal energy.
            coupled = coupling != 0
            denominator = energy0 - model.diagonal_term(other_configs).real[coupled]
            if (denominator == 0).any():
                raise ValueError("PT2 has a coupled external determinant with a zero energy denominator.")
            correction = ((coupling[coupled].abs().square() / norm) / denominator).sum().item()
        corrected = energy0 + correction
        logging.info("Second-order energy correction is %.8f", correction)
        logging.info("Corrected total energy is %.8f", corrected)
        logging.info(
            "Reference error before and after PT2: %.8f, %.8f", energy0 - model.ref_energy, corrected - model.ref_energy
        )


action_dict["pert"] = PerturbationConfig
