"""
This file implements the HAAR algorithm with iterative orbital optimization.
"""

import copy
import logging
import typing
import dataclasses
import pathlib
import omegaconf
import torch
import torch.utils.tensorboard
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from ..utility import losses
from ..utility.context import RuntimeContext
from ..utility.action_dict import action_dict
from ..utility.model_dict import ModelProto
from ..utility.optimizer import scale_learning_rate

# Import components from haar and orbit
from .haar import (
    _DynamicLanczos,
    _fit_amplitudes,
    _sampling_from_last_iteration,
    _merge_pool_from_neural_network_and_pool_from_last_iteration,
)
from .orbit import NaturalOrbitCalculator, _load_orbital_model, _rotated_basis
from ..models.optimized_basis import Model as OptimizedModel, ModelConfig as OptimizedModelConfig


@dataclasses.dataclass
class HaarWithOrbitConfig:
    """
    The two-step optimization process with iterative orbital optimization.
    """

    # Geometry jobs rotate spatial integrals supplied by the model.
    src_fcidump: pathlib.Path | None

    # --- HAAR Config Params (copied from haar.py) ---
    sampling_count_from_neural_network: int = 1024
    sampling_count_from_last_iteration: int = 1024
    krylov_extend_count: int = -1
    krylov_extend_first: bool = False
    krylov_single_extend: bool = False
    krylov_iteration: int = 32
    krylov_threshold: float = 1e-8
    krylov_period: int = 256
    krylov_eigen_count: int = 1
    loss_name: str = "sum_filtered_angle_scaled_log"
    local_step: int = -1
    local_loss: float = 1e-8
    logging_psi: int = 30
    local_batch_count_generation: int = 1
    local_batch_count_apply_within: int = 1
    local_batch_count_loss_function: int = 1

    def __post_init__(self) -> None:
        if self.local_step == -1:
            self.local_step = 10000
        if self.krylov_extend_count == -1:
            self.krylov_extend_count = 2048 if self.krylov_single_extend else 64

    def _optimize_orbit(
        self, context: RuntimeContext, model: ModelProto, configs: torch.Tensor, psi: torch.Tensor, step: int
    ) -> tuple[ModelProto, pathlib.Path]:
        """
        Perform orbital optimization and return a new model with the optimized basis and its path.

        This implementation uses SPATIAL orbital optimization (not spin-orbital) to preserve
        spin symmetry in restricted calculations. The same unitary transformation is applied
        to both alpha and beta spin-orbitals.
        """
        logging.info("Performing orbital optimization for step %d", step)

        n_orbit = typing.cast(typing.Any, model).n_qubits // 2
        calculator = NaturalOrbitCalculator(n_orbit)

        # 1. Calculate SPATIAL RDM (this now returns spatial orbital RDM, not spin-orbital)
        rdm_spatial = calculator.calculate_rdm(configs, psi)

        # 2. Diagonalize SPATIAL RDM to get Natural Orbitals
        # eigvals are occupation numbers, U_spatial columns are NOs in old basis
        eigvals, U_spatial = torch.linalg.eigh(rdm_spatial)

        # 3. Reorder U_spatial to be closest to Identity
        # We want to maximize sum(|U_{ii}|^2)
        cost_matrix = -(U_spatial.abs() ** 2).cpu().numpy()
        # For a square cost matrix the returned rows are ordered 0..n-1.
        # Apply the same permutation to occupations and orbital columns.
        _, col_ind = linear_sum_assignment(cost_matrix)
        U_spatial = U_spatial[:, col_ind]
        eigvals = eigvals[col_ind]

        output_data = _rotated_basis(model, self.src_fcidump, U_spatial, eigvals)

        optimized_path = context.folder() / f"optimized_basis_step_{step}.pt"
        torch.save(output_data, optimized_path)
        logging.info("Saved optimized basis to %s", optimized_path)

        new_model = OptimizedModel(OptimizedModelConfig(model_path=optimized_path))
        context.configure_model_devices(new_model)

        return new_model, optimized_path

    def main(
        self,
        context: RuntimeContext,
        runtime_config: omegaconf.DictConfig,
        checkpoint_data: dict[str, typing.Any],
    ) -> None:
        data = checkpoint_data
        model = _load_orbital_model(context, runtime_config.model, data)

        network = context.create_network(runtime_config.network, model, checkpoint_data.get("network"))

        optimizer = context.create_optimizer(
            runtime_config.optimizer, network.parameters(), checkpoint_data.get("optimizer")
        )

        logging.info(
            "Arguments Summary: "
            "Sampling Count From neural network: %d, "
            "Sampling Count From Last iteration: %d, "
            "Krylov Extend Count: %d, "
            "Krylov Extend First: %s, "
            "Krylov Single Extend: %s, "
            "Krylov Iteration: %d, "
            "Krylov Threshold: %.10f, "
            "Krylov Period: %d, "
            "Krylov Eigen Count: %d, "
            "Loss Function: %s, "
            "Local Steps: %d, "
            "Local Loss Threshold: %.10f, "
            "Logging Psi: %d, "
            "Local Batch Count For Generation: %d, "
            "Local Batch Count For Apply Within: %d, "
            "Local Batch Count For Loss Function: %d",
            self.sampling_count_from_neural_network,
            self.sampling_count_from_last_iteration,
            self.krylov_extend_count,
            "Yes" if self.krylov_extend_first else "No",
            "Yes" if self.krylov_single_extend else "No",
            self.krylov_iteration,
            self.krylov_threshold,
            self.krylov_period,
            self.krylov_eigen_count,
            self.loss_name,
            self.local_step,
            self.local_loss,
            self.logging_psi,
            self.local_batch_count_generation,
            self.local_batch_count_apply_within,
            self.local_batch_count_loss_function,
        )

        if "haar" not in data and "imag" in data:
            logging.warning("The 'imag' action is deprecated, please use 'haar' instead.")
            data["haar"] = data["imag"]
            del data["imag"]
        if "haar" not in data:
            data["haar"] = {"global": 0, "local": 0, "lanczos": 0, "pool": None, "excited": {}}
        else:
            if "excited" not in data["haar"] or not isinstance(data["haar"]["excited"], dict):
                data["haar"]["excited"] = {}
            if data["haar"]["pool"] is not None:
                pool_configs, pool_psi = data["haar"]["pool"]
                data["haar"]["pool"] = (pool_configs.to(device=context.device), pool_psi.to(device=context.device))

        writer = torch.utils.tensorboard.SummaryWriter(log_dir=context.folder())  # type: ignore

        while True:
            logging.info("Starting a new optimization cycle")

            # --- HAAR SAMPLING & LANCZOS ---
            logging.info("Sampling configurations from neural network")
            configs_from_neural_network, psi_from_neural_network, _, _ = network.generate_unique(
                self.sampling_count_from_neural_network, self.local_batch_count_generation
            )
            logging.info("Sampling configurations from last iteration")
            configs_from_last_iteration, psi_from_last_iteration = _sampling_from_last_iteration(
                data["haar"]["pool"], self.sampling_count_from_last_iteration
            )
            logging.info("Merging configurations")
            configs, original_psi = _merge_pool_from_neural_network_and_pool_from_last_iteration(
                configs_from_neural_network,
                psi_from_neural_network,
                configs_from_last_iteration,
                psi_from_last_iteration,
            )
            logging.info("Sampling completed, unique configurations count: %d", len(configs))

            # --- ORBITAL OPTIMIZATION ---
            # Optimize orbitals using the current best available state (merged pool/network)
            # This ensures Lanczos runs in the optimized basis.
            new_model, optimized_path = self._optimize_orbit(
                context, model, configs, original_psi, data["haar"]["global"]
            )

            # Update model and persist path
            model = new_model
            data["haar"]["model_path"] = str(optimized_path)

            logging.info("Computing target for local optimization (Lanczos)")
            target_energy: torch.Tensor
            lanczos_results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

            for lanczos_results in _DynamicLanczos(
                model=model,
                configs=configs,
                psi=original_psi,
                step=self.krylov_iteration,
                threshold=self.krylov_threshold,
                count_extend=self.krylov_extend_count,
                batch_count_apply_within=self.local_batch_count_apply_within,
                single_extend=self.krylov_single_extend,
                first_extend=self.krylov_extend_first,
                eigen_count=self.krylov_eigen_count,
                period=self.krylov_period,
            ).run():
                target_energy, configs, original_psi = lanczos_results[0]
                logging.info("Current energy: %.10f, samples: %d", target_energy.item(), len(configs))
                writer.add_scalar("haar/lanczos/energy", target_energy, data["haar"]["lanczos"])  # type: ignore
                writer.add_scalar("haar/lanczos/error", target_energy - model.ref_energy, data["haar"]["lanczos"])  # type: ignore
                data["haar"]["lanczos"] += 1

            data["haar"]["excited"][data["haar"]["global"]] = lanczos_results

            # --- TARGET PSI CALCULATION ---
            # Fit the root-summed probability distribution in the rotated determinant basis.
            target_prob = torch.zeros_like(original_psi, dtype=torch.float64)
            for _, _, p in lanczos_results:
                target_prob += (p.conj() * p).real
            original_psi = target_prob.sqrt().to(dtype=torch.complex128)

            max_index = original_psi.abs().argmax()
            target_psi = original_psi / original_psi[max_index]

            logging.info("Local optimization target calculated")

            # --- LOCAL OPTIMIZATION ---
            loss_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(losses, self.loss_name)

            reference_config = configs[max_index.reshape(1)]

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss, _ = _fit_amplitudes(
                    network,
                    configs,
                    target_psi,
                    reference_config,
                    self.local_batch_count_loss_function,
                    loss_func,
                    backward=True,
                )
                return loss

            loss: torch.Tensor
            try_index = 0
            while True:
                state_backup = copy.deepcopy(network.state_dict())
                optimizer_backup = copy.deepcopy(optimizer.state_dict())

                logging.info("Starting local optimization process")
                success = True
                last_loss: float = 0.0
                local_step: int = data["haar"]["local"]
                scale_learning_rate(optimizer, 1 / (1 << try_index))
                for i in range(self.local_step):
                    loss = optimizer.step(closure)  # type: ignore[assignment,arg-type]
                    logging.info("Local optimization in progress, step %d, current loss: %.10f", i, loss.item())
                    writer.add_scalar(f"haar/loss/{self.loss_name}", loss, local_step)  # type: ignore[no-untyped-call]
                    local_step += 1
                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning("Loss is NaN, restoring...")
                        success = False
                        break
                    if loss < self.local_loss:
                        break
                    if abs(loss - last_loss) < self.local_loss:
                        break
                    last_loss = loss.item()
                scale_learning_rate(optimizer, 1 << try_index)
                if success:
                    if any(torch.isnan(param).any() or torch.isinf(param).any() for param in network.parameters()):
                        logging.warning("NaN detected in parameters...")
                        success = False
                if success:
                    logging.info("Local optimization process completed")
                    data["haar"]["local"] = local_step
                    break
                network.load_state_dict(state_backup)
                optimizer.load_state_dict(optimizer_backup)
                try_index = try_index + 1

            loss, amplitudes = _fit_amplitudes(
                network,
                configs,
                target_psi,
                reference_config,
                self.local_batch_count_loss_function,
                loss_func,
                backward=False,
            )
            psi = torch.cat(amplitudes)
            final_energy = ((psi.conj() @ model.apply_within(configs, psi, configs)) / (psi.conj() @ psi)).real
            logging.info(
                "Loss: %.10f, Final energy: %.10f, Target energy: %.10f, Reference energy: %.10f, Final error: %.10f",
                loss.item(),
                final_energy.item(),
                target_energy.item(),
                model.ref_energy,
                final_energy.item() - model.ref_energy,
            )
            writer.add_scalar("haar/energy/state", final_energy, data["haar"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("haar/energy/target", target_energy, data["haar"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("haar/error/state", final_energy - model.ref_energy, data["haar"]["global"])  # type: ignore[no-untyped-call]
            writer.add_scalar("haar/error/target", target_energy - model.ref_energy, data["haar"]["global"])  # type: ignore[no-untyped-call]

            logging.info("Displaying the largest amplitudes")
            indices = target_psi.abs().argsort(descending=True)
            text = []
            for index in indices[: self.logging_psi]:
                this_config = model.show_config(configs[index])
                logging.info(
                    "Configuration: %s, Target amplitude: %s, Final amplitude: %s",
                    this_config,
                    f"{target_psi[index].item():.8f}",
                    f"{psi[index].item():.8f}",
                )
                text.append(
                    f"Configuration: {this_config}, Target amplitude: {target_psi[index].item():.8f}, Final amplitude: {psi[index].item():.8f}"
                )
            writer.add_text("config", "\n".join(text), data["haar"]["global"])  # type: ignore[no-untyped-call]
            writer.flush()  # type: ignore[no-untyped-call]

            data["haar"]["pool"] = (configs, original_psi)
            data["haar"]["global"] += 1
            data["network"] = network.state_dict()
            data["optimizer"] = optimizer.state_dict()
            context.save(data, data["haar"]["global"])
            logging.info("Checkpoint successfully saved")


action_dict["haar_with_orbit"] = HaarWithOrbitConfig
