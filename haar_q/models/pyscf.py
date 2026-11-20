"""
This file provides an interface to work with PySCF-style integral arrays.

The model accepts 1-electron (h1e) and 2-electron (eri) integrals in the same
convention used by PySCF's FCI solver, converting them to the internal Hamiltonian
representation used by HAAR-Q.
"""

import logging
import dataclasses
import numpy
import torch
from ._integrals import _packed_terms
from ..networks.mlp import WaveFunctionElectronUpDown as MlpWaveFunction
from ..networks.mlp import WaveFunctionElectron as MlpWaveFunctionElectron
from ..networks.transformers import WaveFunctionElectronUpDown as TransformersWaveFunction
from ..networks.transformers import WaveFunctionElectron as TransformersWaveFunctionElectron
from ..hamiltonian import Hamiltonian
from ..utility.model_dict import model_dict, ModelProto, NetworkProto, NetworkConfigProto


@dataclasses.dataclass
class ModelConfig:
    """
    The configuration of the PySCF model.

    Integrals should be in chemist's notation (ij|kl), matching PySCF's convention.
    """

    # 1-electron integrals in MO basis, shape (norb, norb)
    h1e: numpy.ndarray
    # 2-electron repulsion integrals in MO basis, shape (norb, norb, norb, norb)
    # or compressed 4-fold symmetric shape (nij, nij) where nij = norb*(norb+1)//2
    eri: numpy.ndarray
    # Number of spatial orbitals
    n_orbit: int
    # Total number of electrons
    n_electron: int
    # Spin (n_alpha - n_beta, i.e. 2*S)
    n_spin: int = 0
    # Constant energy offset (e.g. nuclear repulsion energy)
    nuclear_repulsion: float = 0.0
    # Reference (FCI) energy for logging; defaults to zero
    ref_energy: float = 0.0


class Model(ModelProto[ModelConfig]):
    """
    This class handles models built from PySCF-style integral arrays (h1e and eri).

    The integrals are in chemist's (ij|kl) notation and are converted to the
    internal spin-orbital Hamiltonian representation used by HAAR-Q.
    """

    network_dict: dict[str, type[NetworkConfigProto["Model"]]] = {}

    config_t = ModelConfig

    def __init__(self, args: ModelConfig) -> None:
        n_orbit = args.n_orbit
        n_electron = args.n_electron
        n_spin = args.n_spin

        logging.info(
            "Building PySCF model: %d orbitals, %d electrons, spin=%d",
            n_orbit,
            n_electron,
            n_spin,
        )

        if not 0 < n_orbit <= 16384:
            raise ValueError("Expected 1..16384 spatial orbitals for signed int16 spin-orbital indices.")
        h1e = numpy.asarray(args.h1e, dtype=numpy.float64)
        if h1e.shape != (n_orbit, n_orbit):
            raise ValueError(f"Expected h1e shape ({n_orbit}, {n_orbit}), got {h1e.shape}")
        eri = numpy.asarray(args.eri, dtype=numpy.float64)
        pairs = n_orbit * (n_orbit + 1) // 2
        if eri.shape == (pairs, pairs):
            # The historical fourfold reader fills both pair orders from the
            # lower triangle. Preserve that convention even for asymmetric input.
            eri = eri[numpy.tril_indices(pairs)]
        elif eri.shape != (n_orbit,) * 4:
            raise ValueError(f"Expected full ERIs or a ({pairs}, {pairs}) pair matrix, got {eri.shape}")
        terms = _packed_terms((args.nuclear_repulsion, h1e, eri))
        self.hamiltonian = Hamiltonian(terms, kind="fermi")
        self.hamiltonian._sort_site_kind_coef()
        logging.info("Internal Hamiltonian representation successfully created")

        self.n_qubits: int = n_orbit * 2
        self.n_electrons: int = n_electron
        self.n_spins: int = n_spin
        self.ref_energy: float = args.ref_energy
        logging.info(
            "Identified %d qubits, %d electrons, spin=%d, ref_energy=%.10f",
            self.n_qubits,
            self.n_electrons,
            self.n_spins,
            self.ref_energy,
        )

    def apply_within(self, configs_i: torch.Tensor, psi_i: torch.Tensor, configs_j: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian.apply_within(configs_i, psi_i, configs_j)

    def find_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        count_selected: int,
        configs_exclude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.hamiltonian.find_relative(configs_i, psi_i, count_selected, configs_exclude)

    def list_relative(
        self,
        configs_i: torch.Tensor,
        psi_i: torch.Tensor,
        configs_exclude: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hamiltonian.list_relative(configs_i, psi_i, configs_exclude)

    def diagonal_term(self, configs: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian.diagonal_term(configs)

    def show_config(self, config: torch.Tensor) -> str:
        string = "".join(f"{i:08b}"[::-1] for i in config.cpu().numpy())
        return (
            "["
            + "".join(self._show_config_site(string[index : index + 2]) for index in range(0, self.n_qubits, 2))
            + "]"
        )

    def _show_config_site(self, string: str) -> str:
        match string:
            case "00":
                return " "
            case "10":
                return "↑"
            case "01":
                return "↓"
            case "11":
                return "↕"
            case _:
                raise ValueError(f"Invalid string: {string}")


model_dict["pyscf"] = Model


@dataclasses.dataclass
class MlpConfig:
    """
    The configuration of the MLP network.
    """

    # The hidden widths of the network
    hidden: tuple[int, ...] = (512,)

    def create(self, model: Model) -> NetworkProto:
        """
        Create a MLP network for the model.
        """
        logging.info("Hidden layer widths: %a", self.hidden)

        network = MlpWaveFunction(
            double_sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            spin_up=(model.n_electrons + model.n_spins) // 2,
            spin_down=(model.n_electrons - model.n_spins) // 2,
            hidden_size=self.hidden,
            ordering=+1,
        )

        return network


Model.network_dict["mlp/u1u1"] = MlpConfig


@dataclasses.dataclass
class TransformersConfig:
    """
    The configuration of the transformers network.
    """

    # Embedding dimension
    embedding_dim: int = 512
    # Heads number
    heads_num: int = 8
    # Feedforward dimension
    feed_forward_dim: int = 2048
    # Shared expert number
    shared_expert_num: int = 1
    # Routed expert number
    routed_expert_num: int = 0
    # Selected expert number
    selected_expert_num: int = 0
    # Network depth
    depth: int = 6

    def create(self, model: Model) -> NetworkProto:
        """
        Create a transformers network for the model.
        """
        logging.info(
            "Transformers network configuration: "
            "embedding dimension: %d, "
            "number of heads: %d, "
            "feed-forward dimension: %d, "
            "shared expert number: %d, "
            "routed expert number: %d, "
            "selected expert number: %d, "
            "depth: %d",
            self.embedding_dim,
            self.heads_num,
            self.feed_forward_dim,
            self.shared_expert_num,
            self.routed_expert_num,
            self.selected_expert_num,
            self.depth,
        )

        network = TransformersWaveFunction(
            double_sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            spin_up=(model.n_electrons + model.n_spins) // 2,
            spin_down=(model.n_electrons - model.n_spins) // 2,
            embedding_dim=self.embedding_dim,
            heads_num=self.heads_num,
            feed_forward_dim=self.feed_forward_dim,
            shared_num=self.shared_expert_num,
            routed_num=self.routed_expert_num,
            selected_num=self.selected_expert_num,
            depth=self.depth,
            ordering=+1,
        )

        return network


Model.network_dict["transformers/u1u1"] = TransformersConfig


@dataclasses.dataclass
class MlpElectronConfig:
    """
    The configuration of the MLP network with total electron number conservation.
    """

    # The hidden widths of the network
    hidden: tuple[int, ...] = (512,)

    def create(self, model: Model) -> NetworkProto:
        """
        Create a MLP network for the model.
        """
        logging.info("Hidden layer widths: %a", self.hidden)

        network = MlpWaveFunctionElectron(
            sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            electrons=model.n_electrons,
            hidden_size=self.hidden,
            ordering=+1,
        )

        return network


Model.network_dict["mlp/u1"] = MlpElectronConfig


@dataclasses.dataclass
class TransformersElectronConfig:
    """
    The configuration of the transformers network with total electron number conservation.
    """

    # Embedding dimension
    embedding_dim: int = 512
    # Heads number
    heads_num: int = 8
    # Feedforward dimension
    feed_forward_dim: int = 2048
    # Shared expert number
    shared_expert_num: int = 1
    # Routed expert number
    routed_expert_num: int = 0
    # Selected expert number
    selected_expert_num: int = 0
    # Network depth
    depth: int = 6

    def create(self, model: Model) -> NetworkProto:
        """
        Create a transformers network for the model.
        """
        logging.info(
            "Transformers network configuration: "
            "embedding dimension: %d, "
            "number of heads: %d, "
            "feed-forward dimension: %d, "
            "shared expert number: %d, "
            "routed expert number: %d, "
            "selected expert number: %d, "
            "depth: %d",
            self.embedding_dim,
            self.heads_num,
            self.feed_forward_dim,
            self.shared_expert_num,
            self.routed_expert_num,
            self.selected_expert_num,
            self.depth,
        )

        network = TransformersWaveFunctionElectron(
            sites=model.n_qubits,
            physical_dim=2,
            is_complex=True,
            electrons=model.n_electrons,
            embedding_dim=self.embedding_dim,
            heads_num=self.heads_num,
            feed_forward_dim=self.feed_forward_dim,
            shared_num=self.shared_expert_num,
            routed_num=self.routed_expert_num,
            selected_num=self.selected_expert_num,
            depth=self.depth,
            ordering=+1,
        )

        return network


Model.network_dict["transformers/u1"] = TransformersElectronConfig
