"""
This file provides an interface to work with FCIDUMP files.
"""

import logging
import dataclasses
import re
import gzip
import pathlib
import hashlib
import tempfile
from collections.abc import Iterator
import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from openfermion.config import EQ_TOLERANCE  # type: ignore[import-untyped]
import platformdirs
from ..networks.mlp import WaveFunctionElectronUpDown as MlpWaveFunction
from ..networks.mlp import WaveFunctionElectron as MlpWaveFunctionElectron
from ..networks.transformers import WaveFunctionElectronUpDown as TransformersWaveFunction
from ..networks.transformers import WaveFunctionElectron as TransformersWaveFunctionElectron
from ..hamiltonian import Hamiltonian
from ..utility.model_dict import model_dict, ModelProto, NetworkProto, NetworkConfigProto


@dataclasses.dataclass
class ModelConfig:
    """
    The configuration of the model.
    """

    # The path of the model file
    model_path: pathlib.Path
    # The ref energy of the model, leave empty to read from FCIDUMP.yaml
    ref_energy: float | None = None


def _load_integrals(
    file_name: pathlib.Path, *, headonly: bool = False
) -> tuple[tuple[int, int, int], tuple[float, np.ndarray, np.ndarray] | None]:
    """Read restricted spatial integrals in eightfold-packed chemists' notation."""
    with (
        gzip.open(file_name, "rt", encoding="utf-8") if file_name.suffix == ".gz" else file_name.open(encoding="utf-8")
    ) as stream:
        header = []
        for line in stream:
            header.append(line)
            if re.search(r"&end\b|/", line, flags=re.I):
                break
        else:
            raise ValueError("Missing FCIDUMP header terminator (&END or /).")
        fields = dict(re.findall(r"\b(NORB|NELEC|MS2)\s*=\s*([+-]?\d+)", " ".join(header).upper()))
        if fields.keys() != {"NORB", "NELEC", "MS2"}:
            raise ValueError("FCIDUMP must specify NORB, NELEC and MS2.")
        if re.search(r"\b(?:I?UHF)\s*=\s*(?:1\b|\.?T)", " ".join(header), flags=re.I):
            raise ValueError("The FCIDUMP model requires restricted spatial integrals.")
        n, nel, spin = (int(fields[key]) for key in ("NORB", "NELEC", "MS2"))
        if not 0 < n <= 16384 or (nel - spin) % 2 or not abs(spin) <= nel <= 2 * n - abs(spin):
            raise ValueError("Invalid FCIDUMP orbital count, electron count or spin projection.")
        if headonly:
            return (n, nel, spin), None
        pairs = n * (n + 1) // 2
        eri = np.zeros(pairs * (pairs + 1) // 2, dtype=np.float64)
        h1 = np.zeros((n, n), dtype=np.float64)
        core = 0.0
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            coeff, *indices = parts
            v = float(coeff.replace("D", "E").replace("d", "e"))
            i, j, k, ell = map(int, indices)
            if not all(0 <= index <= n for index in (i, j, k, ell)):
                raise ValueError(f"Orbital index outside 0..{n}: {indices}")
            if i == j == k == ell == 0:
                core = v
            elif j == k == ell == 0:
                continue  # Orbital energies in the Psi4 extension.
            elif i and j and k == ell == 0:
                h1[i - 1, j - 1] = h1[j - 1, i - 1] = v
            elif i and j and k and ell:
                p = max(i, j) * (max(i, j) - 1) // 2 + min(i, j) - 1
                q = max(k, ell) * (max(k, ell) - 1) // 2 + min(k, ell) - 1
                hi, lo = max(p, q), min(p, q)
                eri[hi * (hi + 1) // 2 + lo] = v
            else:
                raise ValueError(f"Invalid FCIDUMP integral indices: {indices}")
    return (n, nel, spin), (core, h1, eri)


def _term_blocks(core: float, h1: np.ndarray, eri: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield columns of normal-ordered sites and their real coefficients in bounded blocks."""
    tol = EQ_TOLERANCE
    if abs(core) >= tol:
        yield np.empty((1, 0), np.int16), np.array([core])
    n = len(h1)
    rows, cols = np.nonzero(np.abs(h1) >= tol)
    for spin in (0, 1):
        yield np.column_stack((2 * rows + spin, 2 * cols + spin)), h1[rows, cols]
    # Index each spin orbital pair (p>q). Only annihilator pairs with the same spin counts contribute.
    s, r = np.triu_indices(2 * n, k=1)
    pair_spin = (s % 2) + (r % 2)
    groups = [(r[pair_spin == group], s[pair_spin == group]) for group in (0, 1, 2)]
    a = np.arange(n)
    hi = np.maximum.outer(a, a)
    pair = hi * (hi + 1) // 2 + np.minimum.outer(a, a)

    def half_integral(p: np.integer, s: np.ndarray, q: np.integer, r: np.ndarray) -> np.ndarray:
        ps, qr = pair[p // 2, s // 2], pair[q // 2, r // 2]
        major = np.maximum(ps, qr)
        half = 0.5 * eri[major * (major + 1) // 2 + np.minimum(ps, qr)]
        return np.where(((p % 2) == (s % 2)) & (np.abs(half) >= tol), half, 0.0)

    for p, q in zip(r, s):
        ra, sa = groups[(p % 2) + (q % 2)]
        direct = half_integral(p, sa, q, ra)
        exchange = half_integral(p, ra, q, sa)
        # OpenFermion discards small terms after each normal-order addition.
        # Lexicographic raw spin indices add D, -X, -X, D (half integrals).
        value = direct - exchange
        value[np.abs(value) < tol] = 0
        value -= exchange
        value[np.abs(value) < tol] = 0
        value += direct
        active = np.abs(value) >= tol
        if active.any():
            sites = np.empty((np.count_nonzero(active), 4), np.int16)
            sites[:, 0], sites[:, 1] = p, q
            sites[:, 2], sites[:, 3] = ra[active], sa[active]
            yield sites, value[active]


def _packed_terms(integrals: tuple[float, np.ndarray, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Count first so output allocation is exact, even for a nearly dense active space.
    count = sum(len(values) for _, values in _term_blocks(*integrals))
    logging.info(
        "Preparing %d normal-ordered FCIDUMP terms (%.2f GiB of packed operator data)", count, count * 28 / 2**30
    )
    sites = np.zeros((count, 4), dtype=np.int16)
    kinds = np.full((count, 4), 2, dtype=np.uint8)
    coefficients = np.zeros((count, 2), dtype=np.float64)
    start = 0
    for block, values in _term_blocks(*integrals):
        end = start + len(values)
        width = block.shape[1]
        sites[start:end, :width] = block
        kinds[start:end, : width // 2] = 1
        kinds[start:end, width // 2 : width] = 0
        coefficients[start:end, 0] = values
        start = end
    return torch.from_numpy(sites), torch.from_numpy(kinds), torch.from_numpy(coefficients)


def read_fcidump(
    file_name: pathlib.Path, *, headonly: bool = False
) -> tuple[tuple[int, int, int], dict[tuple[tuple[int, int], ...], complex]]:
    """Read metadata and optionally return the historical dictionary representation."""
    header, integrals = _load_integrals(file_name, headonly=headonly)
    terms = {}
    if integrals is not None:
        for sites, values in _term_blocks(*integrals):
            for row, value in zip(sites, values):
                key = tuple((int(site), int(k < len(row) // 2)) for k, site in enumerate(row))
                terms[key] = complex(value)
    return header, terms


class Model(ModelProto[ModelConfig]):
    """
    This class handles the models from FCIDUMP files.
    """

    network_dict: dict[str, type[NetworkConfigProto["Model"]]] = {}

    config_t = ModelConfig

    def __init__(self, args: ModelConfig) -> None:
        model_path = args.model_path
        model_name = model_path.name
        ref_energy = args.ref_energy

        with model_path.open("rb") as raw:
            checksum = hashlib.file_digest(raw, "sha256").hexdigest() + "v6"
        cache_file = platformdirs.user_cache_path("haar-q", "HAAR-Q") / checksum
        if cache_file.exists():
            logging.info("Loading FCIDUMP metadata from file: %s", model_path)
            (n_orbit, n_electron, n_spin), _ = read_fcidump(model_path, headonly=True)
            logging.info("FCIDUMP metadata successfully loaded")

            logging.info("Mapping FCIDUMP Hamiltonian from cache")
            # Private mappings keep the cache immutable and let the OS reclaim clean
            # CPU pages under memory pressure without a separate anonymous copy.
            packed_hamiltonian = torch.load(cache_file, map_location="cpu", weights_only=True, mmap=True)
            self.hamiltonian = Hamiltonian(packed_hamiltonian, kind="fermi")
            logging.info("FCIDUMP Hamiltonian successfully loaded")
        else:
            logging.info("Loading FCIDUMP Hamiltonian from file: %s", model_path)
            (n_orbit, n_electron, n_spin), integrals = _load_integrals(model_path)
            assert integrals is not None
            logging.info("FCIDUMP Hamiltonian successfully loaded")

            logging.info("Packing restricted spatial integrals into normal-ordered Hamiltonian terms")
            self.hamiltonian = Hamiltonian(_packed_terms(integrals), kind="fermi")
            del integrals
            self.hamiltonian._sort_site_kind_coef()
            logging.info("Internal Hamiltonian representation has been successfully created")

            logging.info("Caching packed FCIDUMP Hamiltonian")
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache_file.parent) as tmp:
                staging = pathlib.Path(tmp) / "terms"
                torch.save((self.hamiltonian.site, self.hamiltonian.kind, self.hamiltonian.coef), staging)
                staging.replace(cache_file)
            logging.info("FCIDUMP Hamiltonian successfully cached")

        self.n_qubits: int = n_orbit * 2
        self.n_electrons: int = n_electron
        self.n_spins: int = n_spin
        logging.info(
            "Identified %d qubits, %d electrons and %d spin",
            self.n_qubits,
            self.n_electrons,
            self.n_spins,
        )

        self.ref_energy: float
        if ref_energy is not None:
            self.ref_energy = ref_energy
        else:
            fcidump_ref_energy_file = model_path.parent / "FCIDUMP.yaml"
            if fcidump_ref_energy_file.exists():
                with open(fcidump_ref_energy_file, "rt", encoding="utf-8") as file:
                    fcidump_ref_energy_data = yaml.safe_load(file)
                self.ref_energy = fcidump_ref_energy_data.get(model_name, 0)
            else:
                self.ref_energy = 0
        logging.info("Reference energy for model is %.10f", self.ref_energy)

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


model_dict["fcidump"] = Model


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
