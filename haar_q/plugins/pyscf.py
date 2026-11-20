"""
PySCF-compatible FCI solver plugin for HAAR-Q.

This module exposes a :class:`Solver` that conforms to the PySCF FCI-solver
interface::

    solver = Solver(config)
    energy, ci = solver.kernel(h1e, eri, norb, nelec)

The ``config`` argument mirrors the YAML config accepted by ``haar_q.__main__``::

    config = {
        "action": {"name": "haar", "params": {}},
        "network": {"name": "mlp/u1u1", "params": {}},
        "optimizer": {"name": "Adam", "params": {"lr": 1e-3}},
        "common": {
            "device": "cpu",
            "dtype": "float64",
            "max_relative_step": 10,
        },
    }

Internally the solver closely mimics :func:`haar_q.__main__.main`:

1. Imports the requested algorithm and model modules.
2. Builds a :class:`~haar_q.models.pyscf.Model` from the supplied integral arrays.
3. Constructs a :class:`_PyscfRuntimeContext` that overrides
   :meth:`~haar_q.utility.context.RuntimeContext.folder` to use a temporary
   directory and :meth:`~haar_q.utility.context.RuntimeContext.create_model` to
   inject the pre-built model.  All other methods — including ``setup()`` and
   ``save()`` — are inherited unchanged from :class:`RuntimeContext` and work
   via normal polymorphism.
4. Runs the algorithm via its ``main()`` method.
5. When the requested cycle limit is reached, the base ``save()`` writes the
   checkpoint to the temporary directory and calls ``sys.exit(0)``.  The
   solver catches the resulting :class:`SystemExit` and reads the saved
   checkpoint file to extract the energy and CI data.

The returned CI array or checkpoint may be passed back as ``ci0`` to warm-start
subsequent calls.
"""

import pathlib
import shutil
import tempfile
import typing

import dacite
import numpy
import omegaconf
import torch

from ..models.pyscf import Model, ModelConfig
from ..utility.action_dict import create_action
from ..utility.context import DACITE_CAST, RuntimeContext
from ..utility.model_dict import ModelProto
from ..utility.sci_checkpoint import checkpoint_state, latest_ritz
from ..utility.spin_order import spin_order_sign


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _CIArray(numpy.ndarray):
    """
    A numpy array subclass that stores a HAAR-Q checkpoint dictionary.

    This allows the FCI solver to return a standard-looking CI wavefunction
    (the array) while still carrying the full state (network, optimizer, etc.)
    needed for a high-performance warm start in the next call to :meth:`kernel`.
    """

    def __new__(cls, input_array: numpy.ndarray, checkpoint: dict[str, typing.Any]) -> "_CIArray":
        obj = numpy.asanyarray(input_array).view(cls)
        obj.checkpoint = checkpoint
        return obj

    def __array_finalize__(self, obj: typing.Any) -> None:
        if obj is None:
            return
        self.checkpoint: dict[str, typing.Any] = getattr(obj, "checkpoint", {})


class _PyscfRuntimeContext(RuntimeContext):
    """
    A :class:`~haar_q.utility.context.RuntimeContext` subclass for programmatic
    (non-Hydra) use from the PySCF solver plugin.

    The only behavioural differences from the base class are:

    * :meth:`folder` returns an explicit path supplied at construction time
      instead of reading the Hydra output directory.
    * :meth:`create_model` returns a pre-built :class:`~haar_q.models.pyscf.Model`
      instead of instantiating one from the OmegaConf config.

    All other methods, including :meth:`~RuntimeContext.setup` and
    :meth:`~RuntimeContext.save`, are inherited unchanged.  Because they call
    ``self.folder()`` internally, they automatically use the temporary
    directory via normal polymorphism.
    """

    def __init__(
        self,
        model_instance: ModelProto,
        folder_path: pathlib.Path,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_instance: ModelProto = model_instance
        self.configure_model_devices(model_instance)
        self._folder_path: pathlib.Path = folder_path

    def folder(self) -> pathlib.Path:
        return self._folder_path

    def create_model(self, model_config: omegaconf.DictConfig) -> ModelProto:
        return self._model_instance


def _spin_counts(nelec: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(nelec, (int, numpy.integer)):
        return (int(nelec) + 1) // 2, int(nelec) // 2
    return int(nelec[0]), int(nelec[1])


def _to_pyscf_ci(
    data: dict[str, typing.Any], action_name: str, norb: int, nelec: int | tuple[int, int]
) -> numpy.ndarray | None:
    """
    Convert HAAR-Q sampled configurations and amplitudes to a PySCF CI vector.
    """
    try:
        from pyscf.fci import cistring  # type: ignore[import-untyped]
    except ImportError:
        return None

    if action_name not in ("haar", "imag"):
        return None

    haar_data = data.get("haar")
    if not haar_data:
        return None
    configs, psi = checkpoint_state(haar_data)

    nalpha, nbeta = _spin_counts(nelec)

    na = cistring.num_strings(norb, nalpha)
    nb = cistring.num_strings(norb, nbeta)

    # Limit CI vector size to avoid memory overflow (approx 10^8 elements ~ 1.6GB)
    if norb >= 64 or na * nb > 10**8:
        return None

    configs_np = configs.cpu().numpy()
    psi_np = psi.cpu().numpy()

    # Unpack bits: HAAR-Q order is little-endian bit-packed bytes
    bits = numpy.unpackbits(configs_np, axis=1, bitorder="little")

    # Extract alpha and beta occupancy (alpha=even, beta=odd bits)
    alpha_bits = bits[:, 0 : 2 * norb : 2]
    beta_bits = bits[:, 1 : 2 * norb : 2]

    # Convert to integer bitmasks
    pw2 = 2 ** numpy.arange(norb, dtype=numpy.uint64)
    alpha_dets = (alpha_bits.astype(numpy.uint64) * pw2).sum(axis=1)
    beta_dets = (beta_bits.astype(numpy.uint64) * pw2).sum(axis=1)

    # Only the requested spin sector fits in the PySCF alpha/beta CI array.
    if numpy.any(alpha_bits.sum(axis=1) != nalpha) or numpy.any(beta_bits.sum(axis=1) != nbeta):
        raise ValueError("Cannot export a CI array outside the requested alpha/beta spin sector.")
    addr_a = cistring.strs2addr(norb, nalpha, alpha_dets)
    addr_b = cistring.strs2addr(norb, nbeta, beta_dets)

    ci_array = numpy.zeros((na, nb), dtype=psi_np.dtype)
    ci_array[addr_a, addr_b] = psi_np * spin_order_sign(alpha_bits, beta_bits)

    return ci_array


def _from_pyscf_ci(ci_array: numpy.ndarray, norb: int, nelec: int | tuple[int, int]) -> dict[str, typing.Any]:
    """
    Convert a PySCF CI vector back to a HAAR-Q checkpoint (sampled pool).
    """
    try:
        from pyscf.fci import cistring  # type: ignore[import-untyped]
    except ImportError:
        return {}

    nalpha, nbeta = _spin_counts(nelec)
    if norb >= 64:
        raise ValueError("PySCF CI arrays require fewer than 64 spatial orbitals. Use a checkpoint for larger spaces.")
    shape = (cistring.num_strings(norb, nalpha), cistring.num_strings(norb, nbeta))
    if ci_array.shape != shape:
        raise ValueError(f"Expected a CI array of shape {shape}, got {ci_array.shape}.")
    idx_a, idx_b = numpy.nonzero(ci_array)

    str_a = cistring.addrs2str(norb, nalpha, idx_a)
    str_b = cistring.addrs2str(norb, nbeta, idx_b)

    # Pack one orbital at a time so conversion only needs O(nonzero CI entries)
    # scratch space in addition to the packed output.
    configs = numpy.zeros((len(idx_a), (2 * norb + 7) // 8), dtype=numpy.uint8)
    lower_beta_parity = numpy.zeros(len(idx_a), dtype=numpy.uint8)
    odd = numpy.zeros_like(lower_beta_parity)
    for j in range(norb):
        a = ((str_a >> j) & 1).astype(numpy.uint8)
        b = ((str_b >> j) & 1).astype(numpy.uint8)
        odd ^= a & lower_beta_parity
        lower_beta_parity ^= b
        configs[:, j // 4] |= (a | (b << 1)) << (2 * (j % 4))
    amplitudes = (ci_array[idx_a, idx_b] * (1 - 2 * odd.astype(numpy.int8))).astype(numpy.complex128)

    return {
        "haar": {
            "global": 0,
            "local": 0,
            "lanczos": 0,
            "pool": (torch.from_numpy(configs), torch.from_numpy(amplitudes)),
            "excited": {},
        }
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Solver:
    """
    PySCF-compatible FCI solver backed by HAAR-Q algorithms.

    The interface mirrors that of ``pyscf.fci.direct_spin1.FCISolver``::

        from haar_q.plugins.pyscf import Solver
        solver = Solver({
            "action": {"name": "haar", "params": {}},
            "network": {"name": "mlp/u1u1", "params": {}},
            "optimizer": {"name": "Adam", "params": {"lr": 1e-3}},
            "common": {"device": "cpu", "max_relative_step": 10},
        })
        energy, ci = solver.kernel(h1e, eri, norb, nelec)

    Parameters
    ----------
    config : dict or omegaconf.DictConfig
        Configuration in the same format as the ``haar_q.__main__`` YAML config.
        Required top-level keys: ``action``, ``network``, ``optimizer``,
        ``common``.  The ``model`` key is ignored (the model is built
        automatically from the integral arrays passed to :meth:`kernel`).
        ``common.max_relative_step`` controls how many new cycles to run on
        each call, including calls that resume an earlier CI checkpoint.
    """

    def __init__(self, config: dict[str, typing.Any] | omegaconf.DictConfig) -> None:
        if isinstance(config, dict):
            self.config: omegaconf.DictConfig = omegaconf.OmegaConf.create(config)
        else:
            self.config = config

    def kernel(
        self,
        h1e: numpy.ndarray,
        eri: numpy.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: dict[str, typing.Any] | numpy.ndarray | None = None,
        nuclear_repulsion: float | None = None,
        ref_energy: float = 0.0,
        **kwargs: typing.Any,
    ) -> tuple[float, numpy.ndarray | dict[str, typing.Any]]:
        """
        Solve the electronic structure problem for the given integrals.

        Parameters
        ----------
        h1e : numpy.ndarray
            1-electron integrals in MO basis, shape ``(norb, norb)``, in
            chemist's notation (same convention as PySCF).
        eri : numpy.ndarray
            2-electron repulsion integrals in MO basis in chemist's (ij|kl)
            notation.  Accepted shapes:

            * ``(norb, norb, norb, norb)`` – full 4-index array.
            * ``(nij, nij)`` – 4-fold symmetric compressed array
              (nij = norb*(norb+1)//2).
        norb : int
            Number of spatial orbitals.
        nelec : int or tuple[int, int]
            Number of electrons.  If a plain ``int``, the electrons are split
            as evenly as possible between alpha and beta spin (rounding alpha
            up).  If a 2-tuple ``(nalpha, nbeta)``, each spin count is used
            directly.
        ci0 : dict or numpy.ndarray, optional
            Initial guess or checkpoint data. If a ``dict``, it is treated as
            the full HAAR-Q checkpoint. If a ``numpy.ndarray``, it is treated as
            the PySCF CI vector and converted to a HAAR-Q pool.
        nuclear_repulsion : float
            Constant nuclear repulsion energy added to the Hamiltonian.
        ref_energy : float
            Reference energy used for logging only.
        **kwargs
            PySCF's ``ecore`` keyword supplies the same constant offset as
            ``nuclear_repulsion``. Supplying inconsistent values raises an error.

        Returns
        -------
        energy : float
            Ground-state energy after global algorithm steps.
        ci : numpy.ndarray or dict
            CI wavefunction as a numpy array. The array is an instance of
            ``_CIArray`` which carries the full HAAR-Q checkpoint for warm-start
            compatibility.
        """
        # ------------------------------------------------------------------
        # 1. Resolve electron counts
        # ------------------------------------------------------------------
        nalpha, nbeta = _spin_counts(nelec)
        ecore = float(kwargs.get("ecore", nuclear_repulsion if nuclear_repulsion is not None else 0.0))
        if nuclear_repulsion is not None and nuclear_repulsion != ecore:
            raise ValueError("ecore and nuclear_repulsion specify different constant energies.")
        if self.config.action.name not in {"haar", "imag"}:
            raise ValueError("The PySCF ground-state Solver requires action.name=haar (or its legacy alias imag).")

        n_electron = nalpha + nbeta
        n_spin = nalpha - nbeta

        # ------------------------------------------------------------------
        # 2. Build the selected algorithm using the CLI parameter schema.
        # ------------------------------------------------------------------
        run = create_action(self.config.action)

        # ------------------------------------------------------------------
        # 3. Build PySCF model from integral arrays
        # ------------------------------------------------------------------
        model = Model(
            ModelConfig(
                h1e=numpy.asarray(h1e, dtype=numpy.float64),
                eri=numpy.asarray(eri, dtype=numpy.float64),
                n_orbit=norb,
                n_electron=n_electron,
                n_spin=n_spin,
                nuclear_repulsion=ecore,
                ref_energy=ref_energy,
            )
        )

        # ------------------------------------------------------------------
        # 4. Build OmegaConf runtime_config — same structure as the YAML
        #    config, but with model.name = "pyscf" so that the algorithm
        #    can log the model type.  create_model() is overridden so the
        #    params field is never read.
        # ------------------------------------------------------------------
        runtime_config: omegaconf.DictConfig = omegaconf.OmegaConf.create(
            {
                "action": omegaconf.OmegaConf.to_container(self.config.action, resolve=True),
                "model": {"name": "pyscf", "params": {}},
                "network": omegaconf.OmegaConf.to_container(self.config.network, resolve=True),
                "optimizer": omegaconf.OmegaConf.to_container(self.config.optimizer, resolve=True),
            }
        )

        # ------------------------------------------------------------------
        # 5. Construct _PyscfRuntimeContext from config.common
        # ------------------------------------------------------------------
        common_data = omegaconf.OmegaConf.to_container(self.config.common, resolve=True)
        assert isinstance(common_data, dict)
        # Use dacite for type conversion (pathlib.Path, torch.device, etc.),
        # matching the same approach used in __main__.py.
        rt = dacite.from_dict(
            data_class=RuntimeContext,
            data=common_data,
            config=dacite.Config(cast=DACITE_CAST),
        )

        # ------------------------------------------------------------------
        # 6. Save caller gradient state; create temp directory
        # ------------------------------------------------------------------
        prev_grad_enabled = torch.is_grad_enabled()
        tmpdir: pathlib.Path | None = None
        try:
            tmpdir = pathlib.Path(tempfile.mkdtemp())
            # Handle ci0: could be full checkpoint dict or amplitudes array
            if ci0 is not None:
                if isinstance(ci0, _CIArray) and ci0.checkpoint:
                    checkpoint_to_save = ci0.checkpoint
                elif isinstance(ci0, numpy.ndarray):
                    checkpoint_to_save = _from_pyscf_ci(ci0, norb, (nalpha, nbeta))
                else:
                    checkpoint_to_save = ci0
                torch.save(checkpoint_to_save, tmpdir / "data.pth")

            context = _PyscfRuntimeContext(
                model_instance=model,
                folder_path=tmpdir,
                parent_path=rt.parent_path,
                random_seed=rt.random_seed,
                checkpoint_interval=rt.checkpoint_interval,
                device=rt.device,
                gpu_devices=rt.gpu_devices,
                dtype=rt.dtype,
                max_absolute_step=rt.max_absolute_step,
                max_relative_step=rt.max_relative_step,
            )

            # ------------------------------------------------------------------
            # 7. Setup context — loads checkpoint / ci0, configures RNG
            # ------------------------------------------------------------------
            checkpoint_data = context.setup()
            if context.cycle_limit_reached(self.config.action.name, checkpoint_data):
                raise ValueError("Warm start already reached max_absolute_step. Increase it or set max_relative_step.")

            # ------------------------------------------------------------------
            # 8. Run the algorithm.
            #    The base save() writes the checkpoint to the temp dir and
            #    then calls sys.exit(0) when max_absolute_step is reached.
            #    We catch the resulting SystemExit so kernel() can return.
            # ------------------------------------------------------------------
            try:
                run.main(
                    context=context,
                    runtime_config=runtime_config,
                    checkpoint_data=checkpoint_data,
                )
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise

            # ------------------------------------------------------------------
            # 9. Read the checkpoint that was saved by save()
            # ------------------------------------------------------------------
            data_path = tmpdir / "data.pth"
            if data_path.exists():
                data: dict[str, typing.Any] = torch.load(data_path, map_location="cpu", weights_only=True)
            else:
                data = {}

            # ------------------------------------------------------------------
            # 10. Extract energy and build ci output
            # ------------------------------------------------------------------
            energy = _extract_energy(data, self.config.action.name, model)
            # Strip the random engine state — it is stale by the next call.
            full_checkpoint: dict[str, typing.Any] = {k: v for k, v in data.items() if k != "random"}

            # Try to convert to PySCF CI vector format
            ci_array = _to_pyscf_ci(data, self.config.action.name, norb, (nalpha, nbeta))
            if ci_array is not None:
                ci: numpy.ndarray | dict[str, typing.Any] = _CIArray(ci_array, full_checkpoint)
            else:
                ci = full_checkpoint

        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)
            torch.set_grad_enabled(prev_grad_enabled)

        return energy, ci


def _extract_energy(data: dict[str, typing.Any], action_name: str, model: ModelProto | None = None) -> float:
    """Return the ground Ritz energy, using the pool expectation for legacy checkpoints."""
    haar = data.get("haar", data.get("imag", {}))
    root = latest_ritz(haar)
    if root is not None:
        return float(root[0])
    if "final_energy" in haar:
        return float(haar["final_energy"])
    pool = haar.get("pool")
    if model is not None and pool is not None:
        configs, psi = pool
        hpsi = model.apply_within(configs, psi, configs)
        return float((torch.vdot(psi, hpsi) / torch.vdot(psi, psi)).real)
    raise ValueError(f"No ground-state energy or wavefunction in the {action_name} checkpoint.")
