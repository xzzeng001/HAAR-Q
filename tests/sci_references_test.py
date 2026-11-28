"""Reference determinant packing, first-cycle selection and restart behavior."""

from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest
import torch

from haar_q import __main__ as cli
from haar_q.algorithms.haar import _DynamicLanczos
from haar_q.algorithms.references import ReferenceDeterminant, fit_anchor, pack_references
from haar_q.models.fcidump import Model, ModelConfig
from haar_q.utility.losses import sum_filtered_angle_scaled_log


def test_single_krylov_expansion_on_molecular_hamiltonian() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "hamiltonians" / "H2.FCIDUMP"
    model = Model(ModelConfig(path))
    solver = _DynamicLanczos(
        model,
        torch.tensor([[3]], dtype=torch.uint8),
        torch.tensor([1], dtype=torch.complex128),
        step=1,
        threshold=1e-12,
        count_extend=1,
        batch_count_apply_within=1,
        single_extend=True,
        first_extend=False,
    )
    [(energy, basis, vector)] = list(solver.run())[-1]
    assert basis.shape == (2, 1) and basis[0, 0].item() == 3 and basis[1, 0].item() in {6, 9, 12}
    torch.testing.assert_close(model.apply_within(basis, vector, basis), energy * vector)


def test_reference_fit_handles_exact_neural_nodes_and_disjoint_support() -> None:
    values = torch.tensor([0, 1j, 0], dtype=torch.complex128, requires_grad=True)
    configs = torch.arange(3).reshape(-1, 1)
    target = torch.tensor([1, 0.5, 0], dtype=torch.complex128)

    def network(rows: torch.Tensor) -> torch.Tensor:
        return values[rows[:, 0]]

    index = fit_anchor(network, configs, target, batches=2)
    assert index.item() == 1
    loss = sum_filtered_angle_scaled_log(values / values[index], target / target[index])
    (gradient,) = torch.autograd.grad(loss, values)
    assert torch.isfinite(loss) and gradient.isfinite().all()
    with pytest.raises(ValueError, match="both the target and the network"):
        fit_anchor(network, configs, torch.tensor([1, 0, 0]), batches=1)


def test_108_spatial_orbital_references_pack_interleaved_spins_and_signed_weights() -> None:
    refs = [ReferenceDeterminant("a" + "0" * 106 + "b", -2e-300), ReferenceDeterminant("b" + "0" * 106 + "a", 1e-300)]
    configs, weights = pack_references(refs, 108, 2, 0, torch.device("cpu"))
    assert list(configs.shape) == [2, 27]
    assert configs[:, [0, 26]].tolist() == [[1, 128], [2, 64]]
    assert not configs[:, 1:-1].any()
    assert weights.tolist() == [-1 + 0j, 0.5 + 0j]


@pytest.mark.parametrize(
    ("refs", "message"),
    [
        ([ReferenceDeterminant("220")], "4 spatial occupations"),
        ([ReferenceDeterminant("abcd")], "from 0, a, b, 2"),
        ([ReferenceDeterminant("aa00")], "NELEC=2, MS2=0"),
        ([ReferenceDeterminant("2a00")], "NELEC=2, MS2=0"),
        ([ReferenceDeterminant("2000")] * 2, "Duplicate"),
        ([ReferenceDeterminant("2000", float("nan"))], "finite"),
        ([ReferenceDeterminant("2000", 0.0)], "nonzero"),
    ],
)
def test_invalid_initial_references_fail_with_actionable_input_error(
    refs: list[ReferenceDeterminant], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pack_references(refs, 4, 2, 0, torch.device("cpu"))


@pytest.mark.parametrize("method", ["haar", "excited"])
def test_molecular_reference_seeds_use_public_input_and_only_first_cycle(
    method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "input.yaml")
    cfg.files.fcidump = str(root / "examples" / "hamiltonians" / "H2.FCIDUMP")
    cfg.sampling_control.seed_count = 1
    cfg.training.optimizer_steps = 1
    cfg.common.max_relative_step = 1
    cfg.options.method.haar = {
        "krylov_extend_count": 0,
        "previous_seed_count": 0,
        "krylov_iteration": 3,
        "initial_references": [
            {"occupation": "20"},
            {"occupation": "ab"},
            {"occupation": "ba", "coefficient": -1.0},
            {"occupation": "02", "coefficient": 0.0},
        ],
    }
    path = tmp_path / "input.yaml"
    OmegaConf.save(cfg, path)
    run = tmp_path / "result"
    argv = ["haar-q", "-i", str(path), f"method={method}", f"hydra.run.dir={run}"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as finished:
        cli.main()
    assert finished.value.code == 0
    state = torch.load(run / "data.pth", weights_only=True)["haar"]
    energy, configs, _ = state["excited"][0][0]
    assert sorted(configs[:, 0].tolist()) == [3, 6, 9, 12]
    # H2's complete 1-alpha/1-beta sector in the shipped FCIDUMP.
    model = Model(ModelConfig(Path(cfg.files.fcidump)))
    basis = torch.tensor([[3], [6], [9], [12]], dtype=torch.uint8)
    eye = torch.eye(4, dtype=torch.complex128)
    dense = torch.stack([model.apply_within(basis, v, basis) for v in eye], dim=1)
    exact = torch.linalg.eigvalsh(dense)
    torch.testing.assert_close(energy, exact[0])
    if method == "excited":
        torch.testing.assert_close(state["excited"][0][1][0], exact[1])
    # A resumed cycle with a single network seed and no selected neighbors must
    # have one row. A single requested root fits this intentionally tiny support.
    monkeypatch.setattr(sys, "argv", [*argv, "states.roots=1", "common.random_seed=null"])
    with pytest.raises(SystemExit) as finished:
        cli.main()
    assert finished.value.code == 0
    resumed = torch.load(run / "data.pth", weights_only=True)["haar"]
    assert resumed["global"] == 2 and len(resumed["excited"][1][0][1]) == 1
