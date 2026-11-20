"""Check numerical stability and symmetry sectors of sampling without replacement."""

from itertools import product
from typing import Any

import pytest
import torch

from haar_q.networks import mlp, transformers
from haar_q.networks.sampling import select_children
from haar_q.utility.bitspack import pack_int


@pytest.mark.parametrize("family", ["mlp", "transformers"])
@pytest.mark.parametrize("sector", ["normal", "electron", "spin"])
def test_unique_sampler_spans_the_allowed_support(family: str, sector: str) -> None:
    torch.manual_seed(153)
    module = mlp if family == "mlp" else transformers
    options: dict[str, Any] = (
        dict(hidden_size=(8,))
        if family == "mlp"
        else dict(embedding_dim=8, heads_num=2, feed_forward_dim=8, shared_num=1, routed_num=2, selected_num=1, depth=1)
    )
    common: dict[str, Any] = dict(is_complex=True, ordering=-1, **options)
    net: torch.nn.Module
    if sector == "normal":
        dim = 5 if family == "transformers" else 2
        rows = list(product(range(dim), repeat=3))
        net = module.WaveFunctionNormal(sites=3, physical_dim=dim, **common)
        bit_size = 4 if dim == 5 else 1
    else:
        bit_size = 1
        binary = list(product(range(2), repeat=4))
        if sector == "electron":
            rows = [r for r in binary if sum(r) == 2]
            net = module.WaveFunctionElectron(sites=4, electrons=2, physical_dim=2, **common)
        else:
            rows = [r for r in binary if sum(r[::2]) == 1 and sum(r[1::2]) == 1]
            net = module.WaveFunctionElectronUpDown(double_sites=4, spin_up=1, spin_down=1, physical_dim=2, **common)

    model = torch.jit.script(net.eval())
    expected = pack_int(torch.tensor(rows, dtype=torch.uint8), bit_size)
    with torch.no_grad():
        configs, amplitudes, _, _ = model.generate_unique(len(expected) + 2)
        assert sorted(configs.tolist()) == sorted(expected.tolist())
        torch.testing.assert_close(amplitudes, model(configs))
        assert amplitudes.abs().square().sum().item() == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize("offset", [0.0, 100.0])
def test_unique_sampler_preserves_rare_float32_states(offset: float) -> None:
    # Squared amplitudes differ by exp(-160). Both log probabilities are finite.
    model = mlp.WaveFunctionNormal(sites=1, physical_dim=2, is_complex=True, hidden_size=(), ordering=1)
    with torch.no_grad():
        for layer in model.amplitude[0].modules():
            if isinstance(layer, mlp.FakeLinear):
                layer.bias.copy_(torch.tensor([offset, offset - 80]))
        configs, amplitudes, _, _ = model.generate_unique(2)
    assert sorted(configs.tolist()) == [[0], [1]]
    torch.testing.assert_close(
        amplitudes[configs[:, 0].argsort()].abs(),
        torch.tensor([1.0, 1.8048513878454153e-35], dtype=torch.float64),
        rtol=2e-6,
        atol=0,
    )


def test_child_selection_matches_conditional_gumbel_and_parent_indices() -> None:
    # A direct evaluation is well conditioned for these moderate log probabilities.
    logits = torch.tensor([[0.4, 0.1, 0.5], [0.2, 0.8, 0]], dtype=torch.float64).log() / 2
    parent_log_prob = torch.tensor([0.3, 0.7], dtype=torch.float64).log()
    parent_score = torch.tensor([0.0, -1.0], dtype=torch.float64)
    log_prob = 2 * logits + parent_log_prob[:, None]
    torch.manual_seed(603)
    gumbels = log_prob - (-torch.rand_like(log_prob).log()).log()
    direct = -((-parent_score[:, None]).exp() + (-gumbels).exp() - (-gumbels.amax(1, keepdim=True)).exp()).log()
    torch.manual_seed(603)
    parents, children, probabilities, scores = select_children(logits, parent_log_prob, parent_score, 4)
    torch.testing.assert_close(scores, direct.flatten().topk(4).values)
    torch.testing.assert_close(scores, direct[parents, children])
    torch.testing.assert_close(probabilities, log_prob[parents, children])
