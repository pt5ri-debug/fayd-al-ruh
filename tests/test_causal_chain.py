import pytest
from core.causal_chain import CausalChain
from core.causal_link import CausalLink


def test_chain_requires_connected_links():
    chain = CausalChain((CausalLink("A", "B", 0.5), CausalLink("B", "C", 0.8)))
    assert chain.confidence == 0.4
    with pytest.raises(ValueError):
        CausalChain((CausalLink("A", "B"), CausalLink("C", "D")))
