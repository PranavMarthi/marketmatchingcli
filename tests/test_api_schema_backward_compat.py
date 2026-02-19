from marketmap.api.schemas import MarketNode


def test_market_node_keeps_xyz_backward_compat() -> None:
    node = MarketNode(id="m1", label="test", x=1.0, y=2.0, z=3.0)
    dumped = node.model_dump()
    assert dumped["x"] == 1.0
    assert dumped["y"] == 2.0
    assert dumped["z"] == 3.0
    assert "global_x" in dumped
    assert "neighborhood_key" in dumped
