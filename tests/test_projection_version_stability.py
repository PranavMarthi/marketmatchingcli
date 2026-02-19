from marketmap.projection.hierarchical_projection import _projection_version


def test_projection_version_stable_for_same_inputs() -> None:
    a = _projection_version("abc123")
    b = _projection_version("abc123")
    assert a == b
