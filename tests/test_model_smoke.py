"""Import-smoke test for gridsense.models."""


def test_importable() -> None:
    import gridsense.models  # noqa: F401
    import gridsense.models.dcrnn  # noqa: F401
    import gridsense.models.gwnet  # noqa: F401
