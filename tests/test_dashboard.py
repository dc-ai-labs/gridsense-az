"""Import-smoke test for the dashboard components (Playwright tests arrive in T+9)."""


def test_importable() -> None:
    from app.components import feeder_map, forecast_chart, metrics_panel, scenario_panel  # noqa: F401
