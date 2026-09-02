from __future__ import annotations

from pathlib import Path

import pytest

from scripts import prepare_public_data


def test_public_setup_refuses_user_supplied_market_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annual = tmp_path / "annual.csv"
    annual.write_text("source\ncashncarry/fifaworldranking CC0\n", encoding="utf-8")
    market = tmp_path / "market.csv"
    market.write_text("source\nuser supplied\n", encoding="utf-8")
    monkeypatch.setattr(prepare_public_data, "ANNUAL_CSV", annual)
    monkeypatch.setattr(prepare_public_data, "SQUAD_PROXY_CSV", market)

    with pytest.raises(RuntimeError, match="refusing to overwrite user-supplied"):
        prepare_public_data.refuse_to_overwrite_user_inputs()
