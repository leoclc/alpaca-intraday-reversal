from __future__ import annotations

import copy
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.config.loader import load_config
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.utils.time import iter_trading_days
from app.watchlist.daily_strategy_builder import build_watchlist
from app.watchlist.node_assets import fetch_asset_symbols, resolve_watchlist_asset_filters, resolve_watchlist_builder_base


def _param_grid() -> Iterable[Dict[str, float]]:
    quantiles = [0.35, 0.4, 0.45]
    qlb = [30, 60, 90]
    vol_ratios = [0.0, 0.5, 0.8]
    lookbacks = [1, 2]
    for q in quantiles:
        for lb in qlb:
            for vol in vol_ratios:
                for rlb in lookbacks:
                    yield {
                        "reversal_quantile": q,
                        "reversal_quantile_lookback_days": lb,
                        "volume_min_ratio": vol,
                        "reversal_lookback_days": rlb,
                    }


def find_params(
    cfg: Optional[Dict] = None,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    min_watchlist_size: int = 5,
    run_id: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    cfg = cfg or load_config()
    run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path(str(cfg.get("logs_dir") or "logs"))
    cfg["watchlists_dir"] = str(logs_dir / "backtests" / run_id / "watchlists_eval")

    base_url = resolve_watchlist_builder_base(cfg)
    asset_filters = resolve_watchlist_asset_filters(cfg) or {}
    logging.info("[TUNER] fetching asset universe via node base=%s filters=%s", base_url, asset_filters)
    symbols = fetch_asset_symbols(base_url=base_url, **asset_filters)
    logging.info("[TUNER] asset universe size=%s", len(symbols))

    store = AlpacaOHLCStore(cfg=cfg)
    trading_days = [d.isoformat() for d in iter_trading_days(start_date, end_date)]
    best: Optional[Tuple[int, Dict[str, float]]] = None

    for params in _param_grid():
        cfg2 = copy.deepcopy(cfg)
        strat = cfg2.setdefault("daily_trend_reversal", {})
        strat["reversal_mode"] = "quantile"
        for key, value in params.items():
            strat[key] = value
        min_size = None
        ok = True
        for date_str in trading_days:
            wl = build_watchlist(cfg2, target_date=date_str, symbols=symbols, data_store=store)
            size = len(wl)
            if min_size is None or size < min_size:
                min_size = size
            if size < min_watchlist_size:
                ok = False
                break
        if ok:
            logging.info("[TUNER] FOUND params=%s min_size=%s", params, min_size)
            return params
        if min_size is not None and (best is None or min_size > best[0]):
            best = (min_size, params)
        logging.info("[TUNER] tried params=%s min_size=%s", params, min_size)

    if best:
        logging.warning("[TUNER] no params hit target; best min_size=%s params=%s", best[0], best[1])
    else:
        logging.warning("[TUNER] no params evaluated")
    return None


if __name__ == "__main__":
    found = find_params()
    if found:
        print("FOUND", found)
    else:
        print("NO MATCH")
