# Alpaca OHLC Daily Trend Reversal

Daily trend filter + 1-day reversal strategy built on cached daily OHLC bars.

## Layout
- `app/data/alpaca_ohlc_store.py`: daily bar cache + Alpaca fetch.
- `app/strategies/daily_trend_reversal.py`: signal generation + trade plan.
- `app/execution/daily_execution_model.py`: shared entry/exit simulation helpers.
- `app/watchlist/daily_strategy_builder.py`: watchlist builder + scoring.
- `app/replay/daily_strategy_replay.py`: replay runner.
- `app/live/runner.py`: live runner (Alpaca REST).

## Usage
- Build watchlist:
  - `python -c "from app.watchlist.daily_strategy_builder import build_watchlist; from app.config.loader import load_config; build_watchlist(load_config())"`
- Node asset universe is used by default via `watchlist_node_base` + `watchlist_asset_filters` in `config.json`.
- Replay:
  - `python -c "from app.replay.daily_strategy_replay import run_replay; from app.config.loader import load_config; run_replay(load_config(), start_date='2025-01-01', end_date='2025-03-31')"`
- Backtest (build watchlist each day, then replay):
  - `python -c "from app.backtest.daily_trend_backtest import run_backtest; from app.config.loader import load_config; run_backtest(load_config(), start_date='2025-01-01', end_date='2025-03-31', out_path='logs/backtest.json')"`
- Live:
  - `python -c "from app.live.runner import run_live; run_live()"`
- Intraday-only flatten (run on a timer close to the bell):
  - `python -c "from app.live.runner import run_flatten; run_flatten()"`
- Parameter sweep:
  - `python -c "from app.backtest.param_sweep_daily_trend import run_param_sweep; from app.config.loader import load_config; run_param_sweep(load_config(), start_date='2025-01-01', end_date='2025-12-31')"`
