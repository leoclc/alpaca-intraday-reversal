from __future__ import annotations

import requests
from typing import Any, Dict, Optional


class AlpacaBroker:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        alp = cfg.get("alpaca") or {}
        self.base_url = str(alp.get("base_url") or "https://paper-api.alpaca.markets").rstrip("/")
        self.data_url = str(alp.get("data_url") or "https://data.alpaca.markets").rstrip("/")
        self.key_id = str(alp.get("api_key_id") or "")
        self.secret_key = str(alp.get("api_secret_key") or "")
        self.timeout = alp.get("timeout_sec") or 10

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def ready(self) -> bool:
        return bool(self.key_id and self.secret_key)

    def get_account(self) -> Optional[Dict[str, Any]]:
        if not self.ready():
            return None
        url = f"{self.base_url}/v2/account"
        resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def submit_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        url = f"{self.base_url}/v2/orders"
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def submit_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_type: str,
        entry_price: Optional[float],
        base_price: Optional[float],
        take_profit: float,
        stop_loss: float,
        tif: str = "day",
    ) -> Dict[str, Any]:
        def _round_price(price: float) -> float:
            # Alpaca rejects sub-penny increments for >= $1 stocks.
            return round(float(price), 2 if price >= 1 else 4)

        def _min_tick(base: float) -> float:
            return 0.01 if base >= 1 else 0.0001

        if base_price is None:
            base_price = float(entry_price) if entry_price is not None else float(take_profit)
        base_price = float(base_price)
        tick = _min_tick(base_price)
        tp = float(take_profit)
        sl = float(stop_loss)
        if str(side).lower() == "buy":
            if tp < base_price + tick:
                tp = base_price + tick
            if sl > base_price - tick:
                sl = base_price - tick
        else:
            if tp > base_price - tick:
                tp = base_price - tick
            if sl < base_price + tick:
                sl = base_price + tick
        tp = _round_price(tp)
        sl = _round_price(sl)

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "qty": int(qty),
            "type": entry_type,
            "time_in_force": tif,
            "order_class": "bracket",
            "take_profit": {"limit_price": tp},
            "stop_loss": {"stop_price": sl},
        }
        if entry_type == "limit":
            if entry_price is None:
                raise ValueError("limit order requires entry_price")
            payload["limit_price"] = _round_price(entry_price)
        return self.submit_order(payload)

    def list_positions(self) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        url = f"{self.base_url}/v2/positions"
        resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_orders(
        self,
        status: str = "open",
        symbols: Optional[list[str]] = None,
        after: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        url = f"{self.base_url}/v2/orders"
        params: Dict[str, Any] = {"status": status, "limit": int(limit)}
        if symbols:
            params["symbols"] = ",".join(symbols)
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        resp = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def cancel_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        url = f"{self.base_url}/v2/orders"
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        resp = requests.delete(url, headers=self._headers(), params=params, timeout=self.timeout)
        if resp.status_code not in (200, 204):
            resp.raise_for_status()
        if resp.content:
            try:
                return resp.json()
            except Exception:
                return {}
        return {}

    def close_position(self, symbol: str) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        url = f"{self.base_url}/v2/positions/{symbol}"
        resp = requests.delete(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
