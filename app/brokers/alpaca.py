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
        take_profit: float,
        stop_loss: float,
        tif: str = "day",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "qty": int(qty),
            "type": entry_type,
            "time_in_force": tif,
            "order_class": "bracket",
            "take_profit": {"limit_price": round(float(take_profit), 4)},
            "stop_loss": {"stop_price": round(float(stop_loss), 4)},
        }
        if entry_type == "limit":
            if entry_price is None:
                raise ValueError("limit order requires entry_price")
            payload["limit_price"] = round(float(entry_price), 4)
        return self.submit_order(payload)

    def list_positions(self) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        url = f"{self.base_url}/v2/positions"
        resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def close_position(self, symbol: str) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        url = f"{self.base_url}/v2/positions/{symbol}"
        resp = requests.delete(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
