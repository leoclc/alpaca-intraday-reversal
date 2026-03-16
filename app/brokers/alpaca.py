from __future__ import annotations

import logging
import re
import time
import requests
from typing import Any, Dict, Optional


def _round_price_for_alpaca(price: float) -> float:
    # Alpaca rejects sub-penny increments for >= $1 stocks.
    px = float(price)
    return round(px, 2 if px >= 1 else 4)


def _min_tick_for_alpaca(base: float) -> float:
    return 0.01 if float(base) >= 1 else 0.0001


def _extract_base_price_from_422(resp: requests.Response) -> Optional[float]:
    try:
        payload = resp.json() if resp is not None else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return None
    if payload.get("base_price") is not None:
        try:
            return float(payload.get("base_price"))
        except Exception:
            pass
    message = str(payload.get("message") or "")
    m = re.search(r"base_price\s*([0-9]+(?:\.[0-9]+)?)", message)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _sanitize_bracket_prices_from_entry_side(
    entry_side: str,
    base_price: float,
    take_profit: float,
    stop_loss: float,
) -> tuple[float, float]:
    # entry_side semantics:
    # - buy  => long entry, TP above, SL below
    # - sell => short entry, TP below, SL above
    tick = _min_tick_for_alpaca(base_price)
    tp = float(take_profit)
    sl = float(stop_loss)
    side = str(entry_side or "").lower().strip()
    if side == "buy":
        if tp < base_price + tick:
            tp = base_price + tick
        if sl > base_price - tick:
            sl = base_price - tick
    else:
        if tp > base_price - tick:
            tp = base_price - tick
        if sl < base_price + tick:
            sl = base_price + tick
    return _round_price_for_alpaca(tp), _round_price_for_alpaca(sl)


def _sanitize_oco_prices_from_close_side(
    close_side: str,
    base_price: float,
    take_profit: float,
    stop_loss: float,
) -> tuple[float, float]:
    # close_side semantics:
    # - sell => close long, TP above, SL below
    # - buy  => close short, TP below, SL above
    tick = _min_tick_for_alpaca(base_price)
    tp = float(take_profit)
    sl = float(stop_loss)
    side = str(close_side or "").lower().strip()
    if side == "sell":
        if tp < base_price + tick:
            tp = base_price + tick
        if sl > base_price - tick:
            sl = base_price - tick
    else:
        if tp > base_price - tick:
            tp = base_price - tick
        if sl < base_price + tick:
            sl = base_price + tick
    return _round_price_for_alpaca(tp), _round_price_for_alpaca(sl)


class AlpacaBroker:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        alp = cfg.get("alpaca") or {}
        self.base_url = str(alp.get("base_url") or "https://paper-api.alpaca.markets").rstrip("/")
        self.data_url = str(alp.get("data_url") or "https://data.alpaca.markets").rstrip("/")
        self.key_id = str(alp.get("api_key_id") or "")
        self.secret_key = str(alp.get("api_secret_key") or "")
        self.timeout = alp.get("timeout_sec") or 10
        self._conn_lost = False
        self._conn_lost_at_monotonic: Optional[float] = None
        self._conn_last_error: str = ""

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def _mark_connection_lost(self, method: str, path: str, exc: Exception) -> None:
        if self._conn_lost:
            return
        self._conn_lost = True
        self._conn_lost_at_monotonic = time.monotonic()
        self._conn_last_error = str(exc)
        logging.error(
            "[LIVE] connectivity_lost broker=alpaca method=%s path=%s error=%s",
            method,
            path,
            exc,
        )

    def _mark_connection_restored(self, method: str, path: str) -> None:
        if not self._conn_lost:
            return
        downtime_sec = 0.0
        if self._conn_lost_at_monotonic is not None:
            downtime_sec = max(0.0, time.monotonic() - self._conn_lost_at_monotonic)
        logging.info(
            "[LIVE] connectivity_restored broker=alpaca method=%s path=%s downtime_sec=%.1f last_error=%s status=stable",
            method,
            path,
            downtime_sec,
            self._conn_last_error or "n/a",
        )
        self._conn_lost = False
        self._conn_lost_at_monotonic = None
        self._conn_last_error = ""

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        use_data_api: bool = False,
    ) -> requests.Response:
        base = self.data_url if use_data_api else self.base_url
        url = f"{base}{path}"
        try:
            resp = requests.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                json=json_body,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            self._mark_connection_lost(method, path, exc)
            raise
        self._mark_connection_restored(method, path)
        return resp

    def ready(self) -> bool:
        return bool(self.key_id and self.secret_key)

    def get_account(self) -> Optional[Dict[str, Any]]:
        if not self.ready():
            return None
        resp = self._request("GET", "/v2/account")
        resp.raise_for_status()
        return resp.json()

    def get_asset(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.ready():
            return None
        sym = str(symbol or "").upper().strip()
        if not sym:
            return None
        resp = self._request("GET", f"/v2/assets/{sym}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def submit_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        resp = self._request("POST", "/v2/orders", json_body=payload)
        resp.raise_for_status()
        return resp.json()

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        if not self.ready():
            return None
        oid = str(order_id or "").strip()
        if not oid:
            return None
        resp = self._request("GET", f"/v2/orders/{oid}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        oid = str(order_id or "").strip()
        if not oid:
            return {}
        resp = self._request("DELETE", f"/v2/orders/{oid}")
        if resp.status_code not in (200, 204):
            resp.raise_for_status()
        if resp.content:
            try:
                return resp.json()
            except Exception:
                return {}
        return {}

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
        if base_price is None:
            base_price = float(entry_price) if entry_price is not None else float(take_profit)
        base_price = float(base_price)
        tp, sl = _sanitize_bracket_prices_from_entry_side(
            str(side or "").lower(),
            base_price,
            float(take_profit),
            float(stop_loss),
        )

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
            payload["limit_price"] = _round_price_for_alpaca(entry_price)
        try:
            return self.submit_order(payload)
        except requests.HTTPError as exc:
            resp = getattr(exc, "response", None)
            if resp is None or int(getattr(resp, "status_code", 0)) != 422:
                raise
            retry_base = _extract_base_price_from_422(resp)
            if retry_base is None or retry_base <= 0:
                raise
            tp_retry, sl_retry = _sanitize_bracket_prices_from_entry_side(
                str(side or "").lower(),
                retry_base,
                float(take_profit),
                float(stop_loss),
            )
            payload["take_profit"] = {"limit_price": tp_retry}
            payload["stop_loss"] = {"stop_price": sl_retry}
            return self.submit_order(payload)

    def submit_oco_exit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        take_profit: float,
        stop_loss: float,
        base_price: float,
        tif: str = "day",
    ) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        base = float(base_price)
        if base <= 0:
            raise ValueError("oco requires positive base_price")
        tp, sl = _sanitize_oco_prices_from_close_side(
            str(side or "").lower(),
            base,
            float(take_profit),
            float(stop_loss),
        )
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "qty": int(qty),
            "type": "limit",
            "limit_price": tp,
            "time_in_force": tif,
            "order_class": "oco",
            "take_profit": {"limit_price": tp},
            "stop_loss": {"stop_price": sl},
        }
        try:
            return self.submit_order(payload)
        except requests.HTTPError as exc:
            resp = getattr(exc, "response", None)
            if resp is None or int(getattr(resp, "status_code", 0)) != 422:
                raise
            retry_base = _extract_base_price_from_422(resp)
            if retry_base is None or retry_base <= 0:
                raise
            tp_retry, sl_retry = _sanitize_oco_prices_from_close_side(
                str(side or "").lower(),
                retry_base,
                float(take_profit),
                float(stop_loss),
            )
            payload["limit_price"] = tp_retry
            payload["take_profit"] = {"limit_price": tp_retry}
            payload["stop_loss"] = {"stop_price": sl_retry}
            return self.submit_order(payload)

    def list_positions(self) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        resp = self._request("GET", "/v2/positions")
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
        params: Dict[str, Any] = {"status": status, "limit": int(limit)}
        if symbols:
            params["symbols"] = ",".join(symbols)
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        resp = self._request("GET", "/v2/orders", params=params)
        resp.raise_for_status()
        return resp.json()

    def cancel_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        if not self.ready():
            raise RuntimeError("Alpaca credentials missing")
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        resp = self._request("DELETE", "/v2/orders", params=params)
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
        resp = self._request("DELETE", f"/v2/positions/{symbol}")
        resp.raise_for_status()
        return resp.json()
