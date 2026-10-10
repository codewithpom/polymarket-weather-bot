"""
Order Execution Router (Paper Trading vs Live CLOB Execution).

Ensures all orders pass through the mechanical three-lock safety gate
before reaching Polymarket's CLOB.
"""

from __future__ import annotations
import os
import sqlite3
import src.paper as paper
import src.risk as risk
from src.config import LIVE, FORCE_LIVE
from src.fees import taker_fee


def execute_order(
    con: sqlite3.Connection,
    strategy: str,
    market_title: str,
    side: str,
    price: float,
    model_prob: float,
    stake: float,
    market_id: str = "",
    token_id: str = "",
    token_index: int = 0,
    meta: dict | None = None,
    is_live: bool = False,
) -> dict:
    """
    Routes order to Paper Trading ledger or Live CLOB client.
    """
    shares = stake / price
    fee_est = taker_fee(shares, price, "weather")

    if not is_live or not LIVE:
        # Paper Execution
        tid = paper.log_trade(
            con=con,
            strategy=strategy,
            market=market_title,
            side=side,
            price=price,
            model_prob=model_prob,
            size=shares,
            fee_est=fee_est,
            market_id=market_id,
            token_index=token_index,
            meta=meta,
        )
        return {
            "mode": "paper",
            "trade_id": tid,
            "shares": shares,
            "price": price,
            "stake": stake,
            "fee_est": fee_est,
            "status": "logged",
        }

    # Live Execution (Strict Gates)
    risk.require_live(con, strategy)

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType
    except ImportError:
        raise RuntimeError("py-clob-client not installed. Cannot place live orders.")

    client = ClobClient(
        host="https://clob.polymarket.com",
        key=os.environ["POLY_PRIVATE_KEY"],
        chain_id=137,  # Polygon Mainnet
    )

    # Round price and size to Polymarket tick specs
    rounded_price = round(price, 2)
    rounded_shares = round(shares, 2)

    order_args = OrderArgs(
        price=rounded_price,
        size=rounded_shares,
        side=side.upper(),
        token_id=token_id,
    )
    resp = client.create_order(order_args)
    order_id = resp.get("orderID", "")

    # Log fill to local database
    tid = paper.log_trade(
        con=con,
        strategy=strategy,
        market=market_title,
        side=side,
        price=rounded_price,
        model_prob=model_prob,
        size=rounded_shares,
        fee_est=fee_est,
        market_id=market_id,
        token_index=token_index,
        meta={"clob_order_id": order_id, **(meta or {})},
    )

    return {
        "mode": "live",
        "trade_id": tid,
        "clob_order_id": order_id,
        "shares": rounded_shares,
        "price": rounded_price,
        "stake": stake,
        "status": "placed",
    }
