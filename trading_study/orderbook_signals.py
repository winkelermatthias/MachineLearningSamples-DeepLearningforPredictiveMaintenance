"""Limit-order-book signal toolkit (for when you have real L2 data).

This session's network policy blocks every market-data host (LOBSTER, Binance,
Yahoo, Stooq all return 403 at the proxy), so no genuine order-book data could
be obtained. Backtesting microstructure signals on *synthetic* books would be
meaningless — the generator would dictate the "alpha" — so this module ships
only the signal computations, verified against hand-computed examples in
``self_test()``. Feed it real LOBSTER / exchange L2 snapshots to use it.

Signals implemented (canonical references):

- Queue imbalance          — Gould & Bonart (2016); Cartea, Donnelly & Jaimungal
- Microprice               — Stoikov (2018), "The Micro-Price"
- Order-flow imbalance     — Cont, Kukanov & Stoikov (2014), "The Price Impact
                             of Order Book Events"
"""

import numpy as np
import pandas as pd


def queue_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    """I_t = (Qb - Qa) / (Qb + Qa) in [-1, 1]; >0 predicts upward moves."""
    return (bid_size - ask_size) / (bid_size + ask_size)


def microprice(bid_px: pd.Series, bid_size: pd.Series,
               ask_px: pd.Series, ask_size: pd.Series) -> pd.Series:
    """Size-weighted mid: P = (Qb*Pa + Qa*Pb) / (Qb + Qa).

    Leans toward the side with *less* queue (the side about to be consumed),
    a better short-horizon fair-value estimate than the mid-quote.
    """
    return (bid_size * ask_px + ask_size * bid_px) / (bid_size + ask_size)


def order_flow_imbalance(bid_px: pd.Series, bid_size: pd.Series,
                         ask_px: pd.Series, ask_size: pd.Series) -> pd.Series:
    """Cont-Kukanov-Stoikov OFI over successive best-quote snapshots.

    e_t = dQb_contribution - dQa_contribution, where a rising (falling) bid
    price counts the full new (old) bid queue, and symmetrically for the ask.
    Positive OFI = net buying pressure; contemporaneous price impact is
    approximately linear in OFI.
    """
    b_prev, a_prev = bid_px.shift(1), ask_px.shift(1)
    qb_prev, qa_prev = bid_size.shift(1), ask_size.shift(1)

    bid_contrib = pd.Series(np.where(
        bid_px > b_prev, bid_size,
        np.where(bid_px == b_prev, bid_size - qb_prev, -qb_prev)),
        index=bid_px.index)
    ask_contrib = pd.Series(np.where(
        ask_px < a_prev, ask_size,
        np.where(ask_px == a_prev, ask_size - qa_prev, -qa_prev)),
        index=ask_px.index)
    return (bid_contrib - ask_contrib).iloc[1:]


def self_test() -> None:
    """Verify the signals against small hand-computed examples."""
    qi = queue_imbalance(pd.Series([300.0]), pd.Series([100.0]))
    assert np.isclose(qi.iloc[0], 0.5), qi.iloc[0]

    mp = microprice(pd.Series([99.0]), pd.Series([300.0]),
                    pd.Series([101.0]), pd.Series([100.0]))
    # heavy bid queue -> microprice leans toward the ask:
    # (300*101 + 100*99)/400 = 100.5
    assert np.isclose(mp.iloc[0], 100.5), mp.iloc[0]

    # t0 -> t1: bid price up (counts full new bid queue 150), ask unchanged
    # with queue growing by 20 (counts -20). OFI = 150 - 20 = 130.
    ofi = order_flow_imbalance(
        bid_px=pd.Series([99.0, 100.0]), bid_size=pd.Series([100.0, 150.0]),
        ask_px=pd.Series([101.0, 101.0]), ask_size=pd.Series([80.0, 100.0]))
    assert np.isclose(ofi.iloc[0], 130.0), ofi.iloc[0]
    print("orderbook_signals self_test: all assertions passed")


if __name__ == "__main__":
    self_test()
