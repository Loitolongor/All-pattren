"""# scripts/candles pattern.py
"""
"""
A small candlestick pattern detector for OHLC(C) CSVs.

Creates a DataFrame with detected patterns per row. Supports:
- Doji
- Hammer / Hanging Man (single-candle body small + long lower wick)
- Bullish Engulfing / Bearish Engulfing (two-candle patterns)

Input CSV: columns Date, Open, High, Low, Close (case-insensitive). Date is optional for detection but recommended.

Usage:
    python "scripts/candles pattern.py" input.csv --output detections.csv

This is intentionally dependency-light: requires pandas and optionally matplotlib for plotting.

"""

from __future__ import annotations
import argparse
import sys
from typing import Optional

import pandas as pd

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLC columns exist and normalize names to O,H,L,C."""
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for name in ("open", "high", "low", "close"):
        if name in cols:
            mapping[cols[name]] = name[0].upper()
        else:
            raise KeyError(f"Input CSV must contain column '{name}' (case-insensitive)")
    df = df.rename(columns=mapping)
    return df

def detect_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """Detect Doji candles.

    A candle is marked Doji when the body (absolute(Open - Close)) is smaller than
    threshold * range(High - Low). threshold default 0.1 means body < 10% of range.
    Returns a boolean Series.
    """
    body = (df["O"] - df["C"]).abs()
    rng = (df["H"] - df["L"]).abs().replace(0, pd.NA)
    is_doji = body < (threshold * rng)
    return is_doji.fillna(False)

def detect_hammer(df: pd.DataFrame, body_to_range: float = 0.25, lower_wick_min: float = 2.0) -> pd.Series:
    """Detect Hammer / Hanging Man-like shapes.

    Criteria (simple heuristic):
    - Small body relative to range: body <= body_to_range * range
    - Lower wick length at least lower_wick_min times the body
    - Upper wick small: upper_wick <= body * lower_wick_min (helps exclude long upper wicks)

    This returns True for both hammer (bullish-looking) and hanging man (bearish-looking).
    """
    body = (df["O"] - df["C"]).abs()
    upper_wick = (df["H"] - df[["O", "C"]].max(axis=1)).abs()
    lower_wick = (df[["O", "C"].min(axis=1) - df["L"]]).abs()
    rng = (df["H"] - df["L"]).abs().replace(0, pd.NA)

    small_body = body <= (body_to_range * rng)
    strong_lower = lower_wick >= (lower_wick_min * body.clip(lower=1e-9))
    small_upper = upper_wick <= (body * lower_wick_min)

    return (small_body & strong_lower & small_upper).fillna(False)

def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detect Bullish and Bearish Engulfing patterns.

    Returns a Series with values: '', 'Bullish Engulfing', 'Bearish Engulfing'.
    The pattern checks two consecutive candles (prev, curr):
      - Bullish Engulfing: prev is bearish (close < open) and curr is bullish (close > open)
        and curr's body fully engulfs prev's body (curr.open <= prev.close and curr.close >= prev.open)
      - Bearish Engulfing: prev bullish and curr bearish with opposite engulfing relation.
    """
    prev = df.shift(1)
    prev_bear = prev["C"] < prev["O"]
    prev_bull = prev["C"] > prev["O"]
    curr_bear = df["C"] < df["O"]
    curr_bull = df["C"] > df["O"]

    be_mask = prev_bear & curr_bull & (df["O"] <= prev["C"]) & (df["C"] >= prev["O"])
    se_mask = prev_bull & curr_bear & (df["O"] >= prev["C"]) & (df["C"] <= prev["O"])

    out = pd.Series([""] * len(df), index=df.index)
    out[be_mask.fillna(False)] = "Bullish Engulfing"
    out[se_mask.fillna(False)] = "Bearish Engulfing"
    return out

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with additional columns describing detected patterns."""
    df2 = df.copy()
    df2["Doji"] = detect_doji(df2)
    df2["HammerLike"] = detect_hammer(df2)
    df2["Engulfing"] = detect_engulfing(df2)

    # Combine human-friendly single 'Pattern' column
    def _combine(row):
        parts = []
        if row["Doji"]:
            parts.append("Doji")
        if row["HammerLike"]:
            parts.append("HammerLike")
        if row["Engulfing"]:
            parts.append(row["Engulfing"])
        return ", ".join(parts)

    df2["Pattern"] = df2.apply(_combine, axis=1)
    return df2

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detect common candlestick patterns in an OHLC CSV")
    parser.add_argument("input", help="Path to input CSV (Date, Open, High, Low, Close)")
    parser.add_argument("--output", "-o", help="Path to write CSV with detections (default: stdout)")
    parser.add_argument("--plot", action="store_true", help="Attempt to plot last 50 candles (requires mplfinance)")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    try:
        df = _normalize_cols(df)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 2

    out = detect_patterns(df)

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Wrote detections to {args.output}")
    else:
        print(out.to_csv(index=False))

    if args.plot:
        try:
            import mplfinance as mpf
            plot_df = df.rename(columns={"O": "Open", "H": "High", "L": "Low", "C": "Close"})
            plot_df.index = pd.to_datetime(plot_df.index) if plot_df.index.dtype == object else plot_df.index
            mpf.plot(plot_df.tail(50), type="candle", style="yahoo", title="Last 50 candles")
        except Exception as e:
            print("Plot failed (mplfinance required):", e, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
