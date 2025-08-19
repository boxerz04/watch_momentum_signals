# -*- coding: utf-8 -*-
"""
watch_momentum_signals.py (GitHub Actions対応版)
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ========= 環境変数 =========
def _clean_env(val: str | None) -> str | None:
    if val is None:
        return None
    return val.split("#", 1)[0].strip()

SLACK_WEBHOOK_URL = _clean_env(os.getenv("SLACK_WEBHOOK_URL", "")) or ""

_TZ = _clean_env(os.getenv("TZ", "Asia/Tokyo")) or "Asia/Tokyo"
JST = timezone(timedelta(hours=9)) if _TZ in ("Asia/Tokyo", "JST") else timezone.utc

def _f(k, default):
    try:
        v = _clean_env(os.getenv(k, str(default)))
        return float(v if v not in (None, "") else default)
    except Exception:
        return float(default)

def _i(k, default):
    try:
        v = _clean_env(os.getenv(k, str(default)))
        return int(v if v not in (None, "") else default)
    except Exception:
        return int(default)

VOLUME_MULT       = _f("VOLUME_MULT", 1.5)
VOL_QUANTILE      = _f("VOL_QUANTILE", 0.80)
VOL_COND_MODE     = (_clean_env(os.getenv("VOL_COND_MODE", "OR")) or "OR").upper()
if VOL_COND_MODE not in ("OR", "AND"):
    VOL_COND_MODE = "OR"
MIN_TURNOVER_JPY  = _f("MIN_TURNOVER_JPY", 0)
ROC_PERIOD        = _i("ROC_PERIOD", 14)
FIRST_SEEN_BD     = _i("FIRST_SEEN_BD", 10)

REQUEST_TIMEOUT   = 12
LOOKBACK_DAYS     = 240

SCRIPT_DIR = Path(__file__).resolve().parent
NOTIFIED_STATE_FILE = "notified_state.json"

# ========= 依存ライブラリ =========
try:
    import yfinance as yf
except Exception:
    yf = None

# ========= Slack通知 =========
def notify_slack(text: str):
    try:
        if not SLACK_WEBHOOK_URL:
            rec = "[WARN] Slack Webhook 未設定\n" + text
            print(rec)
            return
        url = SLACK_WEBHOOK_URL.strip().strip('"').strip("'")
        res = requests.post(url, json={"text": text}, timeout=REQUEST_TIMEOUT)
        if res.status_code != 200:
            print(f"[SLACK] {datetime.now(JST)} status={res.status_code}")
        else:
            print(f"[SLACK] {datetime.now(JST)} OK")
    except Exception as e:
        print(f"[SLACK EXC] {e}\n{text}")

# ========= 状態（前回ヒット日）管理 =========
def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_json(path: Path, data: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _get_prev_hit_date(symbol: str):
    st_path = SCRIPT_DIR / NOTIFIED_STATE_FILE
    st = _load_json(st_path)
    if isinstance(st.get("last_hits"), dict):
        iso = st["last_hits"].get(symbol)
        if iso:
            try:
                return datetime.strptime(iso[:10], "%Y-%m-%d").date()
            except Exception:
                pass
    return None

def _update_last_hit(symbol: str, trade_date):
    st_path = SCRIPT_DIR / NOTIFIED_STATE_FILE
    st = _load_json(st_path)
    if "last_hits" not in st or not isinstance(st["last_hits"], dict):
        st["last_hits"] = {}
    st["last_hits"][symbol] = trade_date.isoformat()
    _save_json(st_path, st)

def build_first_seen_tag(symbol: str, hit_trade_date) -> str:
    tag = ""
    prev = _get_prev_hit_date(symbol)
    if prev is None:
        tag = "🆕初回(初記録)"
    else:
        try:
            gap_bd = int(np.busday_count(prev, hit_trade_date))
        except Exception:
            gap_bd = None
        if gap_bd is not None and gap_bd >= FIRST_SEEN_BD:
            tag = f"🆕初回(前回: {prev.isoformat()}、経過: {gap_bd}営業日)"
    _update_last_hit(symbol, hit_trade_date)
    return tag

# ========= データ取得 =========
def read_watchlist() -> pd.DataFrame:
    p = SCRIPT_DIR / "watchlist.csv"
    df = pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip()
    elif "code" in df.columns:
        df["symbol"] = df["code"].astype(str).str.strip()
    else:
        raise ValueError("watchlist.csv に 'symbol' か 'code' 列が必要です")

    def _to_symbol(s: str) -> str:
        s = str(s).strip()
        return f"{s}.T" if s.isdigit() and not s.endswith(".T") else s

    df["symbol"] = df["symbol"].apply(_to_symbol)
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    return df[["symbol", "name"]]

def fetch_daily(symbol: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance が必要です")
    start = (datetime.now(tz=JST) - timedelta(days=LOOKBACK_DAYS*2)).strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.dropna()

# ========= 指標計算・判定 =========
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["roc14"] = (out["close"] / out["close"].shift(ROC_PERIOD) - 1.0) * 100.0
    out["vol_ma20_prev"] = out["volume"].rolling(20).mean().shift(1)
    out["hi20_prev"] = out["high"].rolling(20).max().shift(1)
    return out

def judge_signal(dfi: pd.DataFrame) -> (bool, str):
    last = dfi.iloc[-1]
    reasons = []
    cond_roc = bool(pd.notna(last.get("roc14")) and last["roc14"] > 0)
    if cond_roc:
        reasons.append(f"ROC14>0 ({last['roc14']:.2f}%)")
    cond_vol = bool(last["volume"] >= last.get("vol_ma20_prev", 0) * VOLUME_MULT)
    if cond_vol:
        reasons.append("Volume Spike")
    cond_hi = bool(pd.notna(last.get("hi20_prev")) and last["close"] > last["hi20_prev"])
    if cond_hi:
        reasons.append("20d High Break")
    ok = cond_roc and cond_vol and cond_hi
    return ok, ", ".join(reasons)

def build_hit_message(symbol: str, name: str, last: pd.Series, reasons: str, header_extra: str = "") -> str:
    yj = f"https://finance.yahoo.co.jp/quote/{symbol}"
    return "\n".join([
        f"🔴 モメンタム候補: {symbol} {name}",
        header_extra,
        f"終値: {last['close']:.2f}, ROC14: {last['roc14']:.2f}%",
        f"出来高: {int(last['volume'])}, 20d前高値: {last['hi20_prev']:.2f}",
        f"条件: {reasons}",
        f"🔗 {yj}"
    ])

# ========= メイン =========
def main():
    try:
        wl = read_watchlist()
    except Exception as e:
        notify_slack(f"[ERROR] watchlist読み込み失敗: {e}")
        return
    hits = 0
    for _, r in wl.iterrows():
        symbol, name = r["symbol"], r["name"]
        try:
            df = fetch_daily(symbol)
            if df.empty: continue
            dfi = calc_indicators(df)
            ok, reasons = judge_signal(dfi)
            if ok:
                last = dfi.iloc[-1]
                tag = build_first_seen_tag(symbol, last["date"].date())
                msg = build_hit_message(symbol, name, last, reasons, header_extra=tag)
                print(msg)
                notify_slack(msg)
                hits += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"{symbol}: {e}")
    if hits == 0:
        notify_slack("⚠️ 本日一致銘柄なし")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
        notify_slack(f"❌ エラー発生:\n{traceback.format_exc()}")
