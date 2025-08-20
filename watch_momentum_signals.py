# -*- coding: utf-8 -*-
"""
watch_momentum_signals.py (GitHub Actions 対応・安全版)

- watchlist.csv を読み込み、各銘柄について以下の “モメンタム条件” を評価
  1) ROC(ROC_PERIOD) > 0
  2) 出来高: 直近20日平均(前日まで)×VOLUME_MULT との倍率条件
              と 昨日まで60本の出来高分位 VOL_QUANTILE (例: 0.80=80%) の
              OR / AND 判定 (VOL_COND_MODE)
     さらに MIN_TURNOVER_JPY（売買代金しきい値）を満たす場合のみ有効
  3) 20日高値(前日まで) を「終値」で上抜け

- Slack へ通知（Incoming Webhook）
- 「🆕初回(前回: YYYY-MM-DD、経過: N営業日)」タグをタイトル直下に表示
  * 前回ヒット日は notified_state.json の last_hits[symbol] に “取引日” を保存
  * GitHub Actions はエフェメラル環境のため、このJSONは毎回リセットされます
    （永続化したい場合は Artifacts やリポジトリコミット等をご検討ください）

環境変数（GitHub Secrets で設定推奨）
  SLACK_WEBHOOK_URL      : Slack Incoming Webhook のURL
  VOLUME_MULT            : 出来高倍率 (default 1.5)
  VOL_QUANTILE           : 分位（0.0〜1.0, default 0.80）
  VOL_COND_MODE          : "OR" or "AND"（倍率と分位の結合方法, default "OR"）
  MIN_TURNOVER_JPY       : 最低売買代金（円, default 0）
  ROC_PERIOD             : ROCの期間 (default 14)
  FIRST_SEEN_BD          : “初回”と表示する営業日ギャップ (default 10)
  TZ                     : タイムゾーン（"Asia/Tokyo" / "JST" など）
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

# ========= 環境パラメータ =========
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
VOL_QUANTILE      = _f("VOL_QUANTILE", 0.80)         # 0.0〜1.0
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
            print("[WARN] Slack Webhook 未設定\n" + text)
            return
        url = SLACK_WEBHOOK_URL.strip().strip('"').strip("'")
        res = requests.post(url, json={"text": text}, timeout=REQUEST_TIMEOUT)
        if res.status_code != 200:
            print(f"[SLACK] {datetime.now(JST)} status={res.status_code} body={res.text[:200]}")
        else:
            print(f"[SLACK] {datetime.now(JST)} OK len={len(text)}")
    except Exception as e:
        print(f"[SLACK EXC] {e}\n{text}")

# ========= JSONユーティリティ（前回ヒット保存） =========
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
    # （旧形式サポートを省略：Actions では永続化しないため通常不要）
    return None

def _update_last_hit(symbol: str, trade_date):
    st_path = SCRIPT_DIR / NOTIFIED_STATE_FILE
    st = _load_json(st_path)
    if "last_hits" not in st or not isinstance(st["last_hits"], dict):
        st["last_hits"] = {}
    st["last_hits"][symbol] = trade_date.isoformat()
    _save_json(st_path, st)

def build_first_seen_tag(symbol: str, hit_trade_date) -> str:
    """
    直近 FIRST_SEEN_BD 営業日以上あけば「🆕初回(前回: YYYY-MM-DD、経過: N営業日)」
    前回記録が無い場合は「🆕初回(初記録)」
    """
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

# ========= yfinance ヘルパ =========
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance が MultiIndex 列を返す場合に単層化する。
    列名タプルから 'Open','High','Low','Close','Adj Close','Volume' のいずれかを優先採用。
    """
    if isinstance(df.columns, pd.MultiIndex):
        keys = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        new_cols = []
        for col in df.columns.to_list():
            chosen = None
            if isinstance(col, tuple):
                for tok in col:
                    t = str(tok)
                    if t in keys:
                        chosen = t
                        break
                if chosen is None:
                    chosen = str(col[0])
            else:
                chosen = str(col)
            new_cols.append(chosen)
        df = df.copy()
        df.columns = new_cols
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]

    # 'Close' が無く 'Adj Close' のみなら 'Close' に寄せる
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    # 余分な列は落とす（存在するものだけ）
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep]

def fetch_daily(symbol: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance が必要です（requirements.txt に yfinance を追加）")

    start = (datetime.now(tz=JST) - timedelta(days=LOOKBACK_DAYS*2)).strftime("%Y-%m-%d")

    # MultiIndex の揺れ対策として group_by='column' を明示
    df = yf.download(
        symbol,
        start=start,
        interval="1d",
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # フラット化 → indexを列化 → 小文字化
    df = _flatten_yf_columns(df).reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # 必要列の存在確認
    need = {"date", "open", "high", "low", "close", "volume"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return out

# ========= ウォッチリスト =========
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

# ========= 指標計算・判定 =========
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["roc14"] = (out["close"] / out["close"].shift(ROC_PERIOD) - 1.0) * 100.0
    out["vol_ma20_prev"] = out["volume"].rolling(20).mean().shift(1)    # 昨日まで
    out["hi20_prev"]     = out["high"].rolling(20).max().shift(1)       # 昨日まで
    return out

def judge_volume_block(dfi: pd.DataFrame) -> tuple[bool, str]:
    last = dfi.iloc[-1]

    # 倍率条件
    cond_ratio = False
    if pd.notna(last.get("vol_ma20_prev")):
        try:
            cond_ratio = float(last["volume"]) >= float(last["vol_ma20_prev"]) * VOLUME_MULT
        except Exception:
            cond_ratio = False

    # 分位条件（昨日まで60本の分布で判定）
    cond_quant = False
    vol_pq = float("nan")
    try:
        vol_hist = dfi["volume"].iloc[:-1].tail(60).dropna().astype(float)
        if len(vol_hist) > 0:
            vol_pq = float(np.percentile(vol_hist, VOL_QUANTILE * 100.0))
            cond_quant = float(last["volume"]) >= vol_pq
    except Exception:
        cond_quant = False

    cond_vol = (cond_ratio and cond_quant) if VOL_COND_MODE == "AND" else (cond_ratio or cond_quant)

    # 売買代金しきい値
    if MIN_TURNOVER_JPY > 0:
        try:
            turnover = float(last["close"]) * float(last["volume"])
            cond_vol = cond_vol and (turnover >= MIN_TURNOVER_JPY)
        except Exception:
            cond_vol = False

    ratio_txt = f"MA20(prev)×{VOLUME_MULT:.2f}" if pd.notna(last.get("vol_ma20_prev")) else "MA20(prev) NA"
    pq_txt = f"P{int(VOL_QUANTILE*100)}" if not np.isnan(vol_pq) else "Pq NA"
    reason = f"Vol[{VOL_COND_MODE}]: ratio({ratio_txt}) / quantile({pq_txt})"
    return cond_vol, reason

def judge_signal(dfi: pd.DataFrame) -> tuple[bool, str]:
    last = dfi.iloc[-1]
    reasons = []

    # ROC
    cond_roc = bool(pd.notna(last.get("roc14")) and last["roc14"] > 0)
    if cond_roc:
        reasons.append(f"ROC{ROC_PERIOD}>0 ({last['roc14']:.2f}%)")

    # 出来高
    cond_vol, vol_reason = judge_volume_block(dfi)
    if cond_vol:
        reasons.append(vol_reason)

    # 20日高値（前日まで）を終値でブレイク
    cond_hi = bool(pd.notna(last.get("hi20_prev")) and last["close"] > last["hi20_prev"])
    if cond_hi:
        reasons.append("20d High Break (close>prev)")

    ok = bool(cond_roc and cond_vol and cond_hi)
    return ok, ", ".join(reasons)

# ========= 通知メッセージ =========
def build_hit_message(symbol: str, name: str, last: pd.Series, reasons: str, header_extra: str | None = None) -> str:
    yj = f"https://finance.yahoo.co.jp/quote/{symbol}"

    # 体裁
    date_str = datetime.now(JST).strftime("%Y-%m-%d")
    time_str = datetime.now(JST).strftime("%H:%M")

    ma20_prev_str = f"{int(last['vol_ma20_prev']):,}" if pd.notna(last.get("vol_ma20_prev")) else "NA"

    lines = [f"🔴 モメンタム候補: {symbol} {name}"]
    if header_extra:
        lines.append(header_extra)

    lines += [
        f"日付: {date_str}  判定: {time_str} JST",
        f"終値: {float(last['close']):.2f}  ROC{ROC_PERIOD}: {float(last['roc14']):.2f}%",
        f"出来高: {int(last['volume']):,} / MA20(prev): {ma20_prev_str}",
        f"20d前高値(前日まで): {float(last['hi20_prev']):.2f}",
        f"条件: {reasons}",
        f"🔗 {yj}",
    ]
    return "\n".join(lines)

# ========= メイン =========
def main():
    try:
        wl = read_watchlist()
    except Exception as e:
        notify_slack(f"[ERROR] watchlist読み込み失敗: {e}")
        return

    hits = 0
    for _, r in wl.iterrows():
        symbol = str(r["symbol"]).strip()
        name   = str(r["name"]).strip()
        try:
            df = fetch_daily(symbol)
            if df.empty or len(df) < 50:
                print(f"{symbol}: データなし/期間不足")
                continue

            dfi = calc_indicators(df)
            ok, reasons = judge_signal(dfi)
            if ok:
                last = dfi.iloc[-1]
                # '取引日'（= データ行の日付）で「初回」判定・保存
                trade_date = last["date"].date()
                tag = build_first_seen_tag(symbol, trade_date)
                header_extra = tag if tag else None

                msg = build_hit_message(symbol, name, last, reasons, header_extra=header_extra)
                print(msg)
                notify_slack(msg)
                hits += 1

            time.sleep(0.2)  # API 負荷を下げる
        except Exception as e:
            print(f"{symbol}: {e}")

    if hits == 0:
        notify_slack(f"⚠️ {datetime.now(JST).strftime('%Y-%m-%d %H:%M')} JST 一致銘柄なし")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err)
        notify_slack(f"❌ エラー発生:\n{err}")
