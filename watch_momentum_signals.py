import os
import requests
import time
import datetime
import logging
import traceback
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# =============================
# GitHub Secrets から環境変数を取得
# =============================
SLACK_WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]
API_KEY = os.environ["API_KEY"]
API_SECRET = os.environ["API_SECRET"]

# =============================
# ログ設定
# =============================
logging.basicConfig(
    filename="momentum_chaser.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# =============================
# Slack通知関数
# =============================
def notify_slack(message: str):
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json={"text": message})
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Slack通知エラー: {e}")
        print(f"Slack通知エラー: {e}")

# =============================
# 銘柄リスト取得
# =============================
def load_tickers(file_path="tickers.csv"):
    try:
        tickers = pd.read_csv(file_path)["code"].astype(str).tolist()
        return tickers
    except Exception as e:
        logging.error(f"ティッカーリスト読み込み失敗: {e}")
        return []

# =============================
# データ取得（yfinance）
# =============================
def fetch_data(ticker: str, period="1mo", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError("データが空です")
        return data
    except Exception as e:
        logging.error(f"{ticker} データ取得失敗: {e}")
        return pd.DataFrame()

# =============================
# モメンタム判定
# =============================
def check_momentum(data: pd.DataFrame, threshold=0.05):
    try:
        if len(data) < 2:
            return False

        latest_close = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[-2]

        change = (latest_close - prev_close) / prev_close
        return change >= threshold
    except Exception as e:
        logging.error(f"モメンタム判定失敗: {e}")
        return False

# =============================
# メイン処理
# =============================
def main():
    logging.info("Momentum Chaser 開始")
    notify_slack("🚀 Momentum Chaser 実行開始")

    tickers = load_tickers()
    if not tickers:
        notify_slack("❌ ティッカーリストが読み込めませんでした")
        return

    for ticker in tqdm(tickers, desc="銘柄スキャン中"):
        data = fetch_data(ticker)
        if data.empty:
            continue

        if check_momentum(data):
            msg = f"🔥 {ticker} がモメンタムシグナルを検出しました"
            notify_slack(msg)
            logging.info(msg)

    notify_slack("✅ Momentum Chaser 実行完了")
    logging.info("Momentum Chaser 完了")

# =============================
# 実行
# =============================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("エラー発生", exc_info=True)
        notify_slack(f"❌ エラー発生: {e}\n{traceback.format_exc()}")
