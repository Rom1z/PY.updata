import sys
import re
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
import ccxt

# --- Универсальный импорт Qt ---
try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
        QGroupBox, QGridLayout
    )
    qt_lib = "PyQt5"
except ImportError:
    from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
        QGroupBox, QGridLayout
    )
    qt_lib = "PySide6"

# --- Signal Result ---
@dataclass
class SignalResult:
    side: str
    confidence: float
    reason: str
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    sl: Optional[float]

# --- Exchanges ---
SUPPORTED_EXCHANGES = {
    "binance": ccxt.binance(),
    "bybit": ccxt.bybit(),
    "okx": ccxt.okx(),
}

# --- Indicators ---
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = ema(macd_line, 9)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def swing_levels(df: pd.DataFrame, lookback: int = 14):
    recent = df.tail(lookback)
    return float(recent["high"].max()), float(recent["low"].min())

# --- Heuristic signal ---
def heuristic_signal(df: pd.DataFrame) -> SignalResult:
    last = df.iloc[-1]; prev = df.iloc[-2]
    score, reasons = 0.0, []

    # EMA trend
    if last["ema50"] > last["ema200"]:
        score += 0.25; reasons.append("EMA50 > EMA200 (uptrend)")
    else:
        score -= 0.25; reasons.append("EMA50 < EMA200 (downtrend)")

    # MACD cross
    macd_cross_up = prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]
    macd_cross_dn = prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]
    if macd_cross_up:
        score += 0.3; reasons.append("MACD bullish cross")
    if macd_cross_dn:
        score -= 0.3; reasons.append("MACD bearish cross")

    # MACD momentum
    hist_slope = (last["macd"] - last["macd_signal"]) - (prev["macd"] - prev["macd_signal"])
    if hist_slope > 0:
        score += 0.15; reasons.append("MACD momentum rising")
    else:
        score -= 0.15; reasons.append("MACD momentum falling")

    # RSI filter
    if last["rsi14"] > 55:
        score += 0.2; reasons.append("RSI > 55")
    elif last["rsi14"] < 45:
        score -= 0.2; reasons.append("RSI < 45")

    score = max(-1.0, min(1.0, score))
    side = "LONG" if score > 0.15 else ("SHORT" if score < -0.15 else "NEUTRAL")
    confidence = float(abs(score))

    atr_val = float(last.get("atr14", np.nan))
    price = float(last["close"])
    swing_hi, swing_lo = swing_levels(df, lookback=14)

    tp1 = tp2 = tp3 = sl = None
    if side == "LONG":
        sl = max(price - atr_val * 1.2, swing_lo or price - atr_val * 1.2)
        tp1 = price + atr_val * 1.0
        tp2 = price + atr_val * 1.5
        tp3 = min(price + atr_val * 2.0, swing_hi or price + atr_val * 2.0)
    elif side == "SHORT":
        sl = min(price + atr_val * 1.2, swing_hi or price + atr_val * 1.2)
        tp1 = price - atr_val * 1.0
        tp2 = price - atr_val * 1.5
        tp3 = max(price - atr_val * 2.0, swing_lo or price - atr_val * 2.0)

    return SignalResult(side, confidence, "; ".join(reasons), tp1, tp2, tp3, sl)

# --- Symbol normalization ---
def normalize_symbol(user_input: str) -> str:
    s = user_input.strip().upper()
    s = re.sub(r"[.\-_/]", "", s)
    s = re.sub(r"(USDT|USD|USDC|PERP|SWAP)$", "/USDT", s)
    return s if "/" in s else s + "/USDT"

def resolve_symbol(user_input: str, markets: Dict[str, dict]) -> Optional[str]:
    norm = normalize_symbol(user_input)
    if norm in markets:
        return norm
    base = norm.replace("/USDT", "")
    for sym in markets:
        if sym.startswith(base):
            return sym
    return None

# --- Worker Thread ---
class FetchWorker(QThread):
    finished = pyqtSignal(pd.DataFrame, SignalResult, str)

    def __init__(self, exchange: ccxt.Exchange, symbol: str, timeframe: str):
        super().__init__()
        self.exchange = exchange; self.symbol = symbol; self.timeframe = timeframe

    def run(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
            df["ema50"] = ema(df["close"], 50)
            df["ema200"] = ema(df["close"], 200)
            df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
            df["rsi14"] = rsi(df["close"], 14)
            df["atr14"] = atr(df, 14)
            sig = heuristic_signal(df)
            self.finished.emit(df, sig, self.symbol)
        except Exception as e:
            print("Error fetching:", e)

# --- Main Window ---
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto TA Desktop App")
        self.setGeometry(200, 200, 700, 500)

        self.markets: Dict[str, dict] = {}
        self.current_exchange: Optional[ccxt.Exchange] = None

        vbox = QVBoxLayout(self)

        # --- Controls ---
        controls = QHBoxLayout()
        self.cmb_exchange = QComboBox(); self.cmb_exchange.addItems(SUPPORTED_EXCHANGES.keys())
        self.cmb_tf = QComboBox(); self.cmb_tf.addItems(["1m","5m","10m","15m","30m","1h","4h","1d"])
        self.cmb_market_type = QComboBox(); self.cmb_market_type.addItems(["spot","swap"])
        self.ed_symbol = QLineEdit(); self.ed_symbol.setPlaceholderText("Enter symbol (e.g., XAUUSD)")
        self.btn_load = QPushButton("Load Markets")
        self.btn_fetch = QPushButton("Fetch Signal")

        controls.addWidget(QLabel("Exchange:")); controls.addWidget(self.cmb_exchange)
        controls.addWidget(QLabel("TF:")); controls.addWidget(self.cmb_tf)
        controls.addWidget(QLabel("Type:")); controls.addWidget(self.cmb_market_type)
        controls.addWidget(self.ed_symbol)
        controls.addWidget(self.btn_load); controls.addWidget(self.btn_fetch)
        vbox.addLayout(controls)

        # --- Signal box ---
        box = QGroupBox("Signal Result"); grid = QGridLayout(box)
        self.lbl_resolved = QLabel("—")
        self.lbl_side = QLabel("—")
        self.lbl_conf = QLabel("—")
        self.lbl_tp1 = QLabel("—")
        self.lbl_tp2 = QLabel("—")
        self.lbl_tp3 = QLabel("—")
        self.lbl_sl = QLabel("—")
        self.lbl_reason = QTextEdit(); self.lbl_reason.setReadOnly(True)

        grid.addWidget(QLabel("Resolved:"), 0,0); grid.addWidget(self.lbl_resolved,0,1)
        grid.addWidget(QLabel("Side:"),1,0); grid.addWidget(self.lbl_side,1,1)
        grid.addWidget(QLabel("Conf:"),2,0); grid.addWidget(self.lbl_conf,2,1)
        grid.addWidget(QLabel("TP1:"),3,0); grid.addWidget(self.lbl_tp1,3,1)
        grid.addWidget(QLabel("TP2:"),4,0); grid.addWidget(self.lbl_tp2,4,1)
        grid.addWidget(QLabel("TP3:"),5,0); grid.addWidget(self.lbl_tp3,5,1)
        grid.addWidget(QLabel("SL:"),6,0); grid.addWidget(self.lbl_sl,6,1)
        grid.addWidget(QLabel("Reasons:"),7,0); grid.addWidget(self.lbl_reason,7,1,3,1)
        vbox.addWidget(box)

        # --- Events ---
        self.btn_load.clicked.connect(self.load_markets)
        self.btn_fetch.clicked.connect(self.fetch_signal)

    def load_markets(self):
        exch_name = self.cmb_exchange.currentText()
        self.current_exchange = SUPPORTED_EXCHANGES[exch_name]
        try:
            self.current_exchange.load_markets()
            self.markets = self.current_exchange.markets
            print(f"Loaded {len(self.markets)} symbols for {exch_name}")
        except Exception as e:
            print("Error loading markets:", e)

    def fetch_signal(self):
        if not self.current_exchange:
            return
        sym = self.ed_symbol.text()
        resolved = resolve_symbol(sym, self.markets)
        if not resolved:
            self.lbl_resolved.setText("Symbol not found")
            return
        tf = self.cmb_tf.currentText()
        # --- Bybit timeframe mapping ---
        tf_map_bybit = {"1m":"1","5m":"5","10m":"10","15m":"15","30m":"30","1h":"60","4h":"240","1d":"D"}
        if self.current_exchange.id == "bybit":
            tf = tf_map_bybit.get(tf, tf)

        self.worker = FetchWorker(self.current_exchange, resolved, tf)
        self.worker.finished.connect(self.on_fetched)
        self.worker.start()

    def on_fetched(self, df: pd.DataFrame, sig: SignalResult, resolved_symbol: str):
        self.lbl_resolved.setText(resolved_symbol or "—")
        self.lbl_side.setText(sig.side)
        self.lbl_conf.setText(f"{sig.confidence:.2f}")
        self.lbl_tp1.setText(f"{sig.tp1:.2f}" if sig.tp1 else "—")
        self.lbl_tp2.setText(f"{sig.tp2:.2f}" if sig.tp2 else "—")
        self.lbl_tp3.setText(f"{sig.tp3:.2f}" if sig.tp3 else "—")
        self.lbl_sl.setText(f"{sig.sl:.2f}" if sig.sl else "—")
        self.lbl_reason.setPlainText(sig.reason)

# --- Entry Point ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App(); win.show()
    sys.exit(app.exec())