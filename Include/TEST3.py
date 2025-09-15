import sys
import re
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd
import ccxt
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
    QGroupBox, QGridLayout
)

# --- Data class для сигналов ---
@dataclass
class SignalResult:
    side: str
    confidence: float
    reason: str
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    sl: Optional[float]

# --- Биржи ---
SUPPORTED_EXCHANGES = {
    "binance": ccxt.binance(),
    "bybit": ccxt.bybit(),
    "okx": ccxt.okx(),
}

# --- Индикаторы ---
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

def stoch_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    rsi_val = rsi(series, period)
    stoch = (rsi_val - rsi_val.rolling(period).min()) / (rsi_val.rolling(period).max() - rsi_val.rolling(period).min())
    return stoch * 100

# --- AI Анализ сигнала ---
def ai_signal(df: pd.DataFrame) -> SignalResult:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score, reasons = 0.0, []

    # EMA тренд
    if last["ema50"] > last["ema200"]:
        score += 0.3; reasons.append("Uptrend detected (EMA50 > EMA200)")
    else:
        score -= 0.3; reasons.append("Downtrend detected (EMA50 < EMA200)")

    # MACD
    macd_cross_up = prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]
    macd_cross_dn = prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]
    if macd_cross_up: score += 0.2; reasons.append("MACD bullish cross")
    if macd_cross_dn: score -= 0.2; reasons.append("MACD bearish cross")

    # RSI
    if last["rsi14"] > 60: score += 0.1; reasons.append("RSI indicates strength")
    elif last["rsi14"] < 40: score -= 0.1; reasons.append("RSI indicates weakness")

    # StochRSI
    stoch = last.get("stoch_rsi",50)
    if stoch < 20: score += 0.1; reasons.append("StochRSI oversold")
    elif stoch > 80: score -= 0.1; reasons.append("StochRSI overbought")

    # ATR для TP/SL
    atr_val = float(last.get("atr14", np.nan))
    price = float(last["close"])
    swing_hi, swing_lo = float(df.tail(14)["high"].max()), float(df.tail(14)["low"].min())

    tp1 = tp2 = tp3 = sl = None
    if score > 0.15:  # LONG
        sl = max(price - atr_val*1.2, swing_lo)
        tp1 = price + atr_val*1.0
        tp2 = price + atr_val*1.5
        tp3 = min(price + atr_val*2.0, swing_hi)
        side = "LONG"
    elif score < -0.15:  # SHORT
        sl = min(price + atr_val*1.2, swing_hi)
        tp1 = price - atr_val*1.0
        tp2 = price - atr_val*1.5
        tp3 = max(price - atr_val*2.0, swing_lo)
        side = "SHORT"
    else:
        side = "NEUTRAL"

    confidence = float(abs(score))
    return SignalResult(side, confidence, "; ".join(reasons), tp1, tp2, tp3, sl)

# --- Нормализация символа ---
def normalize_symbol(user_input: str) -> str:
    s = user_input.strip().upper()
    s = re.sub(r"[.\-_/]", "", s)
    s = re.sub(r"(USDT|USD|USDC|PERP|SWAP)$","/USDT", s)
    return s if "/" in s else s + "/USDT"

def resolve_symbol(user_input: str, markets: Dict[str,dict]) -> Optional[str]:
    norm = normalize_symbol(user_input)
    if norm in markets: return norm
    base = norm.replace("/USDT","")
    for sym in markets:
        if sym.startswith(base): return sym
    return None

# --- Поток загрузки данных ---
class FetchWorker(QThread):
    finished = Signal(pd.DataFrame, SignalResult, str)
    def __init__(self, exchange: ccxt.Exchange, symbol: str, timeframe: str):
        super().__init__()
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

    def run(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
            df["ema50"] = ema(df["close"],50)
            df["ema200"] = ema(df["close"],200)
            df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
            df["rsi14"] = rsi(df["close"],14)
            df["atr14"] = atr(df,14)
            df["stoch_rsi"] = stoch_rsi(df["close"],14)
            sig = ai_signal(df)
            self.finished.emit(df,sig,self.symbol)
        except Exception as e:
            print("Error fetching:",e)

# --- Главное окно ---
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto/Forex TA AI App")
        self.setGeometry(200,200,900,700)
        self.markets: Dict[str,dict]={}
        self.current_exchange: Optional[ccxt.Exchange]=None
        vbox = QVBoxLayout(self)

        # --- Контролы ---
        controls = QHBoxLayout()
        self.cmb_exchange = QComboBox(); self.cmb_exchange.addItems(SUPPORTED_EXCHANGES.keys())
        self.cmb_tf = QComboBox(); self.cmb_tf.addItems(["1m","5m","10m","15m","30m","1h","4h","1d"])
        self.ed_symbol = QLineEdit(); self.ed_symbol.setPlaceholderText("Symbol (XAUUSD)")
        self.btn_load = QPushButton("Load Markets")
        controls.addWidget(QLabel("Exchange:")); controls.addWidget(self.cmb_exchange)
        controls.addWidget(QLabel("TF:")); controls.addWidget(self.cmb_tf)
        controls.addWidget(self.ed_symbol); controls.addWidget(self.btn_load)
        vbox.addLayout(controls)

        # --- Результаты ---
        box = QGroupBox("Signal Result"); grid = QGridLayout(box)
        self.lbl_resolved = QLabel("—"); self.lbl_side = QLabel("—"); self.lbl_conf = QLabel("—")
        self.lbl_tp1 = QLabel("—"); self.lbl_tp2 = QLabel("—"); self.lbl_tp3 = QLabel("—")
        self.lbl_sl = QLabel("—"); self.lbl_reason = QTextEdit(); self.lbl_reason.setReadOnly(True)
        grid.addWidget(QLabel("Resolved:"),0,0); grid.addWidget(self.lbl_resolved,0,1)
        grid.addWidget(QLabel("Side:"),1,0); grid.addWidget(self.lbl_side,1,1)
        grid.addWidget(QLabel("Conf:"),2,0); grid.addWidget(self.lbl_conf,2,1)
        grid.addWidget(QLabel("TP1:"),3,0); grid.addWidget(self.lbl_tp1,3,1)
        grid.addWidget(QLabel("TP2:"),4,0); grid.addWidget(self.lbl_tp2,4,1)
        grid.addWidget(QLabel("TP3:"),5,0); grid.addWidget(self.lbl_tp3,5,1)
        grid.addWidget(QLabel("SL:"),6,0); grid.addWidget(self.lbl_sl,6,1)
        grid.addWidget(QLabel("Reasons:"),7,0); grid.addWidget(self.lbl_reason,7,1,3,1)
        vbox.addWidget(box)

        # --- График ---
        self.chart_canvas = FigureCanvas(Figure(figsize=(6,4)))
        vbox.addWidget(self.chart_canvas)

        # --- События ---
        self.btn_load.clicked.connect(self.load_markets)

        # --- Таймер реального времени ---
        self.timer = QTimer(); self.timer.timeout.connect(self.fetch_signal)
        self.timer.start(5000)

    def load_markets(self):
        exch_name = self.cmb_exchange.currentText()
        self.current_exchange = SUPPORTED_EXCHANGES[exch_name]
        try:
            self.current_exchange.load_markets()
            self.markets = self.current_exchange.markets
            print(f"Loaded {len(self.markets)} symbols for {exch_name}")
        except Exception as e:
            print("Error loading markets:",e)

    def fetch_signal(self):
        if not self.current_exchange: return
        sym = self.ed_symbol.text()
        resolved = resolve_symbol(sym,self.markets) or sym.upper()
        tf = self.cmb_tf.currentText()
        self.worker = FetchWorker(self.current_exchange,resolved,tf)
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

        # --- График ---
        fig = self.chart_canvas.figure; fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(df["close"], color="black", label="Close")
        last_index = len(df)-1
        if sig.side=="LONG": ax.scatter(last_index, df["close"].iloc[-1], color="green", s=100, zorder=5, label="LONG")
        elif sig.side=="SHORT": ax.scatter(last_index, df["close"].iloc[-1], color="red", s=100, zorder=5, label="SHORT")
        ax.legend(); ax.set_title(resolved_symbol)
        fig.tight_layout(); self.chart_canvas.draw()

# --- Точка входа ---
if __name__=="__main__":
    app = QApplication(sys.argv)
    win = App(); win.show()
    sys.exit(app.exec())
