# -*- coding: utf-8 -*-
import sys
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit
)

# -------------------
# ИНИЦИАЛИЗАЦИЯ MT5
# -------------------
if not mt5.initialize():
    print("Ошибка инициализации MT5:", mt5.last_error())
    sys.exit()

# Попробуем найти XAUUSD (некоторые терминалы используют XAUUSD+)
SYMBOL = None
for sym in ["XAUUSD", "XAUUSD+", "XAUUSD."]:
    if mt5.symbol_select(sym, True):
        SYMBOL = sym
        break

if SYMBOL is None:
    print("Символ XAUUSD не найден в терминале")
    sys.exit()

# -------------------
# ФУНКЦИИ ИНДИКАТОРОВ
# -------------------
def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def SMA(series, period):
    return series.rolling(period).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    return macd_line, signal_line

def ATR(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def Bollinger(series, period=20, std=2):
    sma = SMA(series, period)
    stddev = series.rolling(period).std()
    upper = sma + std * stddev
    lower = sma - std * stddev
    return upper, lower

def Stochastic(df, k=14, d=3):
    low_min = df['low'].rolling(k).min()
    high_max = df['high'].rolling(k).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def ADX(df, period=14):
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

def CCI(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.fabs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)

def OBV(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i-1]:
            obv.append(obv[-1] + df['tick_volume'][i])
        elif df['close'][i] < df['close'][i-1]:
            obv.append(obv[-1] - df['tick_volume'][i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def Momentum(series, period=10):
    return series - series.shift(period)

def WilliamsR(df, period=14):
    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()
    return -100 * (high_max - df['close']) / (high_max - low_min)

# -------------------
# АНАЛИЗ
# -------------------
def analyze(timeframe=mt5.TIMEFRAME_M15, bars=500):
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    close = df['close']
    votes = []
    reasons = []

    # EMA20 / EMA50
    if EMA(close, 20).iloc[-1] > EMA(close, 50).iloc[-1]:
        votes.append(1); reasons.append("EMA20 > EMA50 → BUY")
    else:
        votes.append(-1); reasons.append("EMA20 < EMA50 → SELL")

    # EMA200
    if close.iloc[-1] > EMA(close, 200).iloc[-1]:
        votes.append(1); reasons.append("Цена > EMA200 → BUY")
    else:
        votes.append(-1); reasons.append("Цена < EMA200 → SELL")

    # SMA50
    if close.iloc[-1] > SMA(close, 50).iloc[-1]:
        votes.append(1); reasons.append("Цена > SMA50 → BUY")
    else:
        votes.append(-1); reasons.append("Цена < SMA50 → SELL")

    # RSI
    rsi = RSI(close).iloc[-1]
    if rsi < 30:
        votes.append(1); reasons.append("RSI < 30 (перепроданность) → BUY")
    elif rsi > 70:
        votes.append(-1); reasons.append("RSI > 70 (перекупленность) → SELL")
    else:
        votes.append(0); reasons.append("RSI нейтрально")

    # MACD
    macd, signal = MACD(close)
    if macd.iloc[-1] > signal.iloc[-1]:
        votes.append(1); reasons.append("MACD выше сигнала → BUY")
    else:
        votes.append(-1); reasons.append("MACD ниже сигнала → SELL")

    # Bollinger
    upper, lower = Bollinger(close)
    if close.iloc[-1] < lower.iloc[-1]:
        votes.append(1); reasons.append("Цена ниже Bollinger → BUY")
    elif close.iloc[-1] > upper.iloc[-1]:
        votes.append(-1); reasons.append("Цена выше Bollinger → SELL")
    else:
        votes.append(0); reasons.append("Bollinger нейтрально")

    # Stochastic
    stoch_k, stoch_d = Stochastic(df)
    if stoch_k.iloc[-1] < 20 and stoch_d.iloc[-1] < 20:
        votes.append(1); reasons.append("Стохастик < 20 → BUY")
    elif stoch_k.iloc[-1] > 80 and stoch_d.iloc[-1] > 80:
        votes.append(-1); reasons.append("Стохастик > 80 → SELL")
    else:
        votes.append(0); reasons.append("Стохастик нейтрально")

    # ADX
    adx = ADX(df).iloc[-1]
    if adx > 25:
        reasons.append(f"ADX={adx:.1f} (сильный тренд)")
    else:
        reasons.append(f"ADX={adx:.1f} (слабый тренд)")

    # CCI
    cci = CCI(df).iloc[-1]
    if cci < -100:
        votes.append(1); reasons.append("CCI < -100 → BUY")
    elif cci > 100:
        votes.append(-1); reasons.append("CCI > 100 → SELL")
    else:
        votes.append(0); reasons.append("CCI нейтрально")

    # OBV
    obv = OBV(df)
    if obv.iloc[-1] > obv.iloc[-2]:
        votes.append(1); reasons.append("OBV растёт → BUY")
    else:
        votes.append(-1); reasons.append("OBV падает → SELL")

    # Momentum
    mom = Momentum(close).iloc[-1]
    if mom > 0:
        votes.append(1); reasons.append("Momentum > 0 → BUY")
    else:
        votes.append(-1); reasons.append("Momentum < 0 → SELL")

    # Williams %R
    wr = WilliamsR(df).iloc[-1]
    if wr < -80:
        votes.append(1); reasons.append("Williams %R < -80 → BUY")
    elif wr > -20:
        votes.append(-1); reasons.append("Williams %R > -20 → SELL")
    else:
        votes.append(0); reasons.append("Williams %R нейтрально")

    # Итоговый сигнал
    score = sum(votes)
    if score > 2:
        signal = "BUY"
    elif score < -2:
        signal = "SELL"
    else:
        signal = "FLAT"

    atr = ATR(df).iloc[-1]
    sl = close.iloc[-1] - 1.5 * atr if signal == "BUY" else close.iloc[-1] + 1.5 * atr
    tp = close.iloc[-1] + 3 * atr if signal == "BUY" else close.iloc[-1] - 3 * atr

    return signal, score, reasons, close.iloc[-1], sl, tp

# -------------------
# GUI
# -------------------
class AnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XAUUSD Forex Analyzer")
        self.resize(700, 500)

        layout = QVBoxLayout()

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Таймфрейм:"))
        self.tf_box = QComboBox()
        self.tf_box.addItems(["M1","M5","M15","M30","H1","H4","D1"])
        hl.addWidget(self.tf_box)
        layout.addLayout(hl)

        self.btn = QPushButton("Анализировать")
        self.btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn)

        self.result = QTextEdit()
        layout.addWidget(self.result)

        self.setLayout(layout)

    def run_analysis(self):
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf = tf_map[self.tf_box.currentText()]
        signal, score, reasons, price, sl, tp = analyze(tf)

        text = f"Символ: {SYMBOL}\nЦена: {price:.2f}\nСигнал: {signal} (сила {score})\n\n"
        for r in reasons:
            text += "- " + r + "\n"
        text += f"\nРекомендуемый SL: {sl:.2f}\nРекомендуемый TP: {tp:.2f}"
        self.result.setPlainText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AnalyzerApp()
    win.show()
    sys.exit(app.exec())
