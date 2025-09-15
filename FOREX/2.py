# -*- coding: utf-8 -*-
"""
XAUUSD AI-assisted Analyzer (pandas_ta) + simple PySide6 GUI
- Uses pandas_ta for indicators (no ta-lib required)
- Trains a RandomForest on historical features (simple labeling)
- Aggregates indicator votes + ML probability -> final signal (BUY/SELL/HOLD)
- Calculates SL/TP by ATR and rough lot suggestion (risk-based)
IMPORTANT: NO GUARANTEES. Test on demo. Use risk management.
"""

from __future__ import annotations
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
import MetaTrader5 as mt5
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QSpinBox, QCheckBox
)
from PySide6.QtCore import QTimer

# -------------------------
# CONFIG
# -------------------------
BASE_SYMBOL = "XAUUSD"
DEFAULT_TF = "M15"
BARS = 2000

# ML labeling
LOOKAHEAD = 5             # horizon in bars to define label
FUTURE_PCT = 0.0015       # 0.15% move threshold

# Risk / trade settings
DRY_RUN = True
RISK_PER_TRADE = 0.01     # 1% balance
ATR_PERIOD = 14
SL_ATR_MULT = 1.5
TP_SL_RR = 2.0

# ML hyperparams
RF_ESTIMATORS = 200
RANDOM_STATE = 42

# -------------------------
# MT5 helpers
# -------------------------
def mt5_init_or_raise():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

def find_symbol(base: str = BASE_SYMBOL) -> str:
    syms = mt5.symbols_get()
    names = [s.name for s in syms]
    if base in names:
        return base
    for n in names:
        if base in n:
            return n
    # fallback: return base (will likely fail later)
    return base

def get_rates(symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time','open','high','low','close','tick_volume']].set_index('time')
    # replace inf and fill
    df = df.replace([np.inf, -np.inf], np.nan).bfill().fillna(0)
    return df

# -------------------------
# Indicators & features
# -------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Trend
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    df['sma50'] = ta.sma(df['close'], length=50)

    # Momentum
    df['rsi14'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    # Volatility / bands
    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_lower'] = bb[f"BBL_20_2.0"]
    df['bb_mid'] = bb[f"BBM_20_2.0"]
    df['bb_upper'] = bb[f"BBU_20_2.0"]
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    # Oscillators
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']

    df['adx14'] = ta.adx(df['high'], df['low'], df['close'], length=14)[f"ADX_14"]
    df['cci20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    df['obv'] = ta.obv(df['close'], df['tick_volume'])
    df['mom10'] = ta.mom(df['close'], length=10)
    df['wpr14'] = ta.willr(df['high'], df['low'], df['close'], length=14)

    # Derived features
    df['ema20_50'] = df['ema20'] - df['ema50']
    df['ema50_200'] = df['ema50'] - df['ema200']
    df['price_vs_bbmid'] = (df['close'] - df['bb_mid']) / df['bb_mid']

    # Clean NaNs
    df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0)
    return df

# -------------------------
# ML labeling / training
# -------------------------
def make_labels(df: pd.DataFrame, lookahead: int = LOOKAHEAD, thr: float = FUTURE_PCT) -> pd.Series:
    future = df['close'].shift(-lookahead)
    ret = (future - df['close']) / df['close']
    labels = (ret >= thr).astype(int)  # 1 = up, 0 = not up
    return labels

def train_rf(feature_df: pd.DataFrame, labels: pd.Series):
    # select features
    feats = ['ema20_50','ema50_200','price_vs_bbmid','rsi14','macd_hist','atr14',
             'stoch_k','stoch_d','adx14','cci20','obv','mom10','wpr14']
    # drop rows where labels NaN
    valid = labels.dropna().index
    X = feature_df.loc[valid, feats].values
    y = labels.loc[valid].values
    if len(y) < 50:
        # not enough data
        return None, feats, {'error':'not enough data'}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    clf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'test_samples': len(y_test)
    }
    return clf, feats, metrics

# -------------------------
# Indicator voting aggregator
# -------------------------
def indicator_votes(row: pd.Series) -> Tuple[int, List[str]]:
    votes = 0
    reasons = []

    # EMA trend
    if row['ema20'] > row['ema50'] > row['ema200']:
        votes += 1; reasons.append("EMA cluster bullish")
    elif row['ema20'] < row['ema50'] < row['ema200']:
        votes -= 1; reasons.append("EMA cluster bearish")
    else:
        reasons.append("No clear EMA trend")

    # price vs sma50
    if row['close'] > row['sma50']:
        votes += 1; reasons.append("Price > SMA50")
    else:
        votes -= 1; reasons.append("Price < SMA50")

    # RSI
    if row['rsi14'] < 30:
        votes += 1; reasons.append("RSI oversold")
    elif row['rsi14'] > 70:
        votes -= 1; reasons.append("RSI overbought")
    else:
        reasons.append(f"RSI {row['rsi14']:.1f}")

    # MACD hist
    if row['macd_hist'] > 0:
        votes += 1; reasons.append("MACD histogram positive")
    else:
        votes -= 1; reasons.append("MACD histogram negative")

    # Bollinger
    if row['close'] < row['bb_lower']:
        votes += 1; reasons.append("Price below BB lower")
    elif row['close'] > row['bb_upper']:
        votes -= 1; reasons.append("Price above BB upper")
    else:
        reasons.append("Price inside BB")

    # Stochastic
    if row['stoch_k'] < 20 and row['stoch_d'] < 20:
        votes += 1; reasons.append("Stochastic oversold")
    elif row['stoch_k'] > 80 and row['stoch_d'] > 80:
        votes -= 1; reasons.append("Stochastic overbought")
    else:
        reasons.append(f"Stoch K={row['stoch_k']:.1f}")

    # ADX (trend strength)
    if row['adx14'] > 25:
        reasons.append(f"ADX strong {row['adx14']:.1f}")

    # CCI
    if row['cci20'] < -100:
        votes += 1; reasons.append("CCI oversold")
    elif row['cci20'] > 100:
        votes -= 1; reasons.append("CCI overbought")

    # OBV momentum
    # simple check comparing recent OBV slope
    # (we pass row only; in caller we can use series; here we assume caller passes a window)
    # We'll skip OBV detailed here; caller may append

    # Momentum
    if row['mom10'] > 0:
        votes += 1; reasons.append("Momentum positive")
    else:
        votes -= 1; reasons.append("Momentum negative")

    # WPR
    if row['wpr14'] < -80:
        votes += 1; reasons.append("Williams %R deeply oversold")
    elif row['wpr14'] > -20:
        votes -= 1; reasons.append("Williams %R deeply overbought")

    return votes, reasons

# -------------------------
# SL/TP and lot suggestion
# -------------------------
def calc_sl_tp_and_lot(info, price: float, side: str, atr_val: float, account_balance: float):
    if atr_val is None or atr_val == 0:
        return None, None, "N/A"
    sl_dist = SL_ATR_MULT * atr_val
    if side == 'BUY':
        sl = price - sl_dist
        tp = price + sl_dist * TP_SL_RR
    elif side == 'SELL':
        sl = price + sl_dist
        tp = price - sl_dist * TP_SL_RR
    else:
        return None, None, "N/A"

    # rough lot estimation: fallback to conservative estimate
    # tries to use trade_tick_value/size; if unavailable, uses 100 USD per 1 unit move per lot as fallback
    tick_size = getattr(info, 'trade_tick_size', None) or getattr(info, 'point', None)
    tick_value = getattr(info, 'trade_tick_value', None)
    price_move = abs(price - sl)
    if tick_value and tick_size:
        stop_value_per_lot = (price_move / tick_size) * tick_value
    else:
        stop_value_per_lot = price_move * 100.0  # fallback (broker dependent)

    risk_amount = account_balance * RISK_PER_TRADE
    if stop_value_per_lot <= 0:
        lots = 0
    else:
        lots = round(max(getattr(info, 'volume_min', 0.01), risk_amount / stop_value_per_lot), 2)

    return float(sl), float(tp), lots

# -------------------------
# Main pipeline
# -------------------------
@dataclass
class Result:
    symbol: str
    timeframe: str
    price: float
    model_prob: float
    agg_signal: str
    agg_confidence: float
    sl: float
    tp: float
    lots: float
    ml_metrics: Dict
    votes: List[Tuple[str,str]]

def analyze(symbol: str, timeframe_value: int, bars: int = BARS) -> Result:
    df = get_rates(symbol, timeframe_value, bars)
    info = mt5.symbol_info(symbol)
    df_ind = add_indicators(df)

    labels = make_labels(df_ind)
    clf, feats, ml_metrics = train_rf(df_ind, labels)

    # ML prediction on latest (use last index - LOOKAHEAD - 1 to ensure same alignment as labels)
    latest_index = df_ind.index[-(LOOKAHEAD + 1)]
    X_latest = df_ind.loc[[latest_index], feats].values
    model_prob = float(clf.predict_proba(X_latest)[:,1][0]) if clf is not None else 0.5

    # indicator votes (we pass the last row)
    last_row = df_ind.iloc[-1]
    votes_val, reasons = indicator_votes(last_row)

    # OBV slope check (use last 5)
    obv_slope = float((df_ind['obv'].iat[-1] - df_ind['obv'].iat[-6]) if len(df_ind) > 6 else 0)
    if obv_slope > 0:
        votes_val += 1; reasons.append("OBV trending up")
    elif obv_slope < 0:
        votes_val -= 1; reasons.append("OBV trending down")

    # aggregate
    max_votes = 10  # approximate
    agg_conf = (votes_val + max_votes) / (2*max_votes) * 100  # rough 0..100

    # combine ML (prob -> [-1,1]) and indicator votes (normalized)
    ml_component = (model_prob * 2 - 1)  # [-1,1]
    ind_component = (votes_val / max_votes)  # approx [-1,1]
    COMBINE_ML_WEIGHT = 0.6
    COMBINE_IND_WEIGHT = 0.4
    combined_score = COMBINE_ML_WEIGHT * ml_component + COMBINE_IND_WEIGHT * ind_component

    if combined_score > 0.15:
        agg_signal = "BUY"
    elif combined_score < -0.15:
        agg_signal = "SELL"
    else:
        agg_signal = "HOLD"

    # SL/TP/lot
    price = float(df_ind['close'].iat[-1])
    atr_val = float(df_ind['atr14'].iat[-1]) if 'atr14' in df_ind.columns else None
    account = mt5.account_info()
    balance = float(account.balance) if account else 1000.0
    sl, tp, lots = calc_sl_tp_and_lot(info, price, agg_signal, atr_val, balance)

    # pack votes as name/string
    votes_list = [(f"VotesSum", votes_val),]
    votes_list += [(f"reason_{i+1}", r) for i,r in enumerate(reasons)]

    return Result(
        symbol=symbol,
        timeframe=str(timeframe_value),
        price=price,
        model_prob=model_prob,
        agg_signal=agg_signal,
        agg_confidence=float((combined_score+1)/2*100),
        sl=sl,
        tp=tp,
        lots=lots,
        ml_metrics=ml_metrics or {},
        votes=votes_list
    )

# -------------------------
# GUI (light)
# -------------------------
class SimpleGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XAUUSD AI Analyzer (pandas_ta)")
        self.resize(900,700)
        layout = QVBoxLayout()

        top = QHBoxLayout()
        top.addWidget(QLabel("TF:"))
        self.tf = QComboBox()
        self.tf.addItems(['M1','M5','M15','M30','H1','H4','D1'])
        self.tf.setCurrentText(DEFAULT_TF)
        top.addWidget(self.tf)

        self.btn = QPushButton("Analyze")
        top.addWidget(self.btn)

        self.auto = QCheckBox("Auto")
        top.addWidget(self.auto)

        top.addWidget(QLabel("Interval s:"))
        self.intspin = QSpinBox(); self.intspin.setRange(10,3600); self.intspin.setValue(60)
        top.addWidget(self.intspin)

        layout.addLayout(top)

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        layout.addWidget(self.out)

        self.setLayout(layout)
        self.btn.clicked.connect(self.on_click)
        self.timer = QTimer(); self.timer.timeout.connect(self.on_click)
        self.auto.stateChanged.connect(self.toggle_timer)

        # initialize MT5 and symbol
        mt5_init_or_raise()
        self.symbol = find_symbol(BASE_SYMBOL)
        if self.symbol != BASE_SYMBOL:
            self.out.append(f"Using symbol: {self.symbol}")

    def toggle_timer(self):
        if self.auto.isChecked():
            self.timer.start(self.intspin.value()*1000)
        else:
            self.timer.stop()

    def on_click(self):
        try:
            tf_map = {'M1':mt5.TIMEFRAME_M1,'M5':mt5.TIMEFRAME_M5,'M15':mt5.TIMEFRAME_M15,'M30':mt5.TIMEFRAME_M30,'H1':mt5.TIMEFRAME_H1,'H4':mt5.TIMEFRAME_H4,'D1':mt5.TIMEFRAME_D1}
            tf_val = tf_map[self.tf.currentText()]
            res = analyze(self.symbol, tf_val, BARS)
            lines = []
            lines.append(f"Symbol: {res.symbol}")
            lines.append(f"Price: {res.price:.4f}")
            lines.append(f"ML BUY prob: {res.model_prob:.3f}")
            lines.append(f"Final signal: {res.agg_signal} | Confidence: {res.agg_confidence:.1f}%")
            if res.sl and res.tp:
                lines.append(f"Entry ~ {res.price:.4f} | SL {res.sl:.4f} | TP {res.tp:.4f} | Lots {res.lots}")
            lines.append("\nML metrics:")
            lines.append(str(res.ml_metrics))
            lines.append("\nVotes and reasons:")
            for v in res.votes:
                lines.append(str(v))
            lines.append("\nDISCLAIMER: NO GUARANTEE. Test on demo. Use risk management.")
            self.out.setPlainText("\n".join(lines))
        except Exception as e:
            self.out.append(f"[Error] {e}")

# -------------------------
# Run
# -------------------------
def main():
    app = QApplication(sys.argv)
    win = SimpleGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
