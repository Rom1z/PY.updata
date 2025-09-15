# -*- coding: utf-8 -*-
"""
3.py — Исправленная полная версия XAUUSD AI-assisted Analyzer (MT5)
- ~20 индикаторов (pandas_ta)
- 3-классовая RandomForest (DOWN / HOLD / UP)
- Агрегатор индикаторов + ML -> сигнал BUY/SELL/HOLD
- SL и TP1/TP2/TP3 через ATR, приблизительная рекомендация лота
- PySide6 GUI (анализ по кнопке и автоповтор)
IMPORTANT: TEST ON DEMO. NO GUARANTEES.
"""
from __future__ import annotations
import sys, traceback
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import MetaTrader5 as mt5

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QSpinBox, QCheckBox, QProgressBar
)
from PySide6.QtCore import QTimer

# -------------------------
# CONFIG
# -------------------------
BASE_SYMBOL = "XAUUSD"
DEFAULT_TF = "M15"
BARS = 2500

# ML labeling (3-class)
LOOKAHEAD = 5
THR_UP = 0.0016      # >=0.16% => UP
THR_DOWN = -0.0016   # <=-0.16% => DOWN

# Risk / trade settings
RISK_PER_TRADE = 0.01
ATR_PERIOD = 14
SL_ATR_MULT = 1.4
TP1_ATR = 1.0
TP2_ATR = 2.2
TP3_ATR = 3.5

# ML hyperparams
RF_ESTIMATORS = 300
RANDOM_STATE = 42
MIN_TRAIN_SAMPLES = 120

# Combine weights
ML_WEIGHT = 0.55
IND_WEIGHT = 0.45

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
    return base

def get_rates(symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time','open','high','low','close','tick_volume']].set_index('time')
    df = df.rename(columns={'tick_volume':'volume'})

    # replace inf and fill, then cast to float to avoid dtype warnings
    df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0)
    # ensure numeric floats for core columns
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').bfill().ffill().fillna(0).astype(float)
    return df

# -------------------------
# Indicators (~20) — corrected to avoid dtype/efi warnings
# -------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Defensive numeric conversion before computing indicators
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').bfill().ffill().fillna(0).astype(float)

    try:
        # Moving averages
        s_close = df['close']
        df['ema8'] = ta.ema(s_close, length=8).astype(float)
        df['ema20'] = ta.ema(s_close, length=20).astype(float)
        df['ema50'] = ta.ema(s_close, length=50).astype(float)
        df['ema200'] = ta.ema(s_close, length=200).astype(float)
        df['sma50'] = ta.sma(s_close, length=50).astype(float)
        df['wma34'] = ta.wma(s_close, length=34).astype(float)
        df['hma21'] = ta.hma(s_close, length=21).astype(float)
        df['tema30'] = ta.tema(s_close, length=30).astype(float)
        df['kama10'] = ta.kama(s_close, length=10).astype(float)

        # Momentum
        df['rsi14'] = ta.rsi(s_close, length=14).astype(float)
        # MFI: pass numeric arrays
        df['mfi14'] = pd.to_numeric(ta.mfi(df['high'], df['low'], s_close, df['volume'], length=14), errors='coerce').fillna(0).astype(float)
        df['roc12'] = ta.roc(s_close, length=12).astype(float)
        df['mom10'] = ta.mom(s_close, length=10).astype(float)

        # MACD
        macd = ta.macd(s_close, fast=12, slow=26, signal=9)
        df['macd'] = pd.to_numeric(macd.get('MACD_12_26_9'), errors='coerce').fillna(0).astype(float)
        df['macd_sig'] = pd.to_numeric(macd.get('MACDs_12_26_9'), errors='coerce').fillna(0).astype(float)
        df['macd_hist'] = pd.to_numeric(macd.get('MACDh_12_26_9'), errors='coerce').fillna(0).astype(float)

        # Volatility / bands / ATR
        bb = ta.bbands(s_close, length=20, std=2.0)
        df['bbl'] = pd.to_numeric(bb.get("BBL_20_2.0"), errors='coerce').fillna(0).astype(float)
        df['bbm'] = pd.to_numeric(bb.get("BBM_20_2.0"), errors='coerce').fillna(0).astype(float)
        df['bbu'] = pd.to_numeric(bb.get("BBU_20_2.0"), errors='coerce').fillna(0).astype(float)
        df['atr14'] = pd.to_numeric(ta.atr(df['high'], df['low'], s_close, length=ATR_PERIOD), errors='coerce').fillna(0).astype(float)

        # Oscillators
        stoch = ta.stoch(df['high'], df['low'], s_close, k=14, d=3)
        df['stoch_k'] = pd.to_numeric(stoch.get('STOCHk_14_3_3'), errors='coerce').fillna(0).astype(float)
        df['stoch_d'] = pd.to_numeric(stoch.get('STOCHd_14_3_3'), errors='coerce').fillna(0).astype(float)
        df['cci20'] = pd.to_numeric(ta.cci(df['high'], df['low'], s_close, length=20), errors='coerce').fillna(0).astype(float)
        df['wpr14'] = pd.to_numeric(ta.willr(df['high'], df['low'], s_close, length=14), errors='coerce').fillna(0).astype(float)
        adx = ta.adx(df['high'], df['low'], s_close, length=14)
        df['adx14'] = pd.to_numeric(adx.get('ADX_14'), errors='coerce').fillna(0).astype(float)

        # Volume/flow
        df['obv'] = pd.to_numeric(ta.obv(s_close, df['volume']), errors='coerce').fillna(0).astype(float)
        # Correct EFI call: close and volume, length specified once
        df['efi'] = pd.to_numeric(ta.efi(s_close, df['volume'], length=13), errors='coerce').fillna(0).astype(float)

        # SAR and AO
        df['sar'] = pd.to_numeric(ta.sar(df['high'], df['low']), errors='coerce').fillna(0).astype(float)
        ao = ta.ao(df['high'], df['low'])
        # handle AO return types safely
        if isinstance(ao, dict):
            ao_val = ao.get('AO_5_34', ao.get('AO', 0))
        else:
            ao_val = ao if ao is not None else 0
        df['ao'] = pd.to_numeric(ao_val, errors='coerce').fillna(0).astype(float)

        # Derived
        df['ema20_50'] = (df['ema20'] - df['ema50']).astype(float)
        df['ema8_20'] = (df['ema8'] - df['ema20']).astype(float)
        df['price_vs_bbmid'] = ((s_close - df['bbm']) / (df['bbm'] + 1e-9)).astype(float)

    except Exception as e:
        print("[add_indicators] warning:", e)
    # final clean: avoid inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0)
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except Exception:
            pass
    return df

# -------------------------
# Labels (3-class)
# -------------------------
def make_labels(df: pd.DataFrame, lookahead: int = LOOKAHEAD, thr_up: float = THR_UP, thr_down: float = THR_DOWN) -> pd.Series:
    future = df['close'].shift(-lookahead)
    ret = (future - df['close']) / df['close']
    conds = [
        (ret <= thr_down),
        (ret > thr_down) & (ret < thr_up),
        (ret >= thr_up)
    ]
    choices = [0, 1, 2]  # 0=DOWN,1=HOLD,2=UP
    labels = np.select(conds, choices, default=1)
    return pd.Series(labels, index=df.index)

# -------------------------
# Train classifier
# -------------------------
def train_rf(feature_df: pd.DataFrame, labels: pd.Series):
    feats = [
        'ema8','ema20','ema50','ema200','sma50','wma34','hma21','tema30','kama10',
        'rsi14','mfi14','roc12','mom10','macd_hist','atr14','bbm','bbu',
        'stoch_k','stoch_d','cci20','wpr14','adx14','obv','efi','sar','ao',
        'ema20_50','ema8_20','price_vs_bbmid'
    ]
    feats = [f for f in feats if f in feature_df.columns]
    valid_idx = labels.dropna().index
    X = feature_df.loc[valid_idx, feats].values
    y = labels.loc[valid_idx].values
    if len(y) < MIN_TRAIN_SAMPLES:
        return None, feats, {'error':'not enough training samples', 'samples': len(y)}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, shuffle=True, stratify=y)
    clf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'test_samples': int(len(y_test))
    }
    try:
        fi = dict(zip(feats, clf.feature_importances_.round(4)))
        metrics['feature_importances'] = {k:v for k,v in sorted(fi.items(), key=lambda x:-x[1])[:12]}
    except Exception:
        metrics['feature_importances'] = {}
    return clf, feats, metrics

# -------------------------
# Indicator voting
# -------------------------
def indicator_votes(df: pd.DataFrame, idx: pd.Timestamp) -> Tuple[float, List[str]]:
    row = df.loc[idx]
    votes = 0.0
    reasons: List[str] = []

    # EMA cluster
    try:
        if row['ema8'] > row['ema20'] > row['ema50'] > row['ema200']:
            votes += 2; reasons.append("EMA cluster strong bull")
        elif row['ema8'] < row['ema20'] < row['ema50'] < row['ema200']:
            votes -= 2; reasons.append("EMA cluster strong bear")
        else:
            reasons.append("EMAs mixed")
    except Exception:
        reasons.append("EMAs not available")

    # short-term
    try:
        if row['ema8_20'] > 0:
            votes += 0.6; reasons.append("short EMA bullish")
        else:
            votes -= 0.6; reasons.append("short EMA bearish")
    except Exception:
        pass

    # price vs sma50
    try:
        if row['close'] > row['sma50']:
            votes += 0.6; reasons.append("price > SMA50")
        else:
            votes -= 0.6; reasons.append("price < SMA50")
    except Exception:
        pass

    # RSI
    try:
        if row['rsi14'] < 30:
            votes += 0.8; reasons.append("RSI oversold")
        elif row['rsi14'] > 70:
            votes -= 0.8; reasons.append("RSI overbought")
        else:
            reasons.append(f"RSI {row['rsi14']:.1f}")
    except Exception:
        pass

    # MACD hist
    try:
        if row['macd_hist'] > 0:
            votes += 0.7; reasons.append("MACD hist positive")
        else:
            votes -= 0.7; reasons.append("MACD hist negative")
    except Exception:
        pass

    # Bollinger
    try:
        if 'bbl' in row.index and row['close'] < row['bbl']:
            votes += 0.6; reasons.append("price below BB lower (contrarian)")
        elif 'bbu' in row.index and row['close'] > row['bbu']:
            votes -= 0.6; reasons.append("price above BB upper (overbought)")
    except Exception:
        pass

    # Stochastic
    try:
        if row['stoch_k'] < 20 and row['stoch_d'] < 20:
            votes += 0.6; reasons.append("Stoch oversold")
        elif row['stoch_k'] > 80 and row['stoch_d'] > 80:
            votes -= 0.6; reasons.append("Stoch overbought")
    except Exception:
        pass

    # ADX
    try:
        if row['adx14'] > 25:
            reasons.append(f"ADX strong ({row['adx14']:.1f})")
    except Exception:
        pass

    # CCI
    try:
        if row['cci20'] < -100:
            votes += 0.4; reasons.append("CCI oversold")
        elif row['cci20'] > 100:
            votes -= 0.4; reasons.append("CCI overbought")
    except Exception:
        pass

    # OBV slope last 5 bars
    try:
        if len(df) > 6:
            obv_slope = float(df['obv'].iat[-1] - df['obv'].iat[-6])
            if obv_slope > 0:
                votes += 0.5; reasons.append("OBV rising")
            elif obv_slope < 0:
                votes -= 0.5; reasons.append("OBV falling")
    except Exception:
        pass

    # WPR
    try:
        if row['wpr14'] < -80:
            votes += 0.4; reasons.append("WPR deeply oversold")
        elif row['wpr14'] > -20:
            votes -= 0.4; reasons.append("WPR overbought")
    except Exception:
        pass

    # ROC
    try:
        if row['roc12'] > 0:
            votes += 0.3; reasons.append("ROC positive")
        else:
            votes -= 0.3; reasons.append("ROC negative")
    except Exception:
        pass

    return votes, reasons

# -------------------------
# SL / TP / lot calc
# -------------------------
def calc_sl_tp_and_lot(info, price: float, side: str, atr_val: float, account_balance: float):
    if atr_val is None or atr_val <= 0:
        return None, None, None, None, "N/A"
    sl_dist = SL_ATR_MULT * atr_val
    if side == 'BUY':
        sl = price - sl_dist
        tp1 = price + TP1_ATR * atr_val
        tp2 = price + TP2_ATR * atr_val
        tp3 = price + TP3_ATR * atr_val
    elif side == 'SELL':
        sl = price + sl_dist
        tp1 = price - TP1_ATR * atr_val
        tp2 = price - TP2_ATR * atr_val
        tp3 = price - TP3_ATR * atr_val
    else:
        return None, None, None, None, "N/A"

    tick_size = getattr(info, 'trade_tick_size', None) or getattr(info, 'point', None) or 0.01
    tick_value = getattr(info, 'trade_tick_value', None)
    price_move = abs(price - sl)
    if tick_value and tick_size:
        stop_value_per_lot = (price_move / tick_size) * tick_value
    else:
        stop_value_per_lot = price_move * 100.0

    risk_amount = account_balance * RISK_PER_TRADE
    if stop_value_per_lot <= 0:
        lots = 0.0
    else:
        raw = risk_amount / stop_value_per_lot
        min_vol = getattr(info, 'volume_min', 0.01) or 0.01
        lots = round(max(min_vol, raw), 2)

    return float(sl), float(tp1), float(tp2), float(tp3), lots

# -------------------------
# Result dataclass
# -------------------------
@dataclass
class Result:
    symbol: str
    timeframe: str
    price: float
    model_preds: Dict[str, float]
    agg_signal: str
    agg_confidence: float
    sl: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    lots: Optional[float]
    ml_metrics: Dict
    votes: List[Tuple[str,str]]

# -------------------------
# Main pipeline: analyze
# -------------------------
def analyze(symbol: str, timeframe_value: int, bars: int = BARS) -> Result:
    df = get_rates(symbol, timeframe_value, bars)
    info = mt5.symbol_info(symbol)
    df_ind = add_indicators(df)

    labels = make_labels(df_ind)
    clf, feats, ml_metrics = train_rf(df_ind, labels)

    latest_idx = df_ind.index[-1]
    model_probs = {'DOWN':0.33,'HOLD':0.34,'UP':0.33}
    if clf is not None:
        try:
            X_latest = df_ind.loc[[latest_idx], feats].values
            proba = clf.predict_proba(X_latest)[0]
            # map probabilities according to clf.classes_
            for i, cls in enumerate(clf.classes_):
                try:
                    c_int = int(cls)
                except Exception:
                    c_int = None
                if c_int == 0:
                    model_probs['DOWN'] = float(proba[i])
                elif c_int == 1:
                    model_probs['HOLD'] = float(proba[i])
                elif c_int == 2:
                    model_probs['UP'] = float(proba[i])
                else:
                    if i == 0: model_probs['DOWN'] = float(proba[i])
                    if i == 1: model_probs['HOLD'] = float(proba[i])
                    if i == 2: model_probs['UP'] = float(proba[i])
        except Exception as e:
            print("[analyze] model predict error:", e)

    # indicator votes
    votes_val, reasons = indicator_votes(df_ind, latest_idx)

    # normalize and combine
    max_possible = 6.0
    ind_component = max(min(votes_val / max_possible, 1.0), -1.0)
    ml_score = (model_probs['UP'] - model_probs['DOWN'])
    combined_score = ML_WEIGHT * ml_score + IND_WEIGHT * ind_component

    if combined_score > 0.18:
        agg_signal = "BUY"
    elif combined_score < -0.18:
        agg_signal = "SELL"
    else:
        agg_signal = "HOLD"

    agg_confidence = float(min(99.0, abs(combined_score) * 100))

    price = float(df_ind['close'].iat[-1])
    atr_val = float(df_ind['atr14'].iat[-1]) if 'atr14' in df_ind.columns else None
    account = mt5.account_info()
    balance = float(account.balance) if account else 1000.0

    sl = tp1 = tp2 = tp3 = None; lots = None
    if agg_signal in ('BUY','SELL'):
        try:
            sl, tp1, tp2, tp3, lots = calc_sl_tp_and_lot(info, price, agg_signal, atr_val, balance)
        except Exception:
            traceback.print_exc()

    votes_list = [("vote_sum", votes_val)]
    votes_list += [(f"reason_{i+1}", r) for i,r in enumerate(reasons[:20])]

    return Result(
        symbol=symbol,
        timeframe=str(timeframe_value),
        price=price,
        model_preds=model_probs,
        agg_signal=agg_signal,
        agg_confidence=agg_confidence,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        lots=lots,
        ml_metrics=ml_metrics or {},
        votes=votes_list
    )

# -------------------------
# GUI
# -------------------------
class SimpleGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XAUUSD AI Analyzer — MT5")
        self.resize(980,720)
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
        self.intspin = QSpinBox(); self.intspin.setRange(5,3600); self.intspin.setValue(45)
        top.addWidget(self.intspin)

        top.addWidget(QLabel("Bars:"))
        self.barspin = QSpinBox(); self.barspin.setRange(200,5000); self.barspin.setValue(BARS)
        top.addWidget(self.barspin)

        layout.addLayout(top)

        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        layout.addWidget(self.out)

        self.setLayout(layout)
        self.btn.clicked.connect(self.on_click)
        self.timer = QTimer(); self.timer.timeout.connect(self.on_click)
        self.auto.stateChanged.connect(self.toggle_timer)

        # init MT5
        try:
            mt5_init_or_raise()
            self.symbol = find_symbol(BASE_SYMBOL)
            if self.symbol != BASE_SYMBOL:
                self.out.append(f"[Info] Using symbol: {self.symbol}")
        except Exception as e:
            self.out.append(f"[MT5 init error] {e}")
            self.symbol = BASE_SYMBOL

    def toggle_timer(self):
        if self.auto.isChecked():
            self.timer.start(self.intspin.value()*1000)
        else:
            self.timer.stop()

    def on_click(self):
        try:
            self.progress.setValue(5)
            tf_map = {'M1':mt5.TIMEFRAME_M1,'M5':mt5.TIMEFRAME_M5,'M15':mt5.TIMEFRAME_M15,'M30':mt5.TIMEFRAME_M30,'H1':mt5.TIMEFRAME_H1,'H4':mt5.TIMEFRAME_H4,'D1':mt5.TIMEFRAME_D1}
            tf_val = tf_map[self.tf.currentText()]
            bars = int(self.barspin.value())
            self.progress.setValue(20)
            res = analyze(self.symbol, tf_val, bars)
            self.progress.setValue(80)
            lines = []
            lines.append(f"Symbol: {res.symbol}   TF: {self.tf.currentText()}")
            lines.append(f"Price: {res.price:.4f}")
            lines.append(f"Model probs -> UP {res.model_preds['UP']:.3f}  HOLD {res.model_preds['HOLD']:.3f}  DOWN {res.model_preds['DOWN']:.3f}")
            lines.append(f"Final signal: {res.agg_signal} | Confidence: {res.agg_confidence:.1f}%")
            if res.sl is not None:
                lines.append(f"Entry ~ {res.price:.4f}")
                lines.append(f"SL: {res.sl:.4f}")
                lines.append(f"TP1: {res.tp1:.4f} (ATR x{TP1_ATR})")
                lines.append(f"TP2: {res.tp2:.4f} (ATR x{TP2_ATR})")
                lines.append(f"TP3: {res.tp3:.4f} (ATR x{TP3_ATR})")
                lines.append(f"Suggested lots: {res.lots}")
            lines.append("\nML metrics:")
            for k,v in res.ml_metrics.items():
                lines.append(f"  {k}: {v}")
            lines.append("\nVotes and reasons:")
            for v in res.votes:
                lines.append(f"  {v[0]}: {v[1]}")
            lines.append("\nDISCLAIMER: NO GUARANTEE. Backtest and use risk management.")
            self.out.setPlainText("\n".join(lines))
            self.progress.setValue(100)
        except Exception as e:
            self.out.append(f"[Error] {e}\n{traceback.format_exc()}")
            self.progress.setValue(0)

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
