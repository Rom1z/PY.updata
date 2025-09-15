# -*- coding: utf-8 -*-
"""
XAUUSD AI-Analyzer — Русскоязычная версия
- ~30 индикаторов (pandas_ta + самописный PSAR/Donchian/VWAP)
- Ансамбль ML: RandomForest + MLPClassifier -> сигнал ПОКУПАТЬ/ПРОДАВАТЬ/ДЕРЖАТЬ
- Расчёт SL, TP1/TP2/TP3 через ATR и приблизительный расчёт лота
- PySide6 GUI на русском, лог сигналов в CSV
- ВНИМАНИЕ: тестируй на демо. Нет гарантий прибыли.
"""
from __future__ import annotations
import sys, os, traceback, math, time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import MetaTrader5 as mt5

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QSpinBox, QCheckBox, QProgressBar, QFileDialog
)
from PySide6.QtCore import QTimer

# -------------------------
# Конфигурация
# -------------------------
BASE_SYMBOL = "XAUUSD"
DEFAULT_TF = "M15"
BARS = 3000

LOOKAHEAD = 6
THR_UP = 0.0018
THR_DOWN = -0.0018

RISK_PER_TRADE = 0.01
ATR_PERIOD = 14
SL_ATR_MULT = 1.3
TP1_ATR = 1.0
TP2_ATR = 2.0
TP3_ATR = 3.5

RF_ESTIMATORS = 300
MLP_HIDDEN = (64, 32)
RANDOM_STATE = 42
MIN_TRAIN_SAMPLES = 140

ML_WEIGHT = 0.6
IND_WEIGHT = 0.4

LOG_CSV = "signals_log.csv"

# -------------------------
# Утилиты
# -------------------------
def to_series_safe(x, index, fill=0.0):
    """Возвращает pd.Series dtype=float64, выравнивая по index."""
    try:
        if x is None:
            return pd.Series(fill, index=index, dtype='float64')
        if isinstance(x, pd.Series):
            s = x.reindex(index).fillna(fill).astype('float64')
            return s
        if isinstance(x, (np.ndarray, list, tuple)):
            arr = np.array(x)
            # if length differs, try to align by tail
            if arr.shape[0] == len(index):
                return pd.Series(arr, index=index, dtype='float64').fillna(fill)
            elif arr.shape[0] < len(index):
                # pad front with fill
                pad = np.full(len(index) - arr.shape[0], fill)
                return pd.Series(np.concatenate([pad, arr]), index=index, dtype='float64').fillna(fill)
            else:
                # truncate head
                return pd.Series(arr[-len(index):], index=index, dtype='float64').fillna(fill)
        if isinstance(x, dict):
            # try common keys
            # if dict of arrays, pick first
            for v in x.values():
                return to_series_safe(v, index, fill)
        return pd.Series(fill, index=index, dtype='float64')
    except Exception:
        return pd.Series(fill, index=index, dtype='float64')

def ensure_float_cols(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').bfill().ffill().fillna(0).astype('float64')

# -------------------------
# PSAR (самописный) — чтобы не зависеть от отсутствия sar в pandas_ta
# -------------------------
def parabolic_sar(high: pd.Series, low: pd.Series, af0=0.02, af_step=0.02, af_max=0.2):
    """
    Простая реализация Parabolic SAR.
    Возвращает pd.Series тех же индексов.
    """
    length = len(high)
    if length == 0:
        return pd.Series([], dtype='float64')
    sar = np.zeros(length, dtype='float64')
    trend = 1  # 1 = up, -1 = down
    af = af0
    ep = high.iloc[0]  # extreme point
    sar[0] = low.iloc[0]  # initial SAR
    for i in range(1, length):
        prev = sar[i-1]
        if trend == 1:
            sar[i] = prev + af * (ep - prev)
            # ensure sar is not above last two lows
            sar[i] = min(sar[i], low.iloc[i-1], low.iloc[i-2] if i>=2 else low.iloc[i-1])
            if low.iloc[i] < sar[i]:
                # flip to down
                trend = -1
                sar[i] = ep
                ep = low.iloc[i]
                af = af0
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev + af * (ep - prev)
            sar[i] = max(sar[i], high.iloc[i-1], high.iloc[i-2] if i>=2 else high.iloc[i-1])
            if high.iloc[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high.iloc[i]
                af = af0
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)
    return pd.Series(sar, index=high.index, dtype='float64')

# -------------------------
# MT5 helpers (безопасно)
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

    # приведение к float64 заранее (устраняет FutureWarning при присвоении массивов)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.bfill().ffill().fillna(0)
    for c in df.columns:
        try:
            df[c] = df[c].astype('float64')
        except Exception:
            pass
    return df

# -------------------------
# Индикаторы (30+) — добавляем много признаков
# -------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ensure_float_cols(df, ['open','high','low','close','volume'])
    idx = df.index
    s_close = df['close']

    # 1-9: moving averages
    df['ema5'] = to_series_safe(ta.ema(s_close, length=5), idx)
    df['ema8'] = to_series_safe(ta.ema(s_close, length=8), idx)
    df['ema13'] = to_series_safe(ta.ema(s_close, length=13), idx)
    df['ema21'] = to_series_safe(ta.ema(s_close, length=21), idx)
    df['ema50'] = to_series_safe(ta.ema(s_close, length=50), idx)
    df['ema100'] = to_series_safe(ta.ema(s_close, length=100), idx)
    df['ema200'] = to_series_safe(ta.ema(s_close, length=200), idx)
    df['sma50'] = to_series_safe(ta.sma(s_close, length=50), idx)
    df['wma34'] = to_series_safe(ta.wma(s_close, length=34), idx)

    # 10-16: momentum
    df['rsi14'] = to_series_safe(ta.rsi(s_close, length=14), idx)
    # MFI may return ndarray; wrap
    try:
        mfi_raw = ta.mfi(df['high'], df['low'], s_close, df['volume'], length=14)
    except Exception:
        mfi_raw = None
    df['mfi14'] = to_series_safe(mfi_raw, idx)
    df['roc12'] = to_series_safe(ta.roc(s_close, length=12), idx)
    df['mom10'] = to_series_safe(ta.mom(s_close, length=10), idx)
    macd = ta.macd(s_close, fast=12, slow=26, signal=9)
    df['macd'] = to_series_safe(macd.get('MACD_12_26_9') if isinstance(macd, dict) else (macd.iloc[:,0] if hasattr(macd, 'shape') else None), idx)
    df['macd_sig'] = to_series_safe(macd.get('MACDs_12_26_9') if isinstance(macd, dict) else (macd.iloc[:,1] if hasattr(macd, 'shape') and macd.shape[1]>1 else None), idx)
    df['macd_hist'] = to_series_safe(macd.get('MACDh_12_26_9') if isinstance(macd, dict) else (macd.iloc[:,2] if hasattr(macd, 'shape') and macd.shape[1]>2 else None), idx)

    # 17-22: volatility / bands
    bb = ta.bbands(s_close, length=20, std=2.0)
    df['bb_lower'] = to_series_safe(bb.get("BBL_20_2.0") if isinstance(bb, dict) else (bb.iloc[:,0] if hasattr(bb,'shape') else None), idx)
    df['bb_mid'] = to_series_safe(bb.get("BBM_20_2.0") if isinstance(bb, dict) else (bb.iloc[:,1] if hasattr(bb,'shape') and bb.shape[1]>1 else None), idx)
    df['bb_upper'] = to_series_safe(bb.get("BBU_20_2.0") if isinstance(bb, dict) else (bb.iloc[:,2] if hasattr(bb,'shape') and bb.shape[1]>2 else None), idx)
    df['atr14'] = to_series_safe(ta.atr(df['high'], df['low'], s_close, length=ATR_PERIOD), idx)
    # Keltner
    try:
        kelt = ta.kc(s_close, df['high'], df['low'], df['close'], length=20)
        df['kc_lower'] = to_series_safe(kelt.get('KCL_20_2.0') if isinstance(kelt, dict) else (kelt.iloc[:,0] if hasattr(kelt,'shape') else None), idx)
        df['kc_mid'] = to_series_safe(kelt.get('KCM_20_2.0') if isinstance(kelt, dict) else (kelt.iloc[:,1] if hasattr(kelt,'shape') and kelt.shape[1]>1 else None), idx)
        df['kc_upper'] = to_series_safe(kelt.get('KCU_20_2.0') if isinstance(kelt, dict) else (kelt.iloc[:,2] if hasattr(kelt,'shape') and kelt.shape[1]>2 else None), idx)
    except Exception:
        df['kc_lower'] = to_series_safe(None, idx); df['kc_mid'] = to_series_safe(None, idx); df['kc_upper'] = to_series_safe(None, idx)

    # 23-27: oscillators
    stoch = ta.stoch(df['high'], df['low'], s_close, k=14, d=3)
    if isinstance(stoch, dict):
        df['stoch_k'] = to_series_safe(stoch.get('STOCHk_14_3_3'), idx)
        df['stoch_d'] = to_series_safe(stoch.get('STOCHd_14_3_3'), idx)
    else:
        df['stoch_k'] = to_series_safe(stoch.iloc[:,0] if hasattr(stoch,'shape') else None, idx)
        df['stoch_d'] = to_series_safe(stoch.iloc[:,1] if hasattr(stoch,'shape') and stoch.shape[1]>1 else None, idx)
    df['cci20'] = to_series_safe(ta.cci(df['high'], df['low'], s_close, length=20), idx)
    df['wpr14'] = to_series_safe(ta.willr(df['high'], df['low'], s_close, length=14), idx)
    adx = ta.adx(df['high'], df['low'], s_close, length=14)
    if isinstance(adx, dict):
        df['adx14'] = to_series_safe(adx.get('ADX_14'), idx)
    else:
        df['adx14'] = to_series_safe(adx.iloc[:,2] if hasattr(adx,'shape') and adx.shape[1]>2 else None, idx)

    # 28-33: volume/flow/special
    df['obv'] = to_series_safe(ta.obv(s_close, df['volume']), idx)
    try:
        efi_try = ta.efi(high=df['high'], low=df['low'], close=s_close, volume=df['volume'], length=13)
    except Exception:
        try:
            efi_try = ta.efi(s_close, df['volume'], length=13)
        except Exception:
            efi_try = None
    df['efi'] = to_series_safe(efi_try, idx)

    ao_try = None
    try:
        ao_try = ta.ao(df['high'], df['low'])
    except Exception:
        ao_try = None
    df['ao'] = to_series_safe(ao_try, idx)

    # VWAP: cumulative typical price * volume / cumulative volume
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    vtp = tp * df['volume']
    c_vtp = vtp.cumsum()
    c_vol = df['volume'].cumsum().replace(0, np.nan)
    df['vwap'] = (c_vtp / c_vol).fillna(method='ffill').fillna(df['close']).astype('float64')

    # Donchian channels
    dc_high = df['high'].rolling(20).max()
    dc_low = df['low'].rolling(20).min()
    df['donch_high'] = dc_high.fillna(method='bfill').astype('float64')
    df['donch_low'] = dc_low.fillna(method='bfill').astype('float64')

    # Chaikin AD (ADL) and Chaikin oscillator
    try:
        ad = ta.ad(df['high'], df['low'], df['close'], df['volume'])
        df['adl'] = to_series_safe(ad, idx)
    except Exception:
        df['adl'] = to_series_safe(None, idx)

    # PSAR (самописный)
    try:
        df['psar'] = parabolic_sar(df['high'], df['low'])
    except Exception:
        df['psar'] = to_series_safe(None, idx)

    # Derived features
    df['ema20_50'] = (df['ema20'] - df['ema50']).astype('float64') if 'ema20' in df.columns and 'ema50' in df.columns else to_series_safe(None, idx)
    df['price_vs_bbmid'] = ((s_close - df['bb_mid']) / (df['bb_mid'] + 1e-9)).astype('float64')

    # Final clean
    df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0)
    for c in df.columns:
        try:
            df[c] = df[c].astype('float64')
        except Exception:
            pass
    return df

# -------------------------
# Метки (3-класса) и обучение
# -------------------------
def make_labels(df: pd.DataFrame, lookahead: int = LOOKAHEAD, thr_up: float = THR_UP, thr_down: float = THR_DOWN) -> pd.Series:
    future = df['close'].shift(-lookahead)
    ret = (future - df['close']) / df['close']
    conds = [
        (ret <= thr_down),
        (ret > thr_down) & (ret < thr_up),
        (ret >= thr_up)
    ]
    choices = [0, 1, 2]  # 0=SELL,1=HOLD,2=BUY
    labels = np.select(conds, choices, default=1)
    return pd.Series(labels, index=df.index)

def train_models(feature_df: pd.DataFrame, labels: pd.Series):
    feats = [c for c in feature_df.columns if c not in ['open','high','low','close','volume','time']]
    # choose a subset of robust features (avoid columns with NaN)
    feats = [f for f in feats if feature_df[f].isnull().sum() == 0]
    valid_idx = labels.dropna().index
    X = feature_df.loc[valid_idx, feats].values
    y = labels.loc[valid_idx].values
    if len(y) < MIN_TRAIN_SAMPLES:
        return None, None, feats, {'error':'not enough samples', 'samples': len(y)}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    mlp = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=500, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    try:
        mlp.fit(X_train, y_train)
    except Exception:
        mlp = None
    y_pred = rf.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'test_samples': int(len(y_test))
    }
    try:
        fi = dict(zip(feats, rf.feature_importances_.round(4)))
        metrics['feature_importances'] = {k:v for k,v in sorted(fi.items(), key=lambda x:-x[1])[:15]}
    except Exception:
        metrics['feature_importances'] = {}
    return rf, mlp, feats, metrics

# -------------------------
# Индикаторное голосование
# -------------------------
def indicator_votes(df: pd.DataFrame, idx: pd.Timestamp) -> Tuple[float,List[str]]:
    row = df.loc[idx]
    votes = 0.0
    reasons: List[str] = []

    # EMA cluster
    try:
        if row['ema5'] > row['ema13'] > row['ema34'] if 'ema34' in row.index else row.get('ema21',0):
            pass
    except Exception:
        pass

    # Simple weighted rules (пример)
    try:
        if row['ema8'] > row['ema21']:
            votes += 1; reasons.append("Короткая EMA выше долгой")
        else:
            votes -= 1; reasons.append("Короткая EMA ниже долгой")
    except Exception:
        pass

    try:
        if row['close'] > row['vwap']:
            votes += 0.6; reasons.append("Цена выше VWAP")
        else:
            votes -= 0.6; reasons.append("Цена ниже VWAP")
    except Exception:
        pass

    try:
        if row['rsi14'] < 30:
            votes += 1.0; reasons.append("RSI перепродан")
        elif row['rsi14'] > 70:
            votes -= 1.0; reasons.append("RSI перекуплен")
    except Exception:
        pass

    try:
        if row['macd_hist'] > 0:
            votes += 0.8; reasons.append("MACD гист положителен")
        else:
            votes -= 0.8; reasons.append("MACD гист отрицателен")
    except Exception:
        pass

    try:
        if row['stoch_k'] < 20 and row['stoch_d'] < 20:
            votes += 0.6; reasons.append("Stoch перепродан")
        elif row['stoch_k'] > 80 and row['stoch_d'] > 80:
            votes -= 0.6; reasons.append("Stoch перекуплен")
    except Exception:
        pass

    try:
        if row['obv'] > df['obv'].iloc[-6]:
            votes += 0.5; reasons.append("OBV растёт")
        else:
            votes -= 0.5; reasons.append("OBV падает")
    except Exception:
        pass

    try:
        if row['cci20'] < -100:
            votes += 0.4; reasons.append("CCI сильно низкий")
        elif row['cci20'] > 100:
            votes -= 0.4; reasons.append("CCI сильно высокий")
    except Exception:
        pass

    try:
        if row['close'] < row['bb_lower']:
            votes += 0.5; reasons.append("Цена ниже нижней полосы BB")
        elif row['close'] > row['bb_upper']:
            votes -= 0.5; reasons.append("Цена выше верхней полосы BB")
    except Exception:
        pass

    try:
        if row['psar'] < row['close']:
            votes += 0.4; reasons.append("PSAR ниже цены (бычье)")
        else:
            votes -= 0.4; reasons.append("PSAR выше цены (медвежье)")
    except Exception:
        pass

    # дополнительные проверки
    try:
        if row['atr14'] > df['atr14'].rolling(50).mean().iloc[-1]:
            reasons.append("ATR выше среднего — высокая волатильность")
    except Exception:
        pass

    return votes, reasons

# -------------------------
# SL/TP и лот
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
# Dataclass Result
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
# Основной анализатор
# -------------------------
def analyze(symbol: str, timeframe_value: int, bars: int = BARS) -> Result:
    df = get_rates(symbol, timeframe_value, bars)
    info = mt5.symbol_info(symbol)
    df_ind = add_indicators(df)

    labels = make_labels(df_ind)
    rf, mlp, feats, ml_metrics = train_models(df_ind, labels)

    latest_idx = df_ind.index[-1]
    model_probs = {'SELL':0.33,'HOLD':0.34,'BUY':0.33}
    if rf is not None:
        try:
            X_latest = df_ind.loc[[latest_idx], feats].values
            proba_rf = rf.predict_proba(X_latest)[0]
            # map according to rf.classes_
            for i, cls in enumerate(rf.classes_):
                if int(cls) == 0:
                    model_probs['SELL'] = float(proba_rf[i])
                elif int(cls) == 1:
                    model_probs['HOLD'] = float(proba_rf[i])
                elif int(cls) == 2:
                    model_probs['BUY'] = float(proba_rf[i])
            # mlp probabilities (if exists)
            if mlp is not None:
                try:
                    proba_mlp = mlp.predict_proba(X_latest)[0]
                    # combine RF + MLP (усреднение)
                    comb = {}
                    for i, cls in enumerate(rf.classes_):
                        label = 'SELL' if int(cls)==0 else ('HOLD' if int(cls)==1 else 'BUY')
                        comb[label] = 0.5 * model_probs[label] + 0.5 * float(proba_mlp[i])
                    model_probs = comb
                except Exception:
                    pass
        except Exception as e:
            print("[analyze] model predict error:", e)

    # indicator votes
    votes_val, reasons = indicator_votes(df_ind, latest_idx)

    # normalize votes and combine with ML
    max_votes = 6.0
    ind_component = max(min(votes_val / max_votes, 1.0), -1.0)
    ml_score = model_probs['BUY'] - model_probs['SELL']
    combined_score = ML_WEIGHT * ml_score + IND_WEIGHT * ind_component

    if combined_score > 0.18:
        agg_signal = "ПОКУПАТЬ"
    elif combined_score < -0.18:
        agg_signal = "ПРОДАВАТЬ"
    else:
        agg_signal = "ДЕРЖАТЬ"

    agg_confidence = float(min(99.0, abs(combined_score) * 100))

    price = float(df_ind['close'].iat[-1])
    atr_val = float(df_ind['atr14'].iat[-1]) if 'atr14' in df_ind.columns else None
    account = mt5.account_info()
    balance = float(account.balance) if account else 1000.0

    sl = tp1 = tp2 = tp3 = None; lots = None
    if agg_signal in ('ПОКУПАТЬ','ПРОДАВАТЬ'):
        try:
            side = 'BUY' if agg_signal=='ПОКУПАТЬ' else 'SELL'
            sl, tp1, tp2, tp3, lots = calc_sl_tp_and_lot(info, price, side, atr_val, balance)
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
        sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        lots=lots,
        ml_metrics=ml_metrics or {},
        votes=votes_list
    )

# -------------------------
# Логирование сигналов
# -------------------------
def log_signal(res: Result):
    row = {
        'time': pd.Timestamp.now(),
        'symbol': res.symbol,
        'tf': res.timeframe,
        'price': res.price,
        'signal': res.agg_signal,
        'conf': res.agg_confidence,
        'sl': res.sl,
        'tp1': res.tp1,
        'tp2': res.tp2,
        'tp3': res.tp3,
        'lots': res.lots
    }
    df = pd.DataFrame([row])
    if not os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(LOG_CSV, index=False, mode='a', header=False, encoding='utf-8-sig')

# -------------------------
# GUI (русский) — простой
# -------------------------
class RussianGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализатор XAUUSD — ИИ трейдер (русский интерфейс)")
        self.resize(1000, 760)
        layout = QVBoxLayout()

        top = QHBoxLayout()
        top.addWidget(QLabel("ТФ:"))
        self.tf = QComboBox()
        self.tf.addItems(['M1','M5','M15','M30','H1','H4','D1'])
        self.tf.setCurrentText(DEFAULT_TF)
        top.addWidget(self.tf)

        self.btn = QPushButton("Анализировать")
        top.addWidget(self.btn)

        self.auto = QCheckBox("Авто")
        top.addWidget(self.auto)

        top.addWidget(QLabel("Интервал (с):"))
        self.intspin = QSpinBox(); self.intspin.setRange(5,3600); self.intspin.setValue(60)
        top.addWidget(self.intspin)

        top.addWidget(QLabel("Свечей:"))
        self.barspin = QSpinBox(); self.barspin.setRange(200,5000); self.barspin.setValue(BARS)
        top.addWidget(self.barspin)

        self.save_btn = QPushButton("Сохранить логи...")
        top.addWidget(self.save_btn)

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
        self.save_btn.clicked.connect(self.save_logs)

        # init MT5
        try:
            mt5_init_or_raise()
            self.symbol = find_symbol(BASE_SYMBOL)
            if self.symbol != BASE_SYMBOL:
                self.out.append(f"[Инфо] Используется символ: {self.symbol}")
        except Exception as e:
            self.out.append(f"[Ошибка MT5] {e}")
            self.symbol = BASE_SYMBOL

    def toggle_timer(self):
        if self.auto.isChecked():
            self.timer.start(self.intspin.value()*1000)
        else:
            self.timer.stop()

    def save_logs(self):
        if not os.path.exists(LOG_CSV):
            self.out.append("Нет логов для сохранения.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Сохранить логи как...", LOG_CSV, "CSV Files (*.csv)")
        if fname:
            try:
                import shutil
                shutil.copyfile(LOG_CSV, fname)
                self.out.append(f"Логи сохранены в {fname}")
            except Exception as e:
                self.out.append(f"Ошибка при сохранении: {e}")

    def on_click(self):
        try:
            self.progress.setValue(5)
            tf_map = {'M1':mt5.TIMEFRAME_M1,'M5':mt5.TIMEFRAME_M5,'M15':mt5.TIMEFRAME_M15,'M30':mt5.TIMEFRAME_M30,'H1':mt5.TIMEFRAME_H1,'H4':mt5.TIMEFRAME_H4,'D1':mt5.TIMEFRAME_D1}
            tf_val = tf_map[self.tf.currentText()]
            bars = int(self.barspin.value())
            self.progress.setValue(15)
            res = analyze(self.symbol, tf_val, bars)
            self.progress.setValue(80)

            lines = []
            lines.append(f"Символ: {res.symbol}   ТФ: {self.tf.currentText()}")
            lines.append(f"Цена: {res.price:.4f}")
            lines.append(f"ИИ вероятность -> BUY {res.model_preds.get('BUY',0):.3f}  HOLD {res.model_preds.get('HOLD',0):.3f}  SELL {res.model_preds.get('SELL',0):.3f}")
            lines.append(f"Финальный сигнал: {res.agg_signal} | Доверие: {res.agg_confidence:.1f}%")
            if res.sl is not None:
                lines.append(f"Вход: ~{res.price:.4f}")
                lines.append(f"SL: {res.sl:.4f}")
                lines.append(f"TP1: {res.tp1:.4f} (ATR x{TP1_ATR})")
                lines.append(f"TP2: {res.tp2:.4f} (ATR x{TP2_ATR})")
                lines.append(f"TP3: {res.tp3:.4f} (ATR x{TP3_ATR})")
                lines.append(f"Рекомендованный лот: {res.lots}")
            lines.append("\nМетрики модели:")
            for k,v in res.ml_metrics.items():
                lines.append(f"  {k}: {v}")
            lines.append("\nГолоса индикаторов:")
            for v in res.votes:
                lines.append(f"  {v[0]}: {v[1]}")
            lines.append("\nОТКАЗ ОТ ОТВЕТСТВЕННОСТИ: НЕТ ГАРАНТИЙ. ТЕСТИРУЙ НА ДЕМО.")
            self.out.setPlainText("\n".join(lines))
            self.progress.setValue(100)

            # лог сигналов
            try:
                log_signal(res)
            except Exception:
                pass

        except Exception as e:
            self.out.append(f"[Ошибка] {e}\n{traceback.format_exc()}")
            self.progress.setValue(0)

# -------------------------
# Запуск
# -------------------------
def main():
    app = QApplication(sys.argv)
    win = RussianGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
