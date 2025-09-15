# -*- coding: utf-8 -*-
"""
TraderPro AI — Полный исправленный однофайловый скрипт (русский)
Версия: улучшенная, с защитаxми, ML, GUI, логами, дампами, автопоиском символа.
Цель: минимизировать ошибки "missing 'close'" и "Terminal: Call failed".
Автор: сгенерировано ChatGPT (адаптируй под свои нужды).
Дата: 2025-08-27
"""
from __future__ import annotations

# ============================
# Библиотеки и внешние пакеты
# ============================
import sys
import os
import math
import time
import traceback
import threading
import queue
import json
import shutil
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

# Optional heavy dependencies - мы делаем их опциональными и даём fallback
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except Exception:
    mt5 = None
    _MT5_AVAILABLE = False

try:
    import pandas_ta as ta
    _PANDAS_TA_AVAILABLE = True
except Exception:
    ta = None
    _PANDAS_TA_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

try:
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTextEdit, QComboBox, QSpinBox, QCheckBox, QProgressBar, QFileDialog,
        QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QTabWidget, QFormLayout,
        QGroupBox
    )
    from PySide6.QtCore import QTimer, Qt, Signal, QObject
    from PySide6.QtGui import QColor
    _PYSIDE_AVAILABLE = True
except Exception:
    _PYSIDE_AVAILABLE = False

# ============================
# Конфигурация (редактируй как нужно)
# ============================
BASE_SYMBOL = "XAUUSD"
DEFAULT_TF = "M15"
DEFAULT_BARS = 1500  # начальное значение (умеренное)
LOOKAHEAD = 6
THR_UP = 0.0018
THR_DOWN = -0.0018

RISK_PER_TRADE = 0.01
ATR_PERIOD = 14
SL_ATR_MULT = 1.3
TP1_ATR = 1.0
TP2_ATR = 2.0
TP3_ATR = 3.5

RF_ESTIMATORS = 200
MLP_HIDDEN = (96, 48)
RANDOM_STATE = 42
MIN_TRAIN_SAMPLES = 160

ML_WEIGHT = 0.55
IND_WEIGHT = 0.35
NEWS_WEIGHT = 0.10

LOG_CSV = "signals_log.csv"
DEBUG_LOG = "trader_debug.log"
DF_DUMP_ON_ERR = "last_df_dump.pkl"
SETTINGS_FILE = "traderpro_settings.json"

USE_PANDAS_TA_ALL = False and _PANDAS_TA_AVAILABLE
MAX_FEATURES = 800
VAR_THRESHOLD = 1e-9
MAX_CORR = 0.98

MT5_TERMINAL_PATH = None  # При необходимости укажи путь к терминалу: r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Повторные попытки для get_rates
RETRY_COPY_RATES = 3
RETRY_SLEEP = 0.4

# Лимиты защитных проверок
MIN_ROWS_FOR_LABELS = 50
MIN_ROWS_FOR_TRAIN = MIN_TRAIN_SAMPLES

# ============================
# Вспомогательное логирование
# ============================
def debug_log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}"
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line)

debug_log("=== TraderPro AI start ===")

# ============================
# Утилиты
# ============================
def safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"

def to_series_safe(x, index, fill=0.0):
    try:
        if x is None:
            return pd.Series(fill, index=index, dtype='float64')
        if isinstance(x, pd.Series):
            s = x.reindex(index).astype('float64')
            s = s.ffill().bfill().fillna(fill)
            return s
        if isinstance(x, (np.ndarray, list, tuple)):
            arr = np.asarray(x, dtype=float)
            if arr.shape[0] == len(index):
                return pd.Series(arr, index=index, dtype='float64').fillna(fill)
            if arr.shape[0] < len(index):
                pad = np.full(len(index) - arr.shape[0], fill)
                return pd.Series(np.concatenate([pad, arr]), index=index, dtype='float64')
            return pd.Series(arr[-len(index):], index=index, dtype='float64')
        return pd.Series(fill, index=index, dtype='float64')
    except Exception as e:
        debug_log(f"to_series_safe exception: {e}")
        return pd.Series(fill, index=index, dtype='float64')

def ensure_float_cols(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').ffill().bfill().fillna(0).astype('float64')

# ============================
# Parabolic SAR
# ============================
def parabolic_sar(high: pd.Series, low: pd.Series, af0=0.02, af_step=0.02, af_max=0.2):
    n = len(high)
    if n == 0:
        return pd.Series([], dtype='float64')
    sar = np.zeros(n, dtype='float64')
    trend = 1
    af = af0
    ep = high.iloc[0]
    sar[0] = low.iloc[0]
    for i in range(1, n):
        prev = sar[i-1]
        if trend == 1:
            sar[i] = prev + af * (ep - prev)
            sar[i] = min(sar[i], low.iloc[i-1], low.iloc[i-2] if i >= 2 else low.iloc[i-1])
            if low.iloc[i] < sar[i]:
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
            sar[i] = max(sar[i], high.iloc[i-1], high.iloc[i-2] if i >= 2 else high.iloc[i-1])
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

# ============================
# MT5 helpers (robust)
# ============================
def mt5_init_or_raise(login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None, force: bool = False):
    if not _MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package is not installed. Установи через pip: pip install MetaTrader5")
    if force:
        try:
            mt5.shutdown()
        except Exception as e:
            debug_log(f"mt5.shutdown error: {e}")
    ok = False
    try:
        ok = mt5.initialize() if MT5_TERMINAL_PATH is None else mt5.initialize(MT5_TERMINAL_PATH)
    except Exception as e:
        debug_log(f"mt5.initialize exception: {e}")
        try:
            ok = mt5.initialize()
        except Exception as e2:
            debug_log(f"mt5.initialize fallback exception: {e2}")
            ok = False
    if not ok:
        try:
            if MT5_TERMINAL_PATH is not None:
                ok = mt5.initialize(MT5_TERMINAL_PATH)
            else:
                ok = mt5.initialize()
        except Exception as e:
            debug_log(f"mt5.initialize second attempt failed: {e}")
            ok = False
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()} (path={MT5_TERMINAL_PATH})")
    # wait for terminal to warm up
    for _ in range(50):
        ti = mt5.terminal_info()
        ai = mt5.account_info()
        if ti is not None and ai is not None:
            break
        time.sleep(0.1)
    if login and password and server:
        if not mt5.login(login=login, password=password, server=server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    ai = mt5.account_info()
    if ai is None:
        raise RuntimeError(f"MT5 account_info() is None. last_error={mt5.last_error()}")

def list_all_symbols(maxn: int = 500) -> List[str]:
    if not _MT5_AVAILABLE:
        return []
    syms = mt5.symbols_get()
    names = [s.name for s in syms] if syms else []
    names.sort()
    return names[:maxn]

def find_symbol(base: str = BASE_SYMBOL) -> str:
    if not _MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 not available.")
    syms = mt5.symbols_get()
    if not syms:
        raise RuntimeError(f"symbols_get() returned empty. last_error={mt5.last_error()}")
    names = [s.name for s in syms]
    base_norm = base.strip().lower()
    # exact
    for n in names:
        if n.lower() == base_norm:
            debug_log(f"find_symbol: exact match -> {n}")
            return n
    # startswith
    for n in names:
        if n.lower().startswith(base_norm):
            debug_log(f"find_symbol: startswith -> {n}")
            return n
    # contains
    for n in names:
        if base_norm in n.lower():
            debug_log(f"find_symbol: contains -> {n}")
            return n
    # alt variations
    alt = base_norm.replace(' ', '').replace('-', '').replace('_', '')
    for n in names:
        if alt in n.lower().replace(' ', '').replace('-', '').replace('_', ''):
            debug_log(f"find_symbol: alt contains -> {n}")
            return n
    # fallback
    debug_log(f"find_symbol: fallback -> {names[0]}")
    return names[0]

def _ensure_symbol_selected(symbol: str, timeout_sec: float = 3.0):
    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        pass
    start = time.time()
    while time.time() - start < timeout_sec:
        tick = mt5.symbol_info_tick(symbol)
        si = mt5.symbol_info(symbol)
        if tick is not None and si is not None and getattr(si, 'visible', True):
            return
        time.sleep(0.1)

def get_rates(symbol: str, timeframe: int, bars: int, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Robust wrapper for mt5.copy_rates_from_pos().
    Returns DataFrame with index=time and float columns ['open','high','low','close','volume'] if possible.
    If 'close' missing, tries to build fallback; if impossible, returns None.
    """
    if not _MT5_AVAILABLE:
        debug_log("get_rates: MT5 module not available.")
        return None
    _ensure_symbol_selected(symbol)
    plan = [bars, max(1200, bars // 2), max(700, bars // 4), 500, 300]
    errors = []
    for want in plan:
        for attempt in range(RETRY_COPY_RATES):
            rates = None
            try:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(want))
            except Exception as e:
                debug_log(f"copy_rates_from_pos exception: {e}")
                rates = None
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                # normalize time column
                try:
                    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(None)
                except Exception:
                    try:
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                    except Exception:
                        pass
                # Build df2 with expected names
                try:
                    col_map = {c.lower(): c for c in df.columns}
                    df2 = pd.DataFrame(index=df.index)
                    if 'time' in df.columns:
                        df2['time'] = df['time']
                        df2 = df2.set_index('time')
                    else:
                        # fallback to index
                        pass
                    # open/high/low
                    if 'open' in col_map:
                        df2['open'] = df[col_map['open']]
                    if 'high' in col_map:
                        df2['high'] = df[col_map['high']]
                    if 'low' in col_map:
                        df2['low'] = df[col_map['low']]
                    # close - try several fallbacks
                    if 'close' in col_map:
                        df2['close'] = df[col_map['close']]
                    elif 'last' in col_map:
                        df2['close'] = df[col_map['last']]
                    else:
                        # try mean of open/high/low if available
                        if 'open' in df2.columns and 'high' in df2.columns and 'low' in df2.columns:
                            df2['close'] = (df2['open'] + df2['high'] + df2['low']) / 3.0
                    # volume
                    if 'tick_volume' in col_map:
                        df2['volume'] = df[col_map['tick_volume']]
                    elif 'real_volume' in col_map:
                        df2['volume'] = df[col_map['real_volume']]
                    else:
                        df2['volume'] = 0.0
                except Exception as e:
                    debug_log(f"get_rates build exception: {e}")
                    df2 = pd.DataFrame(index=df.index)
                # cleanup
                try:
                    for c in ['open','high','low','close','volume']:
                        if c in df2.columns:
                            df2[c] = pd.to_numeric(df2[c], errors='coerce')
                    df2 = df2.ffill().bfill().fillna(0.0)
                    for c in df2.columns:
                        try:
                            df2[c] = df2[c].astype('float64')
                        except Exception:
                            pass
                except Exception as e:
                    debug_log(f"get_rates cleanup exception: {e}")
                # verify 'close'
                if 'close' not in df2.columns or df2['close'].isnull().all():
                    debug_log(f"get_rates: 'close' missing after build. original cols: {list(df.columns)}")
                    errors.append(("no_close", list(df.columns)))
                    time.sleep(RETRY_SLEEP)
                    continue
                debug_log(f"get_rates success: symbol={symbol} timeframe={timeframe} bars={len(df2)}")
                return df2
            # nothing returned
            try:
                errors.append(mt5.last_error())
            except Exception:
                errors.append(("unknown", "no mt5.last_error"))
            time.sleep(RETRY_SLEEP + 0.15 * attempt)
    debug_log(f"get_rates failed for {symbol}, tf={timeframe}, bars~{bars}. Last errors: {errors[-5:] if errors else 'n/a'}")
    return None

# ============================
# Индикаторный движок
# ============================
def add_indicators(df: pd.DataFrame, use_all_strategy: bool = USE_PANDAS_TA_ALL) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    ensure_float_cols(df, ['open','high','low','close','volume'])
    idx = df.index
    close = df['close']

    # EMA
    try:
        df['ema5'] = close.ewm(span=5, adjust=False).mean()
        df['ema8'] = close.ewm(span=8, adjust=False).mean()
        df['ema13'] = close.ewm(span=13, adjust=False).mean()
        df['ema21'] = close.ewm(span=21, adjust=False).mean()
        df['ema34'] = close.ewm(span=34, adjust=False).mean()
        df['ema50'] = close.ewm(span=50, adjust=False).mean()
        df['ema100'] = close.ewm(span=100, adjust=False).mean()
        df['ema200'] = close.ewm(span=200, adjust=False).mean()
    except Exception as e:
        debug_log(f"EMA error: {e}")

    # RSI
    try:
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / (down + 1e-12)
        df['rsi14'] = (100 - (100 / (1 + rs))).fillna(50.0)
    except Exception as e:
        debug_log(f"RSI error: {e}")

    # MACD
    try:
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal = macd_line.ewm(span=9, adjust=False).mean()
        df['macd'] = macd_line
        df['macd_sig'] = signal
        df['macd_hist'] = macd_line - signal
    except Exception as e:
        debug_log(f"MACD error: {e}")

    # Bollinger Bands
    try:
        ma = close.rolling(20).mean()
        sd = close.rolling(20).std()
        df['bb_mid'] = ma
        df['bb_upper'] = ma + 2 * sd
        df['bb_lower'] = ma - 2 * sd
    except Exception as e:
        debug_log(f"BB error: {e}")

    # ATR
    try:
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(ATR_PERIOD).mean().fillna(method='ffill').fillna(0.0)
    except Exception as e:
        debug_log(f"ATR error: {e}")

    # VWAP
    try:
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        vtp = tp * df['volume']
        c_vtp = vtp.cumsum()
        c_vol = df['volume'].cumsum().replace(0, np.nan)
        df['vwap'] = (c_vtp / c_vol).ffill().bfill().fillna(df['close']).astype('float64')
    except Exception as e:
        debug_log(f"VWAP error: {e}")
        df['vwap'] = df['close']

    # Donchian
    try:
        df['donch_high'] = df['high'].rolling(20).max().bfill().astype('float64')
        df['donch_low'] = df['low'].rolling(20).min().bfill().astype('float64')
    except Exception as e:
        debug_log(f"Donchian error: {e}")

    # ADL
    try:
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-12)
        df['adl'] = (clv * df['volume']).cumsum()
    except Exception as e:
        debug_log(f"ADL error: {e}")

    # PSAR
    try:
        df['psar'] = parabolic_sar(df['high'], df['low'])
    except Exception as e:
        debug_log(f"PSAR error: {e}")

    # derived
    try:
        if 'bb_mid' in df.columns:
            df['price_vs_bbmid'] = ((df['close'] - df['bb_mid']) / (df['bb_mid'] + 1e-9)).astype('float64')
    except Exception:
        pass

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    # limit features
    if len(df.columns) > MAX_FEATURES:
        base_cols = ['open','high','low','close','volume','rsi14','macd','macd_sig','macd_hist','atr14','vwap','ema5','ema8','ema13','ema21','ema34','ema50','ema100','ema200','psar']
        others = [c for c in df.columns if c not in base_cols]
        keep_others = others[:max(0, MAX_FEATURES - len(base_cols))]
        df = df[[c for c in (base_cols + keep_others) if c in df.columns]]

    # variance & correlation filters
    try:
        variances = df.var(axis=0, skipna=True)
        df = df.loc[:, variances > VAR_THRESHOLD]
    except Exception as e:
        debug_log(f"variance filter error: {e}")
    try:
        corr = df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > MAX_CORR)]
        df = df.drop(columns=to_drop, errors='ignore')
    except Exception as e:
        debug_log(f"corr filter error: {e}")

    for c in df.columns:
        try:
            df[c] = df[c].astype('float64')
        except Exception:
            pass

    return df

# ============================
# Labels & ML
# ============================
def make_labels(df: pd.DataFrame, lookahead: int = LOOKAHEAD, thr_up: float = THR_UP, thr_down: float = THR_DOWN) -> Optional[pd.Series]:
    if df is None or 'close' not in df.columns:
        debug_log("make_labels: df is None or 'close' missing")
        return None
    if len(df) < max(lookahead + 5, MIN_ROWS_FOR_LABELS):
        debug_log(f"make_labels: insufficient rows {len(df)}")
        return None
    future = df['close'].shift(-lookahead)
    ret = (future - df['close']) / (df['close'] + 1e-12)
    labels = np.where(ret <= thr_down, 0, np.where(ret >= thr_up, 2, 1))  # 0=SELL,1=HOLD,2=BUY
    return pd.Series(labels, index=df.index, dtype='int64')

def train_models(feature_df: pd.DataFrame, labels: pd.Series):
    if not _SKLEARN_AVAILABLE:
        return None, None, [], {'error': 'sklearn not installed'}
    feats = [c for c in feature_df.columns if c not in ['open','high','low','close','volume']]
    feats = [f for f in feats if feature_df[f].isnull().sum() == 0]
    valid_idx = labels.dropna().index if labels is not None else []
    if len(valid_idx) == 0:
        return None, None, feats, {'error': 'no valid samples', 'samples': 0}
    X = feature_df.loc[valid_idx, feats].values
    y = labels.loc[valid_idx].values
    if len(y) < MIN_TRAIN_SAMPLES:
        return None, None, feats, {'error': 'not enough samples', 'samples': int(len(y))}
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
    except Exception as e:
        debug_log(f"train_test_split error: {e}")
        return None, None, feats, {'error': 'split error', 'exception': safe_str(e)}
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    mlp = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=500, random_state=RANDOM_STATE)
    try:
        rf.fit(X_train, y_train)
    except Exception as e:
        debug_log(f"rf.fit error: {e}")
        return None, None, feats, {'error': 'rf fit failed', 'exception': safe_str(e)}
    try:
        mlp.fit(X_train, y_train)
    except Exception as e:
        debug_log(f"mlp.fit error: {e}")
        mlp = None
    y_pred = rf.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'test_samples': int(len(y_test))
    }
    try:
        fi = dict(zip(feats, rf.feature_importances_.round(5)))
        metrics['feature_importances'] = {k: v for k, v in sorted(fi.items(), key=lambda x: -x[1])[:20]}
    except Exception:
        metrics['feature_importances'] = {}
    return rf, mlp, feats, metrics

# ============================
# Expert60 rules
# ============================
def expert60_votes(df: pd.DataFrame, idx: pd.Timestamp) -> Tuple[float, List[str]]:
    if df is None or idx not in df.index:
        return 0.0, ["Нет данных для Expert60"]
    r = df.loc[idx]
    votes = 0.0
    reasons: List[str] = []
    try:
        bull = r.get('ema8', 0) > r.get('ema21', 0) > r.get('ema50', 0)
        bear = r.get('ema8', 0) < r.get('ema21', 0) < r.get('ema50', 0)
        if bull:
            votes += 1.2; reasons.append("Тренд бычий по EMA (8>21>50)")
        elif bear:
            votes -= 1.2; reasons.append("Тренд медвежий по EMA (8<21<50)")
    except Exception:
        pass
    try:
        if r.get('macd_hist', 0) > 0:
            votes += 0.8; reasons.append("MACD гистограмма > 0")
        else:
            votes -= 0.8; reasons.append("MACD гистограмма < 0")
    except Exception:
        pass
    try:
        if r.get('rsi14', 50) < 28:
            votes += 0.7; reasons.append("RSI перепродан")
        elif r.get('rsi14', 50) > 72:
            votes -= 0.7; reasons.append("RSI перекуплен")
    except Exception:
        pass
    try:
        if r.get('close', 0) > r.get('vwap', r.get('close', 0)):
            votes += 0.6; reasons.append("Цена выше VWAP")
        else:
            votes -= 0.6; reasons.append("Цена ниже VWAP")
    except Exception:
        pass
    try:
        if r.get('close', 0) < r.get('bb_lower', r.get('close', 0) - 1):
            votes += 0.5; reasons.append("Пробой нижней BB — возможный отскок")
        elif r.get('close', 0) > r.get('bb_upper', r.get('close', 0) + 1):
            votes -= 0.5; reasons.append("Пробой верхней BB — риск коррекции")
    except Exception:
        pass
    try:
        if r.get('psar', 0) < r.get('close', 0):
            votes += 0.4; reasons.append("PSAR под ценой (бычий контекст)")
        else:
            votes -= 0.4; reasons.append("PSAR над ценой (медвежий контекст)")
    except Exception:
        pass
    try:
        atr = float(r.get('atr14', 0))
        atr_ma = float(df['atr14'].rolling(50).mean().iloc[-1]) if 'atr14' in df.columns else atr
        if atr_ma > 0 and atr > 1.2 * atr_ma:
            reasons.append("Повышенная волатильность — осторожнее с размером позиции")
    except Exception:
        pass
    return votes, reasons

# ============================
# NewsWatcher (local)
# ============================
class NewsWatcher:
    KEYWORDS_BULL = ['геополит','риск','инфляц','замедление доллара','data weak','rate cut','dovish']
    KEYWORDS_BEAR = ['повышение ставки','rate hike','strong dollar','hawkish','nonfarm strong','cpi beat']
    def __init__(self):
        self.queue = queue.Queue()
        self.last_sentiment = 0.0
    def ingest_text(self, text: str):
        text_l = text.lower()
        score = 0
        for w in self.KEYWORDS_BULL:
            if w in text_l:
                score += 1
        for w in self.KEYWORDS_BEAR:
            if w in text_l:
                score -= 1
        self.last_sentiment = float(max(-3, min(3, score)) / 3.0)
    def get_sentiment(self) -> float:
        return self.last_sentiment
NEWS = NewsWatcher()

# ============================
# TP/SL and lot calculation
# ============================
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
    lots = 0.0 if stop_value_per_lot <= 0 else max(getattr(info, 'volume_min', 0.01) or 0.01, risk_amount / stop_value_per_lot)
    vol_step = getattr(info, 'volume_step', 0.01) or 0.01
    lots = math.floor(lots / vol_step) * vol_step
    lots = round(lots, 2)
    return float(sl), float(tp1), float(tp2), float(tp3), lots

# ============================
# Result dataclass
# ============================
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
    reasons: List[str]

# ============================
# Main analyze pipeline
# ============================
def analyze(symbol: str, timeframe_value: int, bars: int = DEFAULT_BARS) -> Result:
    df = get_rates(symbol, timeframe_value, bars)
    if df is None or df.empty:
        raise RuntimeError(f"Не удалось загрузить данные для {symbol}. Проверь MT5, Market Watch и логин.")
    # dump if problematic
    if 'close' not in df.columns or len(df) < MIN_ROWS_FOR_LABELS:
        try:
            pd.to_pickle(df, DF_DUMP_ON_ERR)
            debug_log(f"analyze: dumped df to {DF_DUMP_ON_ERR}")
        except Exception:
            pass
        raise RuntimeError("Labels cannot be created: missing 'close' column or insufficient data.")
    info = None
    try:
        info = mt5.symbol_info(symbol)
    except Exception:
        pass
    if info is None:
        debug_log(f"symbol_info returned None for {symbol}")
    df_ind = add_indicators(df, use_all_strategy=USE_PANDAS_TA_ALL and _PANDAS_TA_AVAILABLE)
    labels = make_labels(df_ind)
    if labels is None:
        try:
            pd.to_pickle(df_ind, DF_DUMP_ON_ERR)
            debug_log(f"analyze: dumped df_ind to {DF_DUMP_ON_ERR}")
        except Exception:
            pass
        raise RuntimeError("Labels cannot be created: missing 'close' column or insufficient data.")
    rf = mlp = None
    feats = []
    ml_metrics = {}
    if _SKLEARN_AVAILABLE:
        try:
            rf, mlp, feats, ml_metrics = train_models(df_ind, labels)
        except Exception as e:
            debug_log(f"train_models exception: {e}")
            rf = mlp = None
    latest_idx = df_ind.index[-1]
    probs = {'SELL':0.33,'HOLD':0.34,'BUY':0.33}
    if rf is not None:
        try:
            X_latest = df_ind.loc[[latest_idx], [f for f in feats if f in df_ind.columns]].values
            proba_rf = rf.predict_proba(X_latest)[0]
            cls_map = {int(c): p for c,p in zip(rf.classes_, proba_rf)}
            probs['SELL'] = float(cls_map.get(0, probs['SELL']))
            probs['HOLD'] = float(cls_map.get(1, probs['HOLD']))
            probs['BUY']  = float(cls_map.get(2, probs['BUY']))
            if mlp is not None:
                try:
                    proba_mlp = mlp.predict_proba(X_latest)[0]
                    for i, c in enumerate(rf.classes_):
                        lab = 'SELL' if int(c)==0 else ('HOLD' if int(c)==1 else 'BUY')
                        probs[lab] = 0.5 * probs[lab] + 0.5 * float(proba_mlp[i])
                except Exception as e:
                    debug_log(f"mlp predict_proba error: {e}")
        except Exception as e:
            debug_log(f"model predict error: {e}")
    expert_score, expert_reasons = expert60_votes(df_ind, latest_idx)
    ind_component = max(-1.0, min(1.0, expert_score / 6.0))
    news_component = max(-1.0, min(1.0, NEWS.get_sentiment()))
    ml_score = probs['BUY'] - probs['SELL']
    combined_score = ML_WEIGHT * ml_score + IND_WEIGHT * ind_component + NEWS_WEIGHT * news_component
    if combined_score > 0.18:
        agg_signal = "ПОКУПАТЬ"
    elif combined_score < -0.18:
        agg_signal = "ПРОДАВАТЬ"
    else:
        agg_signal = "ДЕРЖАТЬ"
    agg_confidence = float(min(99.0, abs(combined_score) * 100))
    price = float(df_ind['close'].iat[-1])
    atr_val = float(df_ind['atr14'].iat[-1]) if 'atr14' in df_ind.columns else None
    account = None
    try:
        account = mt5.account_info()
    except Exception:
        pass
    balance = float(getattr(account, 'balance', 1000.0))
    sl = tp1 = tp2 = tp3 = None; lots = None
    if agg_signal in ("ПОКУПАТЬ","ПРОДАВАТЬ"):
        try:
            side = 'BUY' if agg_signal == 'ПОКУПАТЬ' else 'SELL'
            sl, tp1, tp2, tp3, lots = calc_sl_tp_and_lot(info, price, side, atr_val, balance)
        except Exception as e:
            debug_log(f"calc_sl_tp_and_lot error: {e}")
    reasons = [
        f"ML: BUY={probs['BUY']:.3f} HOLD={probs['HOLD']:.3f} SELL={probs['SELL']:.3f}",
        f"Expert60 score={expert_score:.2f} (norm={ind_component:.2f})",
        f"News sentiment={news_component:.2f}"
    ] + expert_reasons
    return Result(
        symbol=symbol,
        timeframe=str(timeframe_value),
        price=price,
        model_preds=probs,
        agg_signal=agg_signal,
        agg_confidence=agg_confidence,
        sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        lots=lots,
        ml_metrics=ml_metrics or {},
        reasons=reasons
    )

# ============================
# Logging (signals)
# ============================
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

# ============================
# GUI: PySide6 (если доступно)
# ============================
APP_QSS = """
QWidget { font-family: Segoe UI, Roboto, sans-serif; font-size: 12.5pt; }
QTextEdit { background: #0f1115; color: #e8e8e8; border: 1px solid #23262d; border-radius: 10px; padding: 8px; }
QComboBox, QSpinBox, QLineEdit { background: #10131a; color: #d9dee7; border: 1px solid #2a2f3a; border-radius: 8px; padding: 4px 8px; }
QPushButton { background: #1e293b; color: #e6edf3; border-radius: 10px; padding: 8px 14px; }
QPushButton:hover { background: #334155; }
QProgressBar { border: 1px solid #2a2f3a; border-radius: 8px; text-align: center; }
QProgressBar::chunk { background-color: #22c55e; }
#SignalLamp { border-radius: 12px; border: 2px solid #111; }
"""

if _PYSIDE_AVAILABLE:
    class WorkerSignals(QObject):
        finished = Signal(object)
        error = Signal(tuple)
        progress = Signal(int)

    class AnalyzerWorker(threading.Thread):
        def __init__(self, symbol, tf_val, bars, signals: WorkerSignals):
            super().__init__()
            self.symbol = symbol
            self.tf_val = tf_val
            self.bars = bars
            self.signals = signals
        def run(self):
            try:
                res = analyze(self.symbol, self.tf_val, self.bars)
                self.signals.finished.emit(res)
            except Exception as e:
                tb = traceback.format_exc()
                self.signals.error.emit((e, tb))

    class TraderProGUI(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("TraderPro AI — XAUUSD/FX (Русский)")
            self.resize(1160, 820)
            self.setStyleSheet(APP_QSS)
            self._init_ui()
            # Try initialize MT5
            try:
                mt5_init_or_raise()
                self.statusL.setText("MT5 инициализирован")
            except Exception as e:
                debug_log(f"GUI mt5 init exception: {e}")
                self.statusL.setText(f"[Ошибка MT5] {e}")
        def _init_ui(self):
            layout = QVBoxLayout()
            # top layout
            top = QHBoxLayout()
            top.addWidget(QLabel("Символ:"))
            self.symbolBox = QComboBox()
            self.symbolBox.setEditable(True)
            try:
                symbols = list_all_symbols(500)
                if BASE_SYMBOL not in symbols:
                    symbols.insert(0, BASE_SYMBOL)
                self.symbolBox.addItems(symbols if symbols else [BASE_SYMBOL])
                # try to pre-resolve
                try:
                    found = find_symbol(BASE_SYMBOL)
                    idx = self.symbolBox.findText(found)
                    if idx >= 0:
                        self.symbolBox.setCurrentIndex(idx)
                except Exception:
                    pass
            except Exception:
                self.symbolBox.addItems([BASE_SYMBOL])
            top.addWidget(self.symbolBox)
            top.addWidget(QLabel("ТФ:"))
            self.tfBox = QComboBox(); self.tfBox.addItems(['M1','M5','M15','M30','H1','H4','D1']); self.tfBox.setCurrentText(DEFAULT_TF)
            top.addWidget(self.tfBox)
            top.addWidget(QLabel("Свечей:"))
            self.barSpin = QSpinBox(); self.barSpin.setRange(300, 8000); self.barSpin.setValue(DEFAULT_BARS)
            top.addWidget(self.barSpin)
            self.btnAnalyze = QPushButton("АНАЛИЗ")
            top.addWidget(self.btnAnalyze)
            self.auto = QCheckBox("Авто")
            top.addWidget(self.auto)
            top.addWidget(QLabel("Интервал (с):"))
            self.intSpin = QSpinBox(); self.intSpin.setRange(5, 3600); self.intSpin.setValue(60)
            top.addWidget(self.intSpin)
            self.saveBtn = QPushButton("Сохранить логи…")
            top.addWidget(self.saveBtn)
            layout.addLayout(top)
            # news input
            news_layout = QHBoxLayout()
            news_layout.addWidget(QLabel("Новости/заголовок →"))
            self.newsEdit = QLineEdit(); self.newsEdit.setPlaceholderText("Вставьте текст новости (например: FOMC dovish/hawkish)…")
            news_layout.addWidget(self.newsEdit)
            self.btnNews = QPushButton("Учитывать новость")
            news_layout.addWidget(self.btnNews)
            layout.addLayout(news_layout)
            # lamp + status
            srow = QHBoxLayout()
            self.lamp = QLabel(); self.lamp.setObjectName("SignalLamp"); self.lamp.setFixedSize(24, 24)
            self._setLampColor("gray")
            self.statusL = QLabel("Готово")
            srow.addWidget(QLabel("Сигнал:")); srow.addWidget(self.lamp); srow.addWidget(self.statusL)
            srow.addStretch(1)
            layout.addLayout(srow)
            # progress
            self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0)
            layout.addWidget(self.progress)
            # text output
            self.out = QTextEdit(); self.out.setReadOnly(True)
            layout.addWidget(self.out)
            # reasons list
            layout.addWidget(QLabel("Обоснование сигнала:"))
            self.reasonsList = QListWidget()
            layout.addWidget(self.reasonsList)
            # advanced options
            opts = QHBoxLayout()
            self.chk_use_pandas_ta = QCheckBox("Использовать pandas_ta.AllStrategy (медленно)")
            self.chk_use_pandas_ta.setChecked(USE_PANDAS_TA_ALL)
            opts.addWidget(self.chk_use_pandas_ta)
            self.chk_save_df_on_err = QCheckBox("Дамп DF при ошибке")
            self.chk_save_df_on_err.setChecked(True)
            opts.addWidget(self.chk_save_df_on_err)
            opts.addStretch(1)
            layout.addLayout(opts)
            self.setLayout(layout)
            # connections
            self.btnAnalyze.clicked.connect(self.on_analyze)
            self.btnNews.clicked.connect(self.on_news)
            self.timer = QTimer(); self.timer.timeout.connect(self.on_analyze)
            self.auto.stateChanged.connect(self._toggle_timer)
            self.saveBtn.clicked.connect(self.save_logs)
            self.chk_use_pandas_ta.stateChanged.connect(self._toggle_pandas_ta)
        def _toggle_pandas_ta(self):
            global USE_PANDAS_TA_ALL
            USE_PANDAS_TA_ALL = bool(self.chk_use_pandas_ta.isChecked())
            self.out.append(f"USE_PANDAS_TA_ALL = {USE_PANDAS_TA_ALL}")
        def _toggle_timer(self):
            if self.auto.isChecked():
                self.timer.start(self.intSpin.value() * 1000)
                self.out.append("Автоматический режим включён")
            else:
                self.timer.stop()
                self.out.append("Автоматический режим выключен")
        def _setLampColor(self, name: str):
            m = {'green':'#22c55e','red':'#ef4444','yellow':'#facc15','gray':'#6b7280'}
            col = m.get(name, '#6b7280')
            self.lamp.setStyleSheet(f"#SignalLamp {{ background: {col}; }}")
        def save_logs(self):
            if not os.path.exists(LOG_CSV):
                self.out.append("Нет логов для сохранения.")
                return
            fname, _ = QFileDialog.getSaveFileName(self, "Сохранить логи как…", LOG_CSV, "CSV Files (*.csv)")
            if fname:
                try:
                    shutil.copyfile(LOG_CSV, fname)
                    self.out.append(f"Логи сохранены в {fname}")
                except Exception as e:
                    self.out.append(f"Ошибка при сохранении: {e}")
        def on_news(self):
            txt = self.newsEdit.text().strip()
            if txt:
                NEWS.ingest_text(txt)
                self.out.append(f"Новости учтены. sentiment={NEWS.get_sentiment():+.2f}")
                self.newsEdit.clear()
        def on_analyze(self):
            self.progress.setValue(5)
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            user_symbol = self.symbolBox.currentText().strip()
            if not user_symbol:
                self.out.append("Введите символ.")
                return
            # try resolve symbol
            try:
                resolved = find_symbol(user_symbol)
                if resolved and resolved != user_symbol:
                    idx = self.symbolBox.findText(resolved)
                    if idx >= 0:
                        self.symbolBox.setCurrentIndex(idx)
                    else:
                        self.symbolBox.insertItem(0, resolved)
                        self.symbolBox.setCurrentIndex(0)
                    symbol = resolved
                    self.out.append(f"Символ '{user_symbol}' автоматически разрешён как '{resolved}'")
                else:
                    symbol = user_symbol
            except Exception as e:
                debug_log(f"find_symbol error: {e}")
                symbol = user_symbol
            tf_val = tf_map[self.tfBox.currentText()]
            bars = int(self.barSpin.value())
            self.progress.setValue(10)
            self.out.append(f"Запуск анализа: {symbol} TF={self.tfBox.currentText()} bars={bars}")
            signals = WorkerSignals()
            signals.finished.connect(self._on_worker_finished)
            signals.error.connect(self._on_worker_error)
            worker = AnalyzerWorker(symbol, tf_val, bars, signals)
            worker.start()
            self.progress.setValue(20)
        def _on_worker_finished(self, res: Result):
            try:
                self.progress.setValue(85)
                lines = []
                lines.append(f"Символ: {res.symbol}   ТФ: {self.tfBox.currentText()}")
                lines.append(f"Цена: {res.price:.4f}")
                lines.append(f"ИИ вероятности → BUY {res.model_preds.get('BUY',0):.3f}  HOLD {res.model_preds.get('HOLD',0):.3f}  SELL {res.model_preds.get('SELL',0):.3f}")
                lines.append(f"Финальный сигнал: {res.agg_signal} | Доверие: {res.agg_confidence:.1f}%")
                if res.sl is not None:
                    lines.append(f"Вход: ~{res.price:.4f}")
                    lines.append(f"SL: {res.sl:.4f}")
                    lines.append(f"TP1: {res.tp1:.4f} (ATR x{TP1_ATR})")
                    lines.append(f"TP2: {res.tp2:.4f} (ATR x{TP2_ATR})")
                    lines.append(f"TP3: {res.tp3:.4f} (ATR x{TP3_ATR})")
                    lines.append(f"Реком. лот: {res.lots}")
                lines.append("\nМетрики модели:")
                for k, v in res.ml_metrics.items():
                    lines.append(f"  {k}: {v}")
                lines.append("\nОТКАЗ ОТ ОТВЕТСТВЕННОСТИ: НЕТ ГАРАНТИЙ. ТЕСТ НА ДЕМО.")
                self.out.setPlainText("\n".join(lines))
                self.reasonsList.clear()
                for r in res.reasons[:60]:
                    QListWidgetItem(str(r), self.reasonsList)
                if res.agg_signal == 'ПОКУПАТЬ':
                    self._setLampColor('green')
                elif res.agg_signal == 'ПРОДАВАТЬ':
                    self._setLampColor('red')
                else:
                    self._setLampColor('yellow')
                self.statusL.setText(f"Сигнал: {res.agg_signal} ({res.agg_confidence:.1f}%)")
                try:
                    log_signal(res)
                except Exception as e:
                    debug_log(f"log_signal error: {e}")
                self.progress.setValue(100)
            except Exception as e:
                debug_log(f"_on_worker_finished exception: {e}\n{traceback.format_exc()}")
        def _on_worker_error(self, err_tuple):
            e, tb = err_tuple
            self.out.append(f"[Ошибка] {e}")
            self.out.append(tb)
            debug_log(f"Worker error: {e}\n{tb}")
            msg = str(e)
            if "Terminal: Call failed" in msg or "Не удалось загрузить данные" in msg:
                try:
                    self.out.append("Пытаюсь переинициализировать MT5...")
                    mt5_init_or_raise(force=True)
                    self.out.append("MT5 переинициализирован — попробуйте ещё раз.")
                except Exception as einit:
                    self.out.append(f"Ошибка переинициализации MT5: {einit}")
            if self.chk_save_df_on_err.isChecked():
                try:
                    symbol = self.symbolBox.currentText()
                    tf_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
                    tf_val = tf_map[self.tfBox.currentText()]
                    df = get_rates(symbol, tf_val, int(self.barSpin.value()))
                    if df is not None:
                        df.to_pickle(DF_DUMP_ON_ERR)
                        self.out.append(f"Дамп DF сохранён в {DF_DUMP_ON_ERR}")
                    else:
                        self.out.append("Не удалось получить DF для дампа.")
                except Exception as edump:
                    self.out.append(f"Ошибка дампа DF: {edump}")

# ============================
# Запуск
# ============================
def main():
    if _PYSIDE_AVAILABLE:
        app = QApplication(sys.argv)
        win = TraderProGUI()
        def _on_quit():
            try:
                if _MT5_AVAILABLE:
                    mt5.shutdown()
            except Exception:
                pass
        app.aboutToQuit.connect(_on_quit)
        win.show()
        sys.exit(app.exec())
    else:
        print("PySide6 не установлен — запускаем headless режим.")
        if not _MT5_AVAILABLE:
            print("MetaTrader5 не установлен. Выход.")
            return
        try:
            mt5_init_or_raise()
            print("MT5 инициализирован")
        except Exception as e:
            print("MT5 init error:", e)
            return
        try:
            sym = find_symbol(BASE_SYMBOL)
            print("Using symbol:", sym)
            res = analyze(sym, mt5.TIMEFRAME_M15, DEFAULT_BARS)
            print("Result:", res)
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()

if __name__ == "__main__":
    main()
