# -*- coding: utf-8 -*-
"""
TraderPro AI — Русский интерфейс, встроенный ИИ без внешних LLM (правила + ML)

Исправления:
- Устойчивое mt5.initialize() / переинициализация
- Безопасный выбор символа
- get_rates с ретраями и деградацией размера запроса
- on_analyze пробует переинициализировать MT5 при Terminal: Call failed
- graceful mt5.shutdown() при выходе
"""

from __future__ import annotations
import sys, os, math, time, traceback, threading, queue, json, re
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
    QTextEdit, QComboBox, QSpinBox, QCheckBox, QProgressBar, QFileDialog,
    QLineEdit, QListWidget, QListWidgetItem
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor

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

RF_ESTIMATORS = 350
MLP_HIDDEN = (96, 48)
RANDOM_STATE = 42
MIN_TRAIN_SAMPLES = 160

ML_WEIGHT = 0.55
IND_WEIGHT = 0.35
NEWS_WEIGHT = 0.10

LOG_CSV = "signals_log.csv"

MAX_FEATURES = 800        # после загрузки 200+ индикаторов урезаем
VAR_THRESHOLD = 1e-9      # убираем константы
MAX_CORR = 0.98           # выкидываем сильно коррелирующие

# Если у тебя нестандартный путь к terminal64.exe — укажи здесь:
MT5_TERMINAL_PATH = None  # r"C:\Program Files\MetaTrader 5\terminal64.exe"

# -------------------------
# Утилиты
# -------------------------

def to_series_safe(x, index, fill=0.0):
    try:
        if x is None:
            return pd.Series(fill, index=index, dtype='float64')
        if isinstance(x, pd.Series):
            return x.reindex(index).astype('float64').fillna(method='ffill').fillna(method='bfill').fillna(fill)
        if isinstance(x, (np.ndarray, list, tuple)):
            arr = np.asarray(x, dtype=float)
            if arr.shape[0] == len(index):
                return pd.Series(arr, index=index, dtype='float64').fillna(fill)
            if arr.shape[0] < len(index):
                pad = np.full(len(index) - arr.shape[0], fill)
                return pd.Series(np.concatenate([pad, arr]), index=index, dtype='float64')
            return pd.Series(arr[-len(index):], index=index, dtype='float64')
        return pd.Series(fill, index=index, dtype='float64')
    except Exception:
        return pd.Series(fill, index=index, dtype='float64')


def ensure_float_cols(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').ffill().bfill().fillna(0).astype('float64')


# -------------------------
# Собственные индикаторы
# -------------------------

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


# -------------------------
# MT5 helpers (исправленные/усиленные)
# -------------------------

def mt5_init_or_raise(login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None, force: bool = False):
    """
    Надёжная инициализация MT5:
    - пробуем initialize(), затем initialize(path) если задан MT5_TERMINAL_PATH
    - ждём готовности терминала
    - при наличии логина — выполняем mt5.login(...)
    """
    if force:
        try:
            mt5.shutdown()
        except Exception:
            pass

    ok = False
    try:
        ok = mt5.initialize() if MT5_TERMINAL_PATH is None else mt5.initialize(MT5_TERMINAL_PATH)
    except Exception:
        try:
            ok = mt5.initialize()
        except Exception:
            ok = False

    if not ok:
        # вторая попытка (без/с путём)
        try:
            if MT5_TERMINAL_PATH is not None:
                ok = mt5.initialize(MT5_TERMINAL_PATH)
            else:
                ok = mt5.initialize()
        except Exception:
            ok = False

    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()} (path={MT5_TERMINAL_PATH})")

    # Ждём, пока терминал/сессия не вернут данные
    for _ in range(50):
        ti = mt5.terminal_info()
        ai = mt5.account_info()
        if ti is not None and ai is not None:
            break
        time.sleep(0.1)

    # Если требуется логин (опционально)
    if login and password and server:
        if not mt5.login(login=login, password=password, server=server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    ai = mt5.account_info()
    if ai is None:
        raise RuntimeError(f"MT5 account_info() is None. Not connected? last_error={mt5.last_error()}")


def find_symbol(base: str = BASE_SYMBOL) -> str:
    syms = mt5.symbols_get()
    if not syms:
        raise RuntimeError(f"symbols_get() returned empty. last_error={mt5.last_error()}")
    names = [s.name for s in syms]
    if base in names:
        return base
    for n in names:
        if base in n:
            return n
    return names[0]


def list_all_symbols(maxn: int = 200) -> List[str]:
    syms = mt5.symbols_get()
    names = [s.name for s in syms] if syms else []
    names.sort()
    return names[:maxn]


def _ensure_symbol_selected(symbol: str, timeout_sec: float = 3.0):
    """Выбираем символ и ждём появления тика (значит, подписка на поток есть)."""
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
    # Если не появилось — продолжаем всё равно (бывают "молчащие" символы)


def get_rates(symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
    """
    Надёжная загрузка истории:
    - выбираем символ, ждём данных
    - несколько попыток с паузами
    - если не удаётся — уменьшаем bars (деградация)
    """
    _ensure_symbol_selected(symbol)

    plan = [bars, max(1000, bars // 2), max(500, bars // 4), 500, 300]
    errors = []

    for want in plan:
        for attempt in range(4):
            rates = None
            try:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(want))
            except Exception:
                rates = None
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                # переведём время и уберём timezone-aware (чтобы избежать проблем при индексации)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(None)
                df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].set_index('time')
                df = df.rename(columns={'tick_volume': 'volume'})
                df = df.apply(pd.to_numeric, errors='coerce').ffill().bfill().fillna(0.0)
                for c in df.columns:
                    try:
                        df[c] = df[c].astype('float64')
                    except Exception:
                        pass
                return df
            errors.append(mt5.last_error())
            time.sleep(0.25 + 0.25 * attempt)

    raise RuntimeError(f"copy_rates_from_pos failed for {symbol}, timeframe={timeframe}, bars~{bars}. Last errors: {errors[-3:] if errors else 'n/a'}")


# -------------------------
# Индикаторный движок: 200+ фич
# -------------------------

def add_indicators(df: pd.DataFrame, use_all_strategy: bool = True) -> pd.DataFrame:
    df = df.copy()
    ensure_float_cols(df, ['open', 'high', 'low', 'close', 'volume'])
    idx = df.index
    close = df['close']

    # Базовый набор (дублируем ключевые для устойчивости)
    df['ema5'] = to_series_safe(ta.ema(close, length=5), idx)
    df['ema8'] = to_series_safe(ta.ema(close, length=8), idx)
    df['ema13'] = to_series_safe(ta.ema(close, length=13), idx)
    df['ema21'] = to_series_safe(ta.ema(close, length=21), idx)
    df['ema34'] = to_series_safe(ta.ema(close, length=34), idx)
    df['ema50'] = to_series_safe(ta.ema(close, length=50), idx)
    df['ema100'] = to_series_safe(ta.ema(close, length=100), idx)
    df['ema200'] = to_series_safe(ta.ema(close, length=200), idx)

    # Базы осцилляторов/волатильности
    df['rsi14'] = to_series_safe(ta.rsi(close, length=14), idx)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame):
        df['macd'] = to_series_safe(macd.iloc[:, 0], idx)
        df['macd_sig'] = to_series_safe(macd.iloc[:, 1], idx)
        df['macd_hist'] = to_series_safe(macd.iloc[:, 2], idx)
    bb = ta.bbands(close, length=20, std=2.0)
    if isinstance(bb, pd.DataFrame):
        df['bb_lower'] = to_series_safe(bb.iloc[:, 0], idx)
        df['bb_mid'] = to_series_safe(bb.iloc[:, 1], idx)
        df['bb_upper'] = to_series_safe(bb.iloc[:, 2], idx)
    df['atr14'] = to_series_safe(ta.atr(df['high'], df['low'], close, length=ATR_PERIOD), idx)

    # Собственные: VWAP, Donchian, PSAR, ADL
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    vtp = tp * df['volume']
    c_vtp = vtp.cumsum()
    c_vol = df['volume'].cumsum().replace(0, np.nan)
    df['vwap'] = (c_vtp / c_vol).fillna(method='ffill').fillna(df['close']).astype('float64')
    df['donch_high'] = df['high'].rolling(20).max().bfill().astype('float64')
    df['donch_low'] = df['low'].rolling(20).min().bfill().astype('float64')
    try:
        df['adl'] = to_series_safe(ta.ad(df['high'], df['low'], df['close'], df['volume']), idx)
    except Exception:
        df['adl'] = to_series_safe(None, idx)
    try:
        df['psar'] = parabolic_sar(df['high'], df['low'])
    except Exception:
        df['psar'] = to_series_safe(None, idx)

    # Массовое добавление индикаторов pandas_ta (даёт 200+ столбцов)
    if use_all_strategy:
        try:
            strat = ta.Strategy(name="mega_all", ta=ta.AllStrategy)
            df.ta.strategy(strat)
        except Exception:
            pass

    # Производные признаки
    if 'bb_mid' in df.columns:
        df['price_vs_bbmid'] = ((df['close'] - df['bb_mid']) / (df['bb_mid'] + 1e-9)).astype('float64')

    # Чистка
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    # Срез по числу фич (ускорение)
    if len(df.columns) > MAX_FEATURES:
        base_cols = ['open','high','low','close','volume','rsi14','macd','macd_sig','macd_hist','atr14','vwap','ema5','ema8','ema13','ema21','ema34','ema50','ema100','ema200','psar']
        others = [c for c in df.columns if c not in base_cols]
        keep_others = others[:max(0, MAX_FEATURES - len(base_cols))]
        df = df[base_cols + keep_others]

    # Variance filter
    variances = df.var(axis=0, skipna=True)
    df = df.loc[:, variances > VAR_THRESHOLD]

    # Correlation filter (быстрое приближение)
    try:
        corr = df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > MAX_CORR)]
        df = df.drop(columns=to_drop, errors='ignore')
    except Exception:
        pass

    # Типы
    for c in df.columns:
        try:
            df[c] = df[c].astype('float64')
        except Exception:
            pass
    return df


# -------------------------
# Метки (3 класса) и обучение
# -------------------------

def make_labels(df: pd.DataFrame, lookahead: int = LOOKAHEAD, thr_up: float = THR_UP, thr_down: float = THR_DOWN) -> pd.Series:
    future = df['close'].shift(-lookahead)
    ret = (future - df['close']) / (df['close'] + 1e-12)
    labels = np.where(ret <= thr_down, 0, np.where(ret >= thr_up, 2, 1))  # 0=SELL,1=HOLD,2=BUY
    return pd.Series(labels, index=df.index, dtype='int64')


def train_models(feature_df: pd.DataFrame, labels: pd.Series):
    feats = [c for c in feature_df.columns if c not in ['open','high','low','close','volume']]
    feats = [f for f in feats if feature_df[f].isnull().sum() == 0]
    valid_idx = labels.dropna().index
    X = feature_df.loc[valid_idx, feats].values
    y = labels.loc[valid_idx].values
    if len(y) < MIN_TRAIN_SAMPLES:
        return None, None, feats, {'error': 'not enough samples', 'samples': int(len(y))}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y
    )
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
        fi = dict(zip(feats, rf.feature_importances_.round(5)))
        metrics['feature_importances'] = {k: v for k, v in sorted(fi.items(), key=lambda x: -x[1])[:20]}
    except Exception:
        metrics['feature_importances'] = {}
    return rf, mlp, feats, metrics


# -------------------------
# Expert60 — правила «как опытный трейдер»
# -------------------------

def expert60_votes(df: pd.DataFrame, idx: pd.Timestamp) -> Tuple[float, List[str]]:
    r = df.loc[idx]
    votes = 0.0
    reasons: List[str] = []

    # Тренд по EMA
    try:
        bull = r['ema8'] > r['ema21'] > r['ema50']
        bear = r['ema8'] < r['ema21'] < r['ema50']
        if bull:
            votes += 1.2; reasons.append("Тренд бычий по EMA (8>21>50)")
        elif bear:
            votes -= 1.2; reasons.append("Тренд медвежий по EMA (8<21<50)")
    except Exception:
        pass

    # Импульс MACD
    try:
        if r.get('macd_hist', 0) > 0:
            votes += 0.8; reasons.append("MACD гистограмма > 0")
        else:
            votes -= 0.8; reasons.append("MACD гистограмма < 0")
    except Exception:
        pass

    # RSI экстремумы
    try:
        if r['rsi14'] < 28:
            votes += 0.7; reasons.append("RSI перепродан")
        elif r['rsi14'] > 72:
            votes -= 0.7; reasons.append("RSI перекуплен")
    except Exception:
        pass

    # Положение цены vs VWAP
    try:
        if r['close'] > r['vwap']:
            votes += 0.6; reasons.append("Цена выше VWAP")
        else:
            votes -= 0.6; reasons.append("Цена ниже VWAP")
    except Exception:
        pass

    # Полосы Боллинджера
    try:
        if r['close'] < r.get('bb_lower', r['close'] - 1):
            votes += 0.5; reasons.append("Пробой нижней BB — возможный отскок")
        elif r['close'] > r.get('bb_upper', r['close'] + 1):
            votes -= 0.5; reasons.append("Пробой верхней BB — риск коррекции")
    except Exception:
        pass

    # PSAR
    try:
        if r['psar'] < r['close']:
            votes += 0.4; reasons.append("PSAR под ценой (бычий контекст)")
        else:
            votes -= 0.4; reasons.append("PSAR над ценой (медвежий контекст)")
    except Exception:
        pass

    # Волатильность ATR
    try:
        atr = float(r['atr14'])
        atr_ma = float(df['atr14'].rolling(50).mean().iloc[-1]) if 'atr14' in df.columns else atr
        if atr_ma > 0 and atr > 1.2 * atr_ma:
            reasons.append("Повышенная волатильность — осторожнее с размером позиции")
    except Exception:
        pass

    return votes, reasons


# -------------------------
# Новости/сентимент (локально, без API)
# -------------------------

class NewsWatcher:
    KEYWORDS_BULL = [
        'геополит', 'риск', 'инфляц', 'замедление доллара', 'data weak', 'rate cut', 'dovish',
    ]
    KEYWORDS_BEAR = [
        'повышение ставки', 'rate hike', 'strong dollar', 'hawkish', 'nonfarm strong', 'cpi beat',
    ]

    def __init__(self):
        self.queue = queue.Queue()
        self.last_sentiment = 0.0  # -1..+1

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


# -------------------------
# TP/SL и лот
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
    lots = 0.0 if stop_value_per_lot <= 0 else max(getattr(info, 'volume_min', 0.01) or 0.01, risk_amount / stop_value_per_lot)
    vol_step = getattr(info, 'volume_step', 0.01) or 0.01
    lots = math.floor(lots / vol_step) * vol_step
    lots = round(lots, 2)

    return float(sl), float(tp1), float(tp2), float(tp3), lots


# -------------------------
# Результат
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
    reasons: List[str]


# -------------------------
# Основной анализатор
# -------------------------

def analyze(symbol: str, timeframe_value: int, bars: int = BARS) -> Result:
    df = get_rates(symbol, timeframe_value, bars)
    info = mt5.symbol_info(symbol)
    df_ind = add_indicators(df, use_all_strategy=True)

    labels = make_labels(df_ind)
    rf, mlp, feats, ml_metrics = train_models(df_ind, labels)

    latest_idx = df_ind.index[-1]

    # ML вероятности
    probs = {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33}
    if rf is not None:
        try:
            X_latest = df_ind.loc[[latest_idx], [f for f in feats if f in df_ind.columns]].values
            proba_rf = rf.predict_proba(X_latest)[0]
            cls_map = {int(c): p for c, p in zip(rf.classes_, proba_rf)}
            probs['SELL'] = float(cls_map.get(0, probs['SELL']))
            probs['HOLD'] = float(cls_map.get(1, probs['HOLD']))
            probs['BUY']  = float(cls_map.get(2, probs['BUY']))
            if mlp is not None:
                try:
                    proba_mlp = mlp.predict_proba(X_latest)[0]
                    for i, c in enumerate(rf.classes_):
                        lab = 'SELL' if int(c)==0 else ('HOLD' if int(c)==1 else 'BUY')
                        probs[lab] = 0.5 * probs[lab] + 0.5 * float(proba_mlp[i])
                except Exception:
                    pass
        except Exception as e:
            print("[analyze] model predict error:", e)

    # Expert60
    expert_score, expert_reasons = expert60_votes(df_ind, latest_idx)
    ind_component = max(-1.0, min(1.0, expert_score / 6.0))

    # Новости
    news_component = max(-1.0, min(1.0, NEWS.get_sentiment()))

    # Смешиваем
    ml_score = probs['BUY'] - probs['SELL']           # -1..+1
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
    account = mt5.account_info()
    balance = float(getattr(account, 'balance', 1000.0))

    sl = tp1 = tp2 = tp3 = None; lots = None
    if agg_signal in ("ПОКУПАТЬ", "ПРОДАВАТЬ"):
        try:
            side = 'BUY' if agg_signal == 'ПОКУПАТЬ' else 'SELL'
            sl, tp1, tp2, tp3, lots = calc_sl_tp_and_lot(info, price, side, atr_val, balance)
        except Exception:
            traceback.print_exc()

    reasons = [
        f"ML: BUY={probs['BUY']:.3f} HOLD={probs['HOLD']:.3f} SELL={probs['SELL']:.3f}",
        f"Expert60 score={expert_score:.2f} (норм={ind_component:.2f})",
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


# -------------------------
# Логирование
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
# GUI — современный, со светофором
# -------------------------

APP_QSS = """
QWidget { font-family: Segoe UI, Roboto, sans-serif; font-size: 12.5pt; }
QTextEdit { background: #0f1115; color: #e8e8e8; border: 1px solid #23262d; border-radius: 10px; }
QComboBox, QSpinBox, QLineEdit { background: #10131a; color: #d9dee7; border: 1px solid #2a2f3a; border-radius: 8px; padding: 4px 8px; }
QPushButton { background: #1e293b; color: #e6edf3; border-radius: 10px; padding: 8px 14px; }
QPushButton:hover { background: #334155; }
QProgressBar { border: 1px solid #2a2f3a; border-radius: 8px; text-align: center; }
QProgressBar::chunk { background-color: #22c55e; }
#SignalLamp { border-radius: 12px; border: 2px solid #111; }
"""

class RussianGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TraderPro AI — XAUUSD/FX ИИ (русский интерфейс)")
        self.resize(1160, 820)
        self.setStyleSheet(APP_QSS)

        layout = QVBoxLayout()

        # Верхняя панель
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Символ:"))
        self.symbolBox = QComboBox()
        try:
            mt5_init_or_raise()
            symbols = list_all_symbols(500)
            if BASE_SYMBOL not in symbols:
                symbols.insert(0, BASE_SYMBOL)
            self.symbolBox.addItems(symbols if symbols else [BASE_SYMBOL])
            # стараемся выбрать XAUUSD или близкий
            for i in range(self.symbolBox.count()):
                if BASE_SYMBOL in self.symbolBox.itemText(i):
                    self.symbolBox.setCurrentIndex(i)
                    break
            self.status_init = "MT5 успешно инициализирован"
        except Exception as e:
            self.symbolBox.addItems([BASE_SYMBOL])
            self.status_init = f"MT5 инициализация: {e}"
        bar.addWidget(self.symbolBox)

        bar.addWidget(QLabel("ТФ:"))
        self.tfBox = QComboBox(); self.tfBox.addItems(['M1','M5','M15','M30','H1','H4','D1']); self.tfBox.setCurrentText(DEFAULT_TF)
        bar.addWidget(self.tfBox)

        bar.addWidget(QLabel("Свечей:"))
        self.barSpin = QSpinBox(); self.barSpin.setRange(400, 8000); self.barSpin.setValue(BARS)
        bar.addWidget(self.barSpin)

        self.btnAnalyze = QPushButton("АНАЛИЗ")
        bar.addWidget(self.btnAnalyze)

        self.auto = QCheckBox("Авто")
        bar.addWidget(self.auto)

        bar.addWidget(QLabel("Интервал (с):"))
        self.intSpin = QSpinBox(); self.intSpin.setRange(5, 3600); self.intSpin.setValue(60)
        bar.addWidget(self.intSpin)

        self.saveBtn = QPushButton("Сохранить логи…")
        bar.addWidget(self.saveBtn)

        # Новостной ввод (опционально вставлять текст заголовков)
        bar2 = QHBoxLayout()
        bar2.addWidget(QLabel("Новости/заголовок →"))
        self.newsEdit = QLineEdit(); self.newsEdit.setPlaceholderText("Вставьте текст новости (например: FOMC dovish/hawkish)…")
        bar2.addWidget(self.newsEdit)
        self.btnNews = QPushButton("Учитывать новость")
        bar2.addWidget(self.btnNews)

        # Светофор + статус
        bar3 = QHBoxLayout()
        self.lamp = QLabel(); self.lamp.setObjectName("SignalLamp"); self.lamp.setFixedSize(24, 24)
        self._setLampColor("gray")
        self.statusL = QLabel(self.status_init if hasattr(self, 'status_init') else "Готово")
        bar3.addWidget(QLabel("Сигнал:")); bar3.addWidget(self.lamp); bar3.addWidget(self.statusL)
        bar3.addStretch(1)

        layout.addLayout(bar)
        layout.addLayout(bar2)
        layout.addLayout(bar3)

        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Вывод
        self.out = QTextEdit(); self.out.setReadOnly(True)
        layout.addWidget(self.out)

        # Причины (список)
        layout.addWidget(QLabel("Обоснование сигнала:"))
        self.reasonsList = QListWidget()
        layout.addWidget(self.reasonsList)

        self.setLayout(layout)

        # Сигналы
        self.btnAnalyze.clicked.connect(self.on_analyze)
        self.btnNews.clicked.connect(self.on_news)
        self.timer = QTimer(); self.timer.timeout.connect(self.on_analyze)
        self.auto.stateChanged.connect(self._toggle_timer)
        self.saveBtn.clicked.connect(self.save_logs)

        # MT5 init feedback
        try:
            mt5_init_or_raise()
            self.statusL.setText("MT5 инициализирован")
        except Exception as e:
            self.statusL.setText(f"[Ошибка MT5] {e}")

    def _toggle_timer(self):
        if self.auto.isChecked():
            self.timer.start(self.intSpin.value() * 1000)
        else:
            self.timer.stop()

    def _setLampColor(self, name: str):
        m = {
            'green': '#22c55e',
            'red': '#ef4444',
            'yellow': '#facc15',
            'gray': '#6b7280'
        }
        col = m.get(name, '#6b7280')
        self.lamp.setStyleSheet(f"#SignalLamp {{ background: {col}; }}")

    def save_logs(self):
        if not os.path.exists(LOG_CSV):
            self.out.append("Нет логов для сохранения.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Сохранить логи как…", LOG_CSV, "CSV Files (*.csv)")
        if fname:
            try:
                import shutil
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
        try:
            self.progress.setValue(8)
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            symbol = self.symbolBox.currentText()
            tf_val = tf_map[self.tfBox.currentText()]
            bars = int(self.barSpin.value())

            self.progress.setValue(18)
            try:
                res = analyze(symbol, tf_val, bars)
            except RuntimeError as e:
                msg = str(e)
                # Если видим типичную ошибку, пробуем один раз переинициализировать MT5 и повторить
                if ("Terminal: Call failed" in msg) or ("copy_rates_from_pos failed" in msg) or ("account_info() is None" in msg) or ("symbols_get() returned empty" in msg):
                    self.statusL.setText("Переинициализация MT5…")
                    try:
                        mt5_init_or_raise(force=True)
                    except Exception as einit:
                        self.out.append(f"[Ошибка переинициализации MT5] {einit}")
                        self.progress.setValue(0)
                        return
                    # пробуем подобрать символ
                    try:
                        sym_fixed = find_symbol(symbol)
                        if sym_fixed != symbol:
                            idx = self.symbolBox.findText(sym_fixed)
                            if idx >= 0:
                                self.symbolBox.setCurrentIndex(idx)
                            symbol = sym_fixed
                    except Exception:
                        pass
                    # повторная попытка
                    res = analyze(symbol, tf_val, bars)
                else:
                    raise

            self.progress.setValue(85)

            # Текстовый отчёт
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

            # Список причин
            self.reasonsList.clear()
            for r in res.reasons[:40]:
                QListWidgetItem(str(r), self.reasonsList)

            # Светофор
            if res.agg_signal == 'ПОКУПАТЬ':
                self._setLampColor('green')
            elif res.agg_signal == 'ПРОДАВАТЬ':
                self._setLampColor('red')
            else:
                self._setLampColor('yellow')
            self.statusL.setText(f"Сигнал: {res.agg_signal} ({res.agg_confidence:.1f}%)")

            self.progress.setValue(100)

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

    def _on_quit():
        try:
            mt5.shutdown()
        except Exception:
            pass

    app.aboutToQuit.connect(_on_quit)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
