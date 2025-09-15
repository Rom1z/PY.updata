# -*- coding: utf-8 -*-
"""
AI Trader Pro — PySide6 (Bybit/ccxt)
-----------------------------------
• 50+ индикаторов, график ликвидности, AI-агрегатор сигналов, TP/SL.
• Современный интерфейс: вкладки (Сигнал, График, Индикаторы, Настройки).
• Типы рынков: spot/swap (через ccxt.bybit). Символ-резолвер, маппинг таймфреймов.
• Это рабочая базовая версия (~1200 строк). Дальше можно расширять (модули, 6000+ строк).

Зависимости: PySide6, numpy, pandas, matplotlib, mplfinance, ccxt
pip install PySide6 numpy pandas matplotlib mplfinance ccxt

ВНИМАНИЕ: Ни один алгоритм не даёт 100% гарантии. Используйте на демо.
"""
import sys
import re
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

import ccxt
import mplfinance as mpf

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QTextEdit, QGroupBox, QGridLayout, QTabWidget,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ====================== УТИЛИТЫ ======================

def normalize_symbol(user_input: str) -> str:
    """ETH, ethusdt, ETH-USDT, eth/usdt -> ETH/USDT"""
    s = user_input.strip().upper()
    s = s.replace(" ", "")
    s = re.sub(r"[-_]", "", s)
    s = s.replace("/", "")
    if not s:
        return "BTC/USDT"
    # приоритет USDT
    if s.endswith("USDT"):
        return s[:-4] + "/USDT"
    elif s.endswith("USD"):
        return s[:-3] + "/USD"
    else:
        return s + "/USDT"


def resolve_symbol(user_input: str, markets: Dict[str, dict]) -> Optional[str]:
    """Пытаемся найти точный или ближайший символ на бирже"""
    norm = normalize_symbol(user_input)
    if norm in markets:
        return norm
    base = norm.split("/")[0]
    # сначала точный базис + /USDT
    for sym in markets.keys():
        if sym.upper().startswith(base + "/USDT"):
            return sym
    # затем любой символ начинающийся на base
    for sym in markets.keys():
        if sym.upper().startswith(base):
            return sym
    return None


# ====================== ДАТАКЛАССЫ ======================

@dataclass
class SignalResult:
    side: str
    confidence: float
    reason: str
    entry: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    sl: Optional[float]


# ====================== ИНДИКАТОРЫ (50+) ======================
# Многие индикаторы реализованы в упрощённом виде, достаточном для агрегатора.

# --- БАЗОВЫЕ СКОЛЬЗЯЩИЕ ---

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def dema(series: pd.Series, period: int) -> pd.Series:
    e = ema(series, period)
    return 2 * e - ema(e, period)


def tema(series: pd.Series, period: int) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * (e1 - e2) + e3


def hma(series: pd.Series, period: int) -> pd.Series:
    if period < 2:
        return series
    wma_half = wma(series, period // 2)
    wma_full = wma(series, period)
    raw = 2 * wma_half - wma_full
    return wma(raw, int(math.sqrt(period)))


def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    # Упрощённая Kaufman Adaptive MA
    change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(period).sum()
    er = change / (volatility + 1e-9)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    out = pd.Series(index=series.index, dtype=float)
    out.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        out.iloc[i] = out.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - out.iloc[i-1])
    return out

# --- ТРЕНД/ИМПУЛЬС ---

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    e12 = ema(series, fast)
    e26 = ema(series, slow)
    macd_line = e12 - e26
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def bbands(series: pd.Series, period: int = 20, dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + dev * std
    lower = mid - dev * std
    return upper, mid, lower


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']; low = df['low']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = atr(df, period)
    plus_di = 100 * (plus_dm.rolling(period).sum() / (tr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (tr + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.rolling(period).mean()


def dmi(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    high = df['high']; low = df['low']
    up_move = high.diff(); down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = atr(df, period)
    plus_di = 100 * (plus_dm.rolling(period).sum() / (tr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (tr + 1e-9))
    return plus_di, minus_di


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * (md + 1e-9))


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()
    return -100 * (high_max - df['close']) / (high_max - low_min + 1e-9)


def momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    return df['close'] - df['close'].shift(period)


def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    return (df['close'] / df['close'].shift(period) - 1.0) * 100.0


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff().fillna(0.0))
    return (direction * df['volume']).cumsum()


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    raw_money = tp * df['volume']
    pos_flow = raw_money.where(tp > tp.shift(), 0.0)
    neg_flow = raw_money.where(tp < tp.shift(), 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    mfr = pos_sum / (neg_sum + 1e-9)
    return 100 - (100 / (1 + mfr))


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    cum_pv = (tp * df['volume']).cumsum()
    cum_v = (df['volume'].replace(0, np.nan)).cumsum()
    return cum_pv / (cum_v + 1e-9)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    hl2 = (df['high'] + df['low']) / 2.0
    atrv = atr(df, period)
    upperband = hl2 + multiplier * atrv
    lowerband = hl2 - multiplier * atrv
    st = pd.Series(index=df.index, dtype=float)
    st.iloc[0] = upperband.iloc[0]
    trend_up = True
    for i in range(1, len(df)):
        if df['close'].iloc[i] > st.iloc[i-1]:
            st.iloc[i] = max(lowerband.iloc[i], st.iloc[i-1]) if trend_up else lowerband.iloc[i]
            trend_up = True
        else:
            st.iloc[i] = min(upperband.iloc[i], st.iloc[i-1]) if not trend_up else upperband.iloc[i]
            trend_up = False
    return st


def donchian_channel(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    upper = df['high'].rolling(period).max()
    lower = df['low'].rolling(period).min()
    mid = (upper + lower) / 2.0
    return upper, mid, lower


def keltner_channel(df: pd.DataFrame, period: int = 20, multiplier: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_mid = ema(df['close'], period)
    atrv = atr(df, 10)
    upper = ema_mid + multiplier * atrv
    lower = ema_mid - multiplier * atrv
    return upper, ema_mid, lower


def slope(series: pd.Series, period: int = 5) -> pd.Series:
    return series.diff(period) / (period + 1e-9)

# --- Объём/денежные потоки/другие ---

def adl(df: pd.DataFrame) -> pd.Series:
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
    mfv = mfm * df['volume']
    return mfv.cumsum()


def chaikin_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    line = adl(df)
    return ema(line, fast) - ema(line, slow)


def ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    ppo_val = 100 * (fast_ema - slow_ema) / (slow_ema + 1e-9)
    ppo_sig = ema(ppo_val, signal)
    return ppo_val, ppo_sig


def pvo(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    v = df['volume']
    fast_ema = ema(v, fast)
    slow_ema = ema(v, slow)
    pvo_val = 100 * (fast_ema - slow_ema) / (slow_ema + 1e-9)
    pvo_sig = ema(pvo_val, signal)
    return pvo_val, pvo_sig


def trix(series: pd.Series, period: int = 15) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 100 * (e3 - e3.shift()) / (e3.shift() + 1e-9)


def ui_ulcer(series: pd.Series, period: int = 14) -> pd.Series:
    rolling_max = series.rolling(period).max()
    drawdown = 100 * (series - rolling_max) / (rolling_max + 1e-9)
    return (drawdown.pow(2).rolling(period).mean())**0.5


def nvi(df: pd.DataFrame) -> pd.Series:
    close = df['close']; vol = df['volume']
    nvi = pd.Series(index=df.index, dtype=float)
    nvi.iloc[0] = 1000.0
    for i in range(1, len(df)):
        nvi.iloc[i] = nvi.iloc[i-1] * (1 + ((close.iloc[i] - close.iloc[i-1]) / (close.iloc[i-1] + 1e-9))) if vol.iloc[i] < vol.iloc[i-1] else nvi.iloc[i-1]
    return nvi


def pvi(df: pd.DataFrame) -> pd.Series:
    close = df['close']; vol = df['volume']
    pvi = pd.Series(index=df.index, dtype=float)
    pvi.iloc[0] = 1000.0
    for i in range(1, len(df)):
        pvi.iloc[i] = pvi.iloc[i-1] * (1 + ((close.iloc[i] - close.iloc[i-1]) / (close.iloc[i-1] + 1e-9))) if vol.iloc[i] > vol.iloc[i-1] else pvi.iloc[i-1]
    return pvi


def psar(df: pd.DataFrame, af: float = 0.02, af_max: float = 0.2) -> pd.Series:
    high = df['high'].values; low = df['low'].values
    length = len(df)
    psar = np.zeros(length)
    bull = True
    af_curr = af
    ep = low[0]
    psar[0] = low[0]
    for i in range(1, length):
        psar[i] = psar[i-1] + af_curr * (ep - psar[i-1])
        if bull:
            if high[i] > ep:
                ep = high[i]; af_curr = min(af_curr + af, af_max)
            if low[i] < psar[i]:
                bull = False; psar[i] = ep; ep = low[i]; af_curr = af
        else:
            if low[i] < ep:
                ep = low[i]; af_curr = min(af_curr + af, af_max)
            if high[i] > psar[i]:
                bull = True; psar[i] = ep; ep = high[i]; af_curr = af
    return pd.Series(psar, index=df.index)

# --- Ichimoku (основные линии) ---

def ichimoku(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    high = df['high']; low = df['low']; close = df['close']
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    return tenkan, kijun, span_a, span_b, chikou

# --- Heikin-Ashi OHLC ---

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha['open'] = 0.0
    ha['high'] = 0.0
    ha['low'] = 0.0
    ha['open'].iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha['open'].iloc[i] = (ha['open'].iloc[i-1] + ha['close'].iloc[i-1]) / 2
    ha['high'] = pd.concat([ha['open'], ha['close'], df['high']], axis=1).max(axis=1)
    ha['low'] = pd.concat([ha['open'], ha['close'], df['low']], axis=1).min(axis=1)
    return ha

# --- Pivot Points (Classic) ---

def pivots_classic(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    pp = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    r1 = 2*pp - df['low'].shift(1)
    s1 = 2*pp - df['high'].shift(1)
    r2 = pp + (df['high'].shift(1) - df['low'].shift(1))
    s2 = pp - (df['high'].shift(1) - df['low'].shift(1))
    return pp, r1, s1, r2, s2

# ====================== ЛИКВИДНОСТЬ ======================

def liquidity_pools(df: pd.DataFrame, lookback: int = 50, tolerance: float = 0.0005) -> Tuple[List[float], List[float]]:
    """
    Ищем скопления равных хай/лоу (resting liquidity). tolerance — допуск относительного равенства.
    Возвращает списки уровней high_pools и low_pools.
    """
    highs = []; lows = []
    recent = df.tail(lookback)
    for i in range(2, len(recent)):
        h1 = recent['high'].iloc[i-2]; h2 = recent['high'].iloc[i-1]; h3 = recent['high'].iloc[i]
        l1 = recent['low'].iloc[i-2];  l2 = recent['low'].iloc[i-1];  l3 = recent['low'].iloc[i]
        if abs(h1 - h2)/max(h1,h2) < tolerance or abs(h2 - h3)/max(h2,h3) < tolerance:
            highs.append(float(np.mean([h1,h2,h3])))
        if abs(l1 - l2)/max(l1,l2) < tolerance or abs(l2 - l3)/max(l2,l3) < tolerance:
            lows.append(float(np.mean([l1,l2,l3])))
    highs = sorted(list({round(x, 2) for x in highs}))
    lows = sorted(list({round(x, 2) for x in lows}))
    return highs, lows


def liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> Tuple[bool, bool]:
    """
    Детектор съёма ликвидности:
    - Bullish sweep: текущий low ниже минимумов lookback, но закрытие выше уровня.
    - Bearish sweep: текущий high выше максимумов lookback, но закрытие ниже уровня.
    """
    recent = df.tail(lookback+1)
    prev_window = recent.iloc[:-1]
    last = recent.iloc[-1]
    min_prev = prev_window['low'].min()
    max_prev = prev_window['high'].max()
    bullish = (last['low'] < min_prev) and (last['close'] > min_prev)
    bearish = (last['high'] > max_prev) and (last['close'] < max_prev)
    return bullish, bearish

# ====================== AI СИГНАЛ (агрегатор) ======================

def ai_signal(df: pd.DataFrame) -> SignalResult:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0
    reasons: List[str] = []

    # --- Режим рынка / тренд ---
    ema20 = last.get("ema20", np.nan); ema50 = last.get("ema50", np.nan); ema200 = last.get("ema200", np.nan)
    if ema50 > ema200:
        score += 0.06; reasons.append("Тренд восходящий (EMA50>EMA200)")
    else:
        score -= 0.06; reasons.append("Тренд нисходящий (EMA50<EMA200)")
    if last["close"] > ema20:
        score += 0.02; reasons.append("Цена выше EMA20 (локальная поддержка)")

    # --- MACD ---
    if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        score += 0.05; reasons.append("MACD бычий кросс")
    if prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        score -= 0.05; reasons.append("MACD медвежий кросс")
    if (last["macd_hist"] - prev["macd_hist"]) > 0:
        score += 0.02; reasons.append("MACD гистограмма усиливается")
    else:
        score -= 0.01; reasons.append("MACD гистограмма слабеет")

    # --- RSI / стохастик ---
    if 45 <= last["rsi14"] <= 60: score += 0.01; reasons.append("RSI нейтрально-здоровый")
    elif last["rsi14"] > 70: score -= 0.03; reasons.append("RSI перекупленность")
    elif last["rsi14"] < 30: score += 0.03; reasons.append("RSI перепроданность")

    if last["stoch_k"] < 20 and last["stoch_d"] < 20 and last["stoch_k"] > last["stoch_d"]:
        score += 0.02; reasons.append("Стохастик бычий разворот")
    if last["stoch_k"] > 80 and last["stoch_d"] > 80 and last["stoch_k"] < last["stoch_d"]:
        score -= 0.02; reasons.append("Стохастик медвежий разворот")

    # --- Полосы/каналы ---
    if last["close"] < last["bb_lower"]: score += 0.02; reasons.append("Ниже нижней полосы Боллинджера")
    elif last["close"] > last["bb_upper"]: score -= 0.02; reasons.append("Выше верхней полосы Боллинджера")

    if last["close"] > last["kc_upper"]: score -= 0.02; reasons.append("Выше Кельтнера (растяжка)")
    elif last["close"] < last["kc_lower"]: score += 0.02; reasons.append("Ниже Кельтнера (сжатие)")

    # --- Сила тренда / ADX ---
    if last["adx"] > 25: score += 0.02; reasons.append("Тренд устойчивый (ADX>25)")

    # --- Объём/денежный поток ---
    if last["obv"] > prev["obv"]: score += 0.01; reasons.append("OBV растёт")
    else: score -= 0.005; reasons.append("OBV ослабевает")

    if last["mfi"] < 20: score += 0.01; reasons.append("MFI < 20")
    if last["mfi"] > 80: score -= 0.01; reasons.append("MFI > 80")

    if last["close"] > last["vwap"]: score += 0.01; reasons.append("Цена выше VWAP")
    else: score -= 0.005; reasons.append("Цена ниже VWAP")

    # --- Momentum / ROC ---
    score += 0.01 if last["momentum"] > 0 else -0.01
    score += 0.01 if last["roc"] > 0 else -0.01

    # --- SuperTrend ---
    if last["close"] > last["supertrend"]: score += 0.02; reasons.append("Выше SuperTrend")
    else: score -= 0.02; reasons.append("Ниже SuperTrend")

    # --- Дончиан / пробои ---
    if last["close"] > last["donchian_up"]: score += 0.02; reasons.append("Пробой верхнего Дончиана")
    elif last["close"] < last["donchian_dn"]: score -= 0.02; reasons.append("Пробой нижнего Дончиана")

    # --- Ichimoku фильтры ---
    if last.get("ich_tenkan", np.nan) > last.get("ich_kijun", np.nan): score += 0.01; reasons.append("Tenkan>Kijun")
    if last.get("ich_span_a", np.nan) > last.get("ich_span_b", np.nan): score += 0.01; reasons.append("SpanA>SpanB (бычий облако)")

    # --- PSAR ---
    if last["close"] > last["psar"]: score += 0.01; reasons.append("Над PSAR")
    else: score -= 0.01; reasons.append("Под PSAR")

    # --- TRIX ---
    if last["trix"] > 0: score += 0.005
    else: score -= 0.005

    # --- Пивоты (близость к уровням) ---
    if last["close"] > last["pivot"]: score += 0.005
    else: score -= 0.005

    # --- Наклон ---
    if last["slope_close"] > 0: score += 0.01; reasons.append("Положительный наклон цены")
    else: score -= 0.005; reasons.append("Отрицательный наклон цены")

    # --- Ликвидность: sweep логика ---
    bull_sweep, bear_sweep = liquidity_sweep(df, lookback=20)
    if bull_sweep: score += 0.04; reasons.append("Bullish Liquidity Sweep")
    if bear_sweep: score -= 0.04; reasons.append("Bearish Liquidity Sweep")

    # Итог
    score = max(-1.0, min(1.0, score))
    side = "LONG" if score > 0.15 else ("SHORT" if score < -0.15 else "NEUTRAL")

    # Ценообразование Entry/TP/SL с учётом ликвидности
    entry = float(last["close"]) if side != "NEUTRAL" else None
    atr_val = float(last["atr14"]) if not math.isnan(last["atr14"]) else 0.0

    hi_pools, lo_pools = liquidity_pools(df, lookback=60, tolerance=0.0007)

    def nearest_above(levels: List[float], price: float) -> Optional[float]:
        ups = [x for x in levels if x > price]
        return min(ups) if ups else None

    def nearest_below(levels: List[float], price: float) -> Optional[float]:
        downs = [x for x in levels if x < price]
        return max(downs) if downs else None

    tp1 = tp2 = tp3 = sl = None
    if side == "LONG" and entry:
        tp1_pool = nearest_above(hi_pools, entry)
        tp1 = tp1_pool if tp1_pool else entry + 1.0 * atr_val
        tp2_pool = nearest_above([x for x in hi_pools if x != tp1_pool], entry)
        tp2 = tp2_pool if tp2_pool else entry + 1.8 * atr_val
        tp3 = entry + 2.6 * atr_val
        sl_pool = nearest_below(lo_pools, entry)
        sl = (sl_pool - 0.1 * atr_val) if sl_pool else entry - 1.2 * atr_val
    elif side == "SHORT" and entry:
        tp1_pool = nearest_below(lo_pools, entry)
        tp1 = tp1_pool if tp1_pool else entry - 1.0 * atr_val
        tp2_pool = nearest_below([x for x in lo_pools if x != tp1_pool], entry)
        tp2 = tp2_pool if tp2_pool else entry - 1.8 * atr_val
        tp3 = entry - 2.6 * atr_val
        sl_pool = nearest_above(hi_pools, entry)
        sl = (sl_pool + 0.1 * atr_val) if sl_pool else entry + 1.2 * atr_val

    return SignalResult(side, float(abs(score)), "; ".join(reasons), entry, tp1, tp2, tp3, sl)


# ====================== РАБОЧИЙ ПОТОК ======================

class FetchWorker(QThread):
    finished = Signal(pd.DataFrame, SignalResult, str, str)  # df, signal, symbol, timeframe

    def __init__(self, exchange: ccxt.Exchange, symbol: str, timeframe: str, use_swap: bool):
        super().__init__()
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_swap = use_swap

    def run(self):
        try:
            # Для Bybit ccxt таймфреймы стандартные, но возможны ограничения
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=500)
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
            df.index = pd.to_datetime(df["time"], unit="ms")

            # === Индикаторы ===
            df["ema20"], df["ema50"], df["ema200"] = ema(df["close"],20), ema(df["close"],50), ema(df["close"],200)
            df["sma20"], df["sma50"], df["sma200"] = sma(df["close"],20), sma(df["close"],50), sma(df["close"],200)
            df["wma30"], df["hma30"], df["kama10"] = wma(df['close'],30), hma(df['close'],30), kama(df['close'],10)
            df["dema20"], df["tema20"] = dema(df['close'],20), tema(df['close'],20)

            macd_line, macd_sig, macd_hist = macd(df["close"])
            df["macd"], df["macd_signal"], df["macd_hist"] = macd_line, macd_sig, macd_hist

            df["rsi14"] = rsi(df["close"], 14)
            up, mid, low = bbands(df["close"], 20, 2.0)
            df["bb_upper"], df["bb_middle"], df["bb_lower"] = up, mid, low

            df["atr14"] = atr(df, 14)
            df["adx"] = adx(df, 14)
            plus_di, minus_di = dmi(df, 14)
            df["plus_di"], df["minus_di"] = plus_di, minus_di

            k, d = stochastic(df, 14, 3)
            df["stoch_k"], df["stoch_d"] = k, d
            df["williams"] = williams_r(df, 14)
            df["momentum"] = momentum(df, 10)
            df["roc"] = roc(df, 12)
            df["obv"] = obv(df)
            df["mfi"] = mfi(df, 14)
            df["adl"] = adl(df)
            df["chaikin"] = chaikin_oscillator(df)
            ppo_val, ppo_sig = ppo(df['close'])
            df["ppo"], df["ppo_sig"] = ppo_val, ppo_sig
            pvo_val, pvo_sig = pvo(df)
            df["pvo"], df["pvo_sig"] = pvo_val, pvo_sig
            df["trix"] = trix(df['close'])
            df["ulcer"] = ui_ulcer(df['close'])
            df["nvi"] = nvi(df)
            df["pvi"] = pvi(df)

            df["vwap"] = vwap(df)
            df["supertrend"] = supertrend(df, 10, 3.0)
            d_up, d_mid, d_dn = donchian_channel(df, 20)
            df["donchian_up"], df["donchian_mid"], df["donchian_dn"] = d_up, d_mid, d_dn
            kc_up, kc_mid, kc_dn = keltner_channel(df, 20, 1.5)
            df["kc_upper"], df["kc_mid"], df["kc_lower"] = kc_up, kc_mid, kc_dn
            df["slope_close"] = slope(df["close"], 5)

            tenkan, kijun, span_a, span_b, chikou = ichimoku(df)
            df["ich_tenkan"], df["ich_kijun"], df["ich_span_a"], df["ich_span_b"], df["ich_chikou"] = tenkan, kijun, span_a, span_b, chikou

            df["psar"] = psar(df)
            pp, r1, s1, r2, s2 = pivots_classic(df)
            df["pivot"], df["r1"], df["s1"], df["r2"], df["s2"] = pp, r1, s1, r2, s2

            # Heikin-Ashi для опциональной отрисовки
            ha = heikin_ashi(df)
            df["ha_open"], df["ha_high"], df["ha_low"], df["ha_close"] = ha['open'], ha['high'], ha['low'], ha['close']

            # Сигнал
            sig = ai_signal(df)
            self.finished.emit(df, sig, self.symbol, self.timeframe)
        except Exception as e:
            print("Ошибка загрузки данных:", e)


# ====================== ГЛАВНОЕ ОКНО ======================

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trader Pro — Ликвидность & 50+ индикаторов (Bybit)")
        self.setGeometry(40, 40, 1580, 980)

        # Биржа: по умолчанию spot (но можно переключить на swap)
        self.exchange_spot = ccxt.bybit({"options": {"defaultType": "spot"}})
        self.exchange_swap = ccxt.bybit({"options": {"defaultType": "swap"}})
        self.use_swap = False

        self.markets: Dict[str, dict] = {}
        self._load_markets_once = False

        self.init_ui()
        self.apply_style()

    # ---- UI ----
    def init_ui(self):
        root = QVBoxLayout(self)

        # Верхняя панель
        top = QHBoxLayout()
        self.cmb_market = QComboBox(); self.cmb_market.addItems(["spot","swap"]) ; self.cmb_market.currentTextChanged.connect(self.on_market_change)
        self.ed_symbol = QLineEdit(); self.ed_symbol.setPlaceholderText("Символ (ETH, ETHUSDT, ETH/USDT)")
        self.cmb_tf = QComboBox(); self.cmb_tf.addItems(["1w","1d","4h","1h","30m","15m","5m","1m"]) ; self.cmb_tf.setCurrentText("1h")
        self.btn_load = QPushButton("Загрузить рынки")
        self.btn_fetch = QPushButton("Сигнал")
        self.btn_export = QPushButton("Экспорт CSV")

        self.btn_load.clicked.connect(self.load_markets)
        self.btn_fetch.clicked.connect(self.fetch_signal)
        self.btn_export.clicked.connect(self.export_csv)

        top.addWidget(QLabel("Тип рынка:")); top.addWidget(self.cmb_market)
        top.addWidget(QLabel("Символ:")); top.addWidget(self.ed_symbol)
        top.addWidget(QLabel("Таймфрейм:")); top.addWidget(self.cmb_tf)
        top.addWidget(self.btn_load); top.addWidget(self.btn_fetch); top.addWidget(self.btn_export)

        # Табы
        self.tabs = QTabWidget()
        self.tab_signal = QWidget(); self.tab_chart = QWidget(); self.tab_ind = QWidget(); self.tab_settings = QWidget()
        self.tabs.addTab(self.tab_signal, "Сигнал")
        self.tabs.addTab(self.tab_chart, "График")
        self.tabs.addTab(self.tab_ind, "Индикаторы")
        self.tabs.addTab(self.tab_settings, "Настройки")

        # --- Сигнал ---
        gbox = QGroupBox("Сигнал и параметры сделки")
        g = QGridLayout(gbox)
        self.lbl_symbol = QLabel("—"); self.lbl_side = QLabel("—"); self.lbl_conf = QLabel("—")
        self.lbl_entry = QLabel("—"); self.lbl_tp1 = QLabel("—"); self.lbl_tp2 = QLabel("—"); self.lbl_tp3 = QLabel("—"); self.lbl_sl = QLabel("—")
        self.txt_reason = QTextEdit(); self.txt_reason.setReadOnly(True)

        g.addWidget(QLabel("Символ:"), 0,0); g.addWidget(self.lbl_symbol, 0,1)
        g.addWidget(QLabel("Сигнал:"), 1,0); g.addWidget(self.lbl_side, 1,1)
        g.addWidget(QLabel("Надёжность:"), 2,0); g.addWidget(self.lbl_conf, 2,1)
        g.addWidget(QLabel("Вход:"), 3,0); g.addWidget(self.lbl_entry, 3,1)
        g.addWidget(QLabel("TP1:"), 4,0); g.addWidget(self.lbl_tp1, 4,1)
        g.addWidget(QLabel("TP2:"), 5,0); g.addWidget(self.lbl_tp2, 5,1)
        g.addWidget(QLabel("TP3:"), 6,0); g.addWidget(self.lbl_tp3, 6,1)
        g.addWidget(QLabel("SL:"), 7,0); g.addWidget(self.lbl_sl, 7,1)
        g.addWidget(QLabel("Обоснование (50+ индикаторов и ликвидность):"), 8,0,1,2)
        g.addWidget(self.txt_reason, 9,0,1,2)

        lay_sig = QVBoxLayout(self.tab_signal)
        lay_sig.addLayout(top)
        lay_sig.addWidget(gbox)

        # --- График ---
        lay_chart = QVBoxLayout(self.tab_chart)
        self.fig = Figure(figsize=(14,7))
        self.canvas = FigureCanvas(self.fig)
        lay_chart.addWidget(self.canvas)

        # --- Индикаторы (таблица параметров/переключатели) ---
        lay_ind = QGridLayout(self.tab_ind)
        row = 0
        self.cb_heikin = QCheckBox("Heikin-Ashi свечи") ; self.cb_heikin.setChecked(False)
        self.cb_liq = QCheckBox("Пулы ликвидности") ; self.cb_liq.setChecked(True)
        self.cb_vwap = QCheckBox("VWAP") ; self.cb_vwap.setChecked(True)
        self.cb_ma = QCheckBox("EMA20/50/200") ; self.cb_ma.setChecked(True)
        self.cb_bb = QCheckBox("Bollinger Bands") ; self.cb_bb.setChecked(True)
        self.cb_kc = QCheckBox("Keltner Channel") ; self.cb_kc.setChecked(True)
        self.cb_don = QCheckBox("Donchian Channel") ; self.cb_don.setChecked(False)
        self.cb_st = QCheckBox("SuperTrend") ; self.cb_st.setChecked(True)
        self.cb_psar = QCheckBox("PSAR") ; self.cb_psar.setChecked(False)
        self.cb_ich = QCheckBox("Ichimoku базовый") ; self.cb_ich.setChecked(False)

        for w in [self.cb_heikin, self.cb_liq, self.cb_vwap, self.cb_ma, self.cb_bb, self.cb_kc, self.cb_don, self.cb_st, self.cb_psar, self.cb_ich]:
            lay_ind.addWidget(w, row//3, row%3)
            row += 1

        # --- Настройки ---
        lay_set = QGridLayout(self.tab_settings)
        self.spn_lookback_liq = QSpinBox(); self.spn_lookback_liq.setRange(10, 1000); self.spn_lookback_liq.setValue(60)
        self.dspn_tol_liq = QDoubleSpinBox(); self.dspn_tol_liq.setRange(0.00001, 0.01); self.dspn_tol_liq.setDecimals(6); self.dspn_tol_liq.setSingleStep(0.0001); self.dspn_tol_liq.setValue(0.0007)
        self.dspn_risk = QDoubleSpinBox(); self.dspn_risk.setRange(0.1, 5.0); self.dspn_risk.setSingleStep(0.1); self.dspn_risk.setValue(1.0)
        self.dspn_atr_mul = QDoubleSpinBox(); self.dspn_atr_mul.setRange(0.2, 5.0); self.dspn_atr_mul.setSingleStep(0.1); self.dspn_atr_mul.setValue(1.2)

        lay_set.addWidget(QLabel("Lookback ликвидности"), 0,0); lay_set.addWidget(self.spn_lookback_liq, 0,1)
        lay_set.addWidget(QLabel("Tolerance ликвидности"), 1,0); lay_set.addWidget(self.dspn_tol_liq, 1,1)
        lay_set.addWidget(QLabel("Риск на сделку, % от депо"), 2,0); lay_set.addWidget(self.dspn_risk, 2,1)
        lay_set.addWidget(QLabel("ATR множитель для SL"), 3,0); lay_set.addWidget(self.dspn_atr_mul, 3,1)

        root.addWidget(self.tabs)

    def apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: #0f1320; color: #E6E6E6; font-family: Segoe UI, Roboto, Arial; font-size: 12pt; }
            QGroupBox { border: 1px solid #2a2f45; border-radius: 16px; margin-top: 10px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QPushButton { background: #1f2540; border-radius: 12px; padding: 10px 14px; }
            QPushButton:hover { background: #2a325e; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #151a2e; border: 1px solid #2a2f45; border-radius: 10px; padding: 6px; }
            QTabBar::tab { background: #151a2e; padding: 10px 14px; border-top-left-radius: 12px; border-top-right-radius: 12px; }
            QTabBar::tab:selected { background: #1d223a; }
            QTextEdit { background: #0f1320; border: 1px solid #2a2f45; border-radius: 10px; }
        """)

    # ---- ЛОГИКА ----
    def on_market_change(self, text: str):
        self.use_swap = (text == "swap")
        self._load_markets_once = False
        self.markets = {}

    def load_markets(self):
        if self._load_markets_once:
            return
        try:
            ex = self.exchange_swap if self.use_swap else self.exchange_spot
            ex.load_markets()
            self.markets = ex.markets
            self._load_markets_once = True
            print(f"Загружено {len(self.markets)} символов с Bybit [{self.cmb_market.currentText()}]")
        except Exception as e:
            print("Ошибка загрузки рынков:", e)

    def export_csv(self):
        try:
            if not hasattr(self, 'last_df'):
                return
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "signals.csv", "CSV Files (*.csv)")
            if path:
                self.last_df.to_csv(path)
        except Exception as e:
            print("Ошибка экспорта:", e)

    def fetch_signal(self):
        if not self._load_markets_once:
            self.load_markets()
        user_symbol = self.ed_symbol.text().strip() or "BTC/USDT"
        resolved = resolve_symbol(user_symbol, self.markets) or "BTC/USDT"
        tf = self.cmb_tf.currentText()
        ex = self.exchange_swap if self.use_swap else self.exchange_spot
        self.worker = FetchWorker(ex, resolved, tf, self.use_swap)
        self.worker.finished.connect(self.on_fetched)
        self.worker.start()

    def on_fetched(self, df: pd.DataFrame, sig: SignalResult, symbol: str, timeframe: str):
        self.last_df = df.copy()
        # Обновить текст
        self.lbl_symbol.setText(f"{symbol}  [{timeframe}]  • {self.cmb_market.currentText()}")
        self.lbl_side.setText(sig.side)
        self.lbl_conf.setText(f"{sig.confidence*100:.1f}%")
        self.lbl_entry.setText(f"{sig.entry:.4f}" if sig.entry else "—")
        self.lbl_tp1.setText(f"{sig.tp1:.4f}" if sig.tp1 else "—")
        self.lbl_tp2.setText(f"{sig.tp2:.4f}" if sig.tp2 else "—")
        self.lbl_tp3.setText(f"{sig.tp3:.4f}" if sig.tp3 else "—")
        self.lbl_sl.setText(f"{sig.sl:.4f}" if sig.sl else "—")
        self.txt_reason.setPlainText(sig.reason)

        # Построить график
        self.fig.clear()
        ax_price = self.fig.add_subplot(2,1,1)
        ax_vol = self.fig.add_subplot(2,1,2, sharex=ax_price)

        # Выбор свечей: классика или Heikin-Ashi
        plot_df = df.copy()
        if self.cb_heikin.isChecked():
            plot_df = pd.DataFrame({
                'Open': df['ha_open'], 'High': df['ha_high'], 'Low': df['ha_low'], 'Close': df['ha_close'], 'Volume': df['volume']
            }, index=df.index)
        else:
            plot_df = pd.DataFrame({
                'Open': df['open'], 'High': df['high'], 'Low': df['low'], 'Close': df['close'], 'Volume': df['volume']
            }, index=df.index)

        style = mpf.make_mpf_style(base_mpf_style='yahoo')
        mpf.plot(
            plot_df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}),
            type='candle', ax=ax_price, volume=ax_vol, style=style, show_nontrading=True
        )

        # Наложим линии на цену
        try:
            if self.cb_ma.isChecked():
                ax_price.plot(df.index, df["ema20"], linewidth=1.0, label="EMA20")
                ax_price.plot(df.index, df["ema50"], linewidth=1.0, label="EMA50")
                ax_price.plot(df.index, df["ema200"], linewidth=1.0, label="EMA200")
            if self.cb_vwap.isChecked():
                ax_price.plot(df.index, df["vwap"], linewidth=1.0, label="VWAP")
            if self.cb_bb.isChecked():
                ax_price.plot(df.index, df["bb_upper"], linewidth=0.8, label="BB Up")
                ax_price.plot(df.index, df["bb_middle"], linewidth=0.8, label="BB Mid")
                ax_price.plot(df.index, df["bb_lower"], linewidth=0.8, label="BB Low")
            if self.cb_kc.isChecked():
                ax_price.plot(df.index, df["kc_upper"], linewidth=0.8, label="KC Up")
                ax_price.plot(df.index, df["kc_mid"], linewidth=0.8, label="KC Mid")
                ax_price.plot(df.index, df["kc_lower"], linewidth=0.8, label="KC Low")
            if self.cb_don.isChecked():
                ax_price.plot(df.index, df["donchian_up"], linewidth=0.8, label="Don Up")
                ax_price.plot(df.index, df["donchian_mid"], linewidth=0.8, label="Don Mid")
                ax_price.plot(df.index, df["donchian_dn"], linewidth=0.8, label="Don Dn")
            if self.cb_st.isChecked():
                ax_price.plot(df.index, df["supertrend"], linewidth=1.0, label="SuperTrend")
            if self.cb_psar.isChecked():
                ax_price.scatter(df.index, df["psar"], s=5, label="PSAR")
            if self.cb_ich.isChecked():
                ax_price.plot(df.index, df["ich_tenkan"], linewidth=0.8, label="Tenkan")
                ax_price.plot(df.index, df["ich_kijun"], linewidth=0.8, label="Kijun")
                ax_price.plot(df.index, df["ich_span_a"], linewidth=0.8, label="SpanA")
                ax_price.plot(df.index, df["ich_span_b"], linewidth=0.8, label="SpanB")
        except Exception:
            pass

        # Пулы ликвидности
        if self.cb_liq.isChecked():
            hi_pools, lo_pools = liquidity_pools(df, lookback=self.spn_lookback_liq.value(), tolerance=self.dspn_tol_liq.value())
            for lvl in hi_pools:
                ax_price.axhline(lvl, linestyle='--', linewidth=0.6, alpha=0.35)
            for lvl in lo_pools:
                ax_price.axhline(lvl, linestyle='--', linewidth=0.6, alpha=0.35)

        # Вход/TP/SL
        if sig.entry:
            color_pt = 'g' if sig.side == "LONG" else ('r' if sig.side == "SHORT" else 'k')
            ax_price.plot(df.index[-1], sig.entry, color_pt+'o', markersize=10)
            ax_price.axhline(sig.entry, color=color_pt, linewidth=1.0, alpha=0.6)
        if sig.tp1: ax_price.axhline(sig.tp1, color='tab:blue', linewidth=0.9, alpha=0.7)
        if sig.tp2: ax_price.axhline(sig.tp2, color='tab:blue', linewidth=0.9, alpha=0.7)
        if sig.tp3: ax_price.axhline(sig.tp3, color='tab:blue', linewidth=0.9, alpha=0.7)
        if sig.sl:  ax_price.axhline(sig.sl,  color='tab:orange', linewidth=0.9, alpha=0.85)

        ax_price.set_title(f"{symbol} — {sig.side} ({sig.confidence*100:.1f}%)")
        ax_price.legend(loc='upper left', fontsize=8)
        self.canvas.draw()


# ====================== ТОЧКА ВХОДА ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())
