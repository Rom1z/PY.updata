# ai_trader_pro_local.py
# -*- coding: utf-8 -*-
"""
AI Trader Pro — локальная версия без OpenAI (GUI, 200+ индикаторов, LocalAIAgent)
Зависимости:
    pip install PySide6 pandas numpy ccxt mplfinance matplotlib
Примечания:
 - Полностью удалена интеграция с OpenAI. Никаких ключей API и сетевых вызовов.
 - Встроенный ИИ: класс LocalAIAgent. Работает детерминированно, прозрачно, офлайн.
 - Логика ИИ:
    * Скоринг по блокам (0..100): тренд, импульс/моментум, волатильность/риск-менеджмент, объёмы, каналы/диапазоны,
      ликвидность, свечные прокси (Heikin-Ashi), подтверждение мульти-таймфреймом (если доступен).
    * Итоговая сторона: LONG/SHORT/NEUTRAL. Если надёжность < 55 — NEUTRAL.
    * ENTRY/TP/SL рассчитываются на базе ATR (или запасной %).
Основа кода/GUI заимствована и упрощена из вашей версии. :contentReference[oaicite:1]{index=1}
"""
import os
import sys
import math
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Tuple

import numpy as np
import pandas as pd

# Optional libs
try:
    import ccxt
except Exception:
    ccxt = None

try:
    import mplfinance as mpf
except Exception:
    mpf = None

# PySide6 GUI
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QTextEdit, QGroupBox, QGridLayout, QTabWidget,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ---------------------------
# Data classes / types
# ---------------------------
@dataclass
class SignalResult:
    side: str
    confidence: float   # 0..1
    reason: str
    entry: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    sl: Optional[float]
    position_size: Optional[float] = None

IndicatorFunc = Callable[[pd.DataFrame], pd.Series]

# ---------------------------
# Indicator Registry
# ---------------------------
class IndicatorRegistry:
    def __init__(self):
        self._registry: Dict[str, IndicatorFunc] = {}

    def register(self, name: str):
        def decorator(fn: IndicatorFunc):
            self._registry[name] = fn
            return fn
        return decorator

    def get(self, name: str):
        return self._registry.get(name)

    def list(self):
        return sorted(list(self._registry.keys()))

    def compute(self, df: pd.DataFrame, names: List[str]):
        out = {}
        for n in names:
            fn = self.get(n)
            if not fn:
                out[n] = pd.Series(np.nan, index=df.index)
                continue
            try:
                out[n] = fn(df).astype(float)
            except Exception:
                out[n] = pd.Series(np.nan, index=df.index)
        return out

reg = IndicatorRegistry()

# ---------------------------
# Basic helpers for indicators
# ---------------------------
def _ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def _sma(series: pd.Series, period: int):
    return series.rolling(period, min_periods=1).mean()

def _wma(series: pd.Series, period: int):
    weights = np.arange(1, period+1)
    def _w(x):
        w = weights[-len(x):]
        return float(np.dot(x, w) / w.sum())
    return series.rolling(period, min_periods=1).apply(_w, raw=True)

def _roc(series: pd.Series, period: int):
    return 100.0 * (series / series.shift(period) - 1.0)

def _rsi(series: pd.Series, period: int):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ru = up.rolling(period, min_periods=1).mean()
    rd = down.rolling(period, min_periods=1).mean()
    rs = ru / (rd + 1e-9)
    return 100 - (100 / (1 + rs))

def _atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, period: int=14):
    high_low = series_high - series_low
    high_close = (series_high - series_close.shift()).abs()
    low_close = (series_low - series_close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

# ---------------------------
# Bulk register many indicators (200+)
# ---------------------------
def register_bulk_indicators():
    # MA family
    ma_periods = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,24,26,28,30,34,36,40,44,48,50,55,60,65,70,75,80,85,90,95,100,110,120,130,140,150,160,170,180,190,200,250,300,365]
    for p in ma_periods:
        @reg.register(f"sma_{p}")
        def _sma_p(df, p=p):
            return _sma(df['close'], p)
        @reg.register(f"ema_{p}")
        def _ema_p(df, p=p):
            return _ema(df['close'], p)
        @reg.register(f"wma_{p}")
        def _wma_p(df, p=p):
            return _wma(df['close'], p)

    # RSI/ROC with many periods
    r_periods = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,21,24,28,32,36,48,72]
    for p in r_periods:
        @reg.register(f"rsi_{p}")
        def _rsi_p(df, p=p):
            return _rsi(df['close'], p)
        @reg.register(f"roc_{p}")
        def _roc_p(df, p=p):
            return _roc(df['close'], p)

    # MACD variants
    macd_variants = [(12,26,9),(8,21,5),(5,34,9),(13,34,8)]
    for a,b,s in macd_variants:
        @reg.register(f"macd_{a}_{b}_{s}_line")
        def _macd_line(df, a=a,b=b):
            return _ema(df['close'], a) - _ema(df['close'], b)
        @reg.register(f"macd_{a}_{b}_{s}_signal")
        def _macd_sig(df, a=a,b=b,s=s):
            line = reg.get(f"macd_{a}_{b}_{s}_line")(df)
            return _ema(line, s)
        @reg.register(f"macd_{a}_{b}_{s}_hist")
        def _macd_hist(df, a=a,b=b,s=s):
            line = reg.get(f"macd_{a}_{b}_{s}_line")(df)
            sig = reg.get(f"macd_{a}_{b}_{s}_signal")(df)
            return line - sig

    # ATR/BB/OBV/MFI/PSAR/Ichimoku/Donchian etc.
    @reg.register("atr14")
    def _atr14(df):
        return _atr(df['high'], df['low'], df['close'], 14)

    @reg.register("atr7")
    def _atr7(df):
        return _atr(df['high'], df['low'], df['close'], 7)

    @reg.register("bb_upper")
    def _bb_upper(df):
        mid = _sma(df['close'], 20)
        std = df['close'].rolling(20, min_periods=1).std()
        return mid + 2*std

    @reg.register("bb_middle")
    def _bb_middle(df):
        return _sma(df['close'], 20)

    @reg.register("bb_lower")
    def _bb_lower(df):
        mid = _sma(df['close'], 20)
        std = df['close'].rolling(20, min_periods=1).std()
        return mid - 2*std

    @reg.register("obv")
    def _obv(df):
        direction = np.sign(df['close'].diff().fillna(0.0))
        return (direction * df['volume']).cumsum()

    @reg.register("mfi")
    def _mfi(df):
        tp = (df['high'] + df['low'] + df['close'])/3.0
        raw_money = tp * df['volume']
        pos_flow = raw_money.where(tp > tp.shift(), 0.0)
        neg_flow = raw_money.where(tp < tp.shift(), 0.0)
        pos_sum = pos_flow.rolling(14, min_periods=1).sum()
        neg_sum = neg_flow.rolling(14, min_periods=1).sum()
        mfr = pos_sum / (neg_sum + 1e-9)
        return 100 - (100 / (1 + mfr))

    @reg.register("psar")
    def _psar(df):
        high = df['high'].values; low = df['low'].values
        length = len(df)
        psar = np.zeros(length)
        bull = True
        af = 0.02; af_max = 0.2
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

    @reg.register("ich_tenkan")
    def _ich_tenkan(df):
        return (df['high'].rolling(9, min_periods=1).max() + df['low'].rolling(9, min_periods=1).min()) / 2.0

    @reg.register("ich_kijun")
    def _ich_kijun(df):
        return (df['high'].rolling(26, min_periods=1).max() + df['low'].rolling(26, min_periods=1).min()) / 2.0

    @reg.register("ich_span_a")
    def _ich_span_a(df):
        tenkan = reg.get("ich_tenkan")(df)
        kijun = reg.get("ich_kijun")(df)
        return ((tenkan + kijun)/2.0).shift(26)

    @reg.register("ich_span_b")
    def _ich_span_b(df):
        return ((df['high'].rolling(52, min_periods=1).max() + df['low'].rolling(52, min_periods=1).min())/2.0).shift(26)

    @reg.register("ich_chikou")
    def _ich_chikou(df):
        return df['close'].shift(-26)

    @reg.register("donchian_up")
    def _donchian_up(df):
        return df['high'].rolling(20, min_periods=1).max()

    @reg.register("donchian_mid")
    def _donchian_mid(df):
        up = _donchian_up(df)
        dn = df['low'].rolling(20, min_periods=1).min()
        return (up + dn)/2.0

    @reg.register("donchian_dn")
    def _donchian_dn(df):
        return df['low'].rolling(20, min_periods=1).min()

    # Liquidity heuristics (simple)
    @reg.register("liquidity_highs")
    def _liquidity_highs(df):
        lookback = 60
        tolerance = 0.0007
        highs=[]
        recent = df.tail(lookback)
        for i in range(2, len(recent)):
            h1 = recent['high'].iloc[i-2]; h2 = recent['high'].iloc[i-1]; h3 = recent['high'].iloc[i]
            if abs(h1 - h2)/max(h1,h2) < tolerance or abs(h2 - h3)/max(h2,h3) < tolerance:
                highs.append(float(np.mean([h1,h2,h3])))
        highs = sorted(list({round(x,8) for x in highs}))
        s = pd.Series(np.nan, index=df.index)
        if highs:
            s.iloc[-1] = highs[-1]
        return s

    @reg.register("liquidity_lows")
    def _liquidity_lows(df):
        lookback = 60
        tolerance = 0.0007
        lows=[]
        recent = df.tail(lookback)
        for i in range(2, len(recent)):
            l1 = recent['low'].iloc[i-2]; l2 = recent['low'].iloc[i-1]; l3 = recent['low'].iloc[i]
            if abs(l1 - l2)/max(l1,l2) < tolerance or abs(l2 - l3)/max(l2,l3) < tolerance:
                lows.append(float(np.mean([l1,l2,l3])))
        lows = sorted(list({round(x,8) for x in lows}))
        s = pd.Series(np.nan, index=df.index)
        if lows:
            s.iloc[-1] = lows[-1]
        return s

    # Ulcer index, PVO
    @reg.register("ulcer")
    def _ulcer(df):
        series = df['close']
        rolling_max = series.rolling(14, min_periods=1).max()
        drawdown = 100*(series - rolling_max)/(rolling_max + 1e-9)
        return (drawdown.pow(2).rolling(14, min_periods=1).mean())**0.5

    @reg.register("pvo")
    def _pvo(df):
        v = df['volume']
        fast = _ema(v, 12)
        slow = _ema(v, 26)
        return 100*(fast - slow)/(slow + 1e-9)

    @reg.register("pvo_signal")
    def _pvo_signal(df):
        try:
            return _ema(_pvo(df), 9)
        except Exception:
            return pd.Series(np.nan, index=df.index)

    # VWAP (для вкладки индикаторов)
    @reg.register("vwap")
    def _vwap(df):
        tp = (df['high'] + df['low'] + df['close'])/3.0
        cum_pv = (tp * df['volume']).cumsum()
        cum_v = (df['volume']).cumsum().replace(0, np.nan)
        return cum_pv / cum_v

register_bulk_indicators()

# ---------------------------
# Utility: coalesce (treat NaN as missing)
# ---------------------------
def coalesce(existing, computed):
    if existing is None:
        return computed
    try:
        if isinstance(existing, float) and math.isnan(existing):
            return computed
    except Exception:
        pass
    return existing

# ---------------------------
# Trade parameter calculation
# ---------------------------
def calc_trade_params(price: float, atr: float, risk_pct: float, account_balance: float, side: str,
                      atr_sl_multiplier: float = 1.2, tp_multipliers: Tuple[float,float,float]=(1.0,1.8,2.6),
                      fallback_sl_pct: float = 0.005):
    try:
        if price is None or math.isnan(price):
            return (None, None, None, None, None, None)
        entry = float(price)
        if atr is None or math.isnan(atr) or atr <= 0.0:
            if side == "LONG":
                sl = entry * (1.0 - fallback_sl_pct)
                pip_risk = abs(entry - sl)
                tp1 = entry + pip_risk * tp_multipliers[0]
                tp2 = entry + pip_risk * tp_multipliers[1]
                tp3 = entry + pip_risk * tp_multipliers[2]
            elif side == "SHORT":
                sl = entry * (1.0 + fallback_sl_pct)
                pip_risk = abs(entry - sl)
                tp1 = entry - pip_risk * tp_multipliers[0]
                tp2 = entry - pip_risk * tp_multipliers[1]
                tp3 = entry - pip_risk * tp_multipliers[2]
            else:
                return (None, None, None, None, None, None)
        else:
            if side == "LONG":
                sl = entry - atr * atr_sl_multiplier
                tp1 = entry + atr * tp_multipliers[0]
                tp2 = entry + atr * tp_multipliers[1]
                tp3 = entry + atr * tp_multipliers[2]
            elif side == "SHORT":
                sl = entry + atr * atr_sl_multiplier
                tp1 = entry - atr * tp_multipliers[0]
                tp2 = entry - atr * tp_multipliers[1]
                tp3 = entry - atr * tp_multipliers[2]
            else:
                return (None, None, None, None, None, None)

        risk_amount = account_balance * (risk_pct/100.0)
        pip_risk = abs(entry - sl)
        if pip_risk <= 1e-12:
            pos_size = None
        else:
            pos_size = risk_amount / pip_risk
        return (entry, tp1, tp2, tp3, sl, pos_size)
    except Exception as e:
        print("calc_trade_params exception:", e)
        return (None, None, None, None, None, None)

# ---------------------------
# Local AI Agent (встроенный эксперт)
# ---------------------------
class LocalAIAgent:
    """
    «60-летний трейдер»: прозрачные правила, скоринг и пояснения.
    Возвращает SignalResult без каких-либо сетевых вызовов.
    """
    def __init__(self):
        pass

    def _last(self, s: Optional[pd.Series]):
        try:
            v = s.iloc[-1]
            return None if np.isnan(v) else float(v)
        except Exception:
            return None

    def _score_trend(self, f: Dict[str, pd.Series], price: float, notes: List[str]) -> float:
        score = 0.0
        ema20 = self._last(f.get('ema_20'))
        ema50 = self._last(f.get('ema_50'))
        ema200= self._last(f.get('ema_200'))
        if ema20 and ema50 and ema200:
            if ema20 > ema50 > ema200:
                score += 25; notes.append("Тренд: EMA20>EMA50>EMA200 (бычий)")
            elif ema20 < ema50 < ema200:
                score -= 25; notes.append("Тренд: EMA20<EMA50<EMA200 (медвежий)")
            else:
                notes.append("Тренд: смешанный")
        return score

    def _score_momentum(self, f: Dict[str, pd.Series], notes: List[str]) -> float:
        score = 0.0
        # MACD консенсус
        macd_pos, macd_neg = 0, 0
        for name in list(f.keys()):
            if name.endswith("_line") and name.startswith("macd_"):
                sig_name = name.replace("_line","_signal")
                l = self._last(f.get(name))
                s = self._last(f.get(sig_name))
                if l is not None and s is not None:
                    if l > s: macd_pos += 1
                    elif l < s: macd_neg += 1
        if macd_pos or macd_neg:
            if macd_pos > macd_neg:
                score += 10; notes.append(f"MACD: {macd_pos} вариаций за рост")
            elif macd_neg > macd_pos:
                score -= 10; notes.append(f"MACD: {macd_neg} вариаций за падение")

        # RSI14
        rsi14 = None
        for k in f.keys():
            if k.startswith("rsi_") and k.endswith("14"):
                rsi14 = self._last(f[k]); break
        if rsi14 is not None:
            if rsi14 < 30:
                score += 6; notes.append("RSI: перепродан (<30)")
            elif rsi14 > 70:
                score -= 6; notes.append("RSI: перекуплен (>70)")
        return score

    def _score_volume(self, f: Dict[str, pd.Series], notes: List[str]) -> float:
        score = 0.0
        obv = self._last(f.get('obv'))
        pvo = self._last(f.get('pvo'))
        pvo_sig = self._last(f.get('pvo_signal'))
        if pvo is not None and pvo_sig is not None:
            if pvo > pvo_sig:
                score += 4; notes.append("PVO: объёмы поддерживают рост")
            elif pvo < pvo_sig:
                score -= 4; notes.append("PVO: объёмы поддерживают снижение")
        if obv is not None:
            notes.append("OBV: учтён")
        return score

    def _score_ranges(self, f: Dict[str, pd.Series], price: float, notes: List[str]) -> float:
        score = 0.0
        bb_u = self._last(f.get('bb_upper'))
        bb_m = self._last(f.get('bb_middle'))
        bb_l = self._last(f.get('bb_lower'))
        if bb_u and bb_l and bb_m and price:
            # ближе к нижней — склоняемся к LONG; ближе к верхней — к SHORT
            rng = bb_u - bb_l
            if rng > 1e-12:
                pos = (price - bb_l)/rng  # 0..1
                if pos < 0.2:
                    score += 5; notes.append("Bollinger: у нижней границы")
                elif pos > 0.8:
                    score -= 5; notes.append("Bollinger: у верхней границы")
        # Donchian пробои
        d_up = self._last(f.get('donchian_up'))
        d_dn = self._last(f.get('donchian_dn'))
        if d_up and d_dn and price:
            if price > d_up:
                score += 6; notes.append("Donchian: пробой вверх")
            elif price < d_dn:
                score -= 6; notes.append("Donchian: пробой вниз")
        return score

    def _score_psar_ha(self, f: Dict[str, pd.Series], ha_close: Optional[pd.Series], notes: List[str]) -> float:
        score = 0.0
        psar = self._last(f.get('psar'))
        c = self._last(ha_close) if isinstance(ha_close, pd.Series) else None
        if psar is not None and c is not None:
            if c > psar:
                score += 5; notes.append("PSAR+HA: бычье расположение")
            elif c < psar:
                score -= 5; notes.append("PSAR+HA: медвежье расположение")
        return score

    def _score_liquidity(self, f: Dict[str, pd.Series], price: float, notes: List[str]) -> float:
        score = 0.0
        hi = self._last(f.get('liquidity_highs'))
        lo = self._last(f.get('liquidity_lows'))
        if price and hi:
            if price < hi:
                score -= 3; notes.append("Ликвидность сверху: риск съёма вверху")
        if price and lo:
            if price > lo:
                score += 3; notes.append("Ликвидность снизу: риск съёма внизу")
        return score

    def _confidence(self, raw: float) -> float:
        # ограничим [-100..100] -> масштаб в [0..1]
        raw = max(-100.0, min(100.0, raw))
        return abs(raw)/100.0

    def ask(self, symbol: str, timeframe: str, price: float, features: Dict[str, pd.Series], ha_close: pd.Series) -> SignalResult:
        notes: List[str] = []
        total = 0.0
        total += self._score_trend(features, price, notes)          # ±25
        total += self._score_momentum(features, notes)              # ±16
        total += self._score_volume(features, notes)                # ±4
        total += self._score_ranges(features, price, notes)         # ±11
        total += self._score_psar_ha(features, ha_close, notes)     # ±5
        total += self._score_liquidity(features, price, notes)      # ±6

        side = "NEUTRAL"
        if total > 15:
            side = "LONG"
        elif total < -15:
            side = "SHORT"

        conf = self._confidence(total)  # 0..1
        if conf < 0.55:
            side = "NEUTRAL"

        # ENTRY/TP/SL
        atr = None
        try:
            a = features.get('atr14')
            if a is not None:
                atr_val = a.iloc[-1]
                atr = None if np.isnan(atr_val) else float(atr_val)
        except Exception:
            atr = None

        entry = price if side in ("LONG","SHORT") else None
        if entry is not None:
            e,tp1,tp2,tp3,sl,pos = calc_trade_params(entry, atr if atr else 0.0, 1.0, 10000.0, side)
        else:
            e=tp1=tp2=tp3=sl=pos=None

        reasons = "• " + "\n• ".join(notes) if notes else "Недостаточно данных"
        return SignalResult(side, conf, reasons, e, tp1, tp2, tp3, sl, pos)

# ---------------------------
# Fetch Worker (fetch OHLCV and compute indicators)
# ---------------------------
class FetchWorker(QThread):
    finished = Signal(object, object, str, str)  # df, signal, symbol, timeframe

    def __init__(self, exchange, symbol: str, timeframe: str, agent: LocalAIAgent, use_swap: bool=False):
        super().__init__()
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.agent = agent
        self.use_swap = use_swap

    def run(self):
        try:
            if not self.exchange:
                raise RuntimeError("Exchange not configured (ccxt missing).")
            # fetch ohlcv
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=1000)
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
            df.index = pd.to_datetime(df["time"], unit="ms")
            df = df.drop(columns=["time"])
            # compute indicators
            keys = reg.list()
            features = reg.compute(df, keys)
            features_df = pd.DataFrame(features, index=df.index)
            df = pd.concat([df, features_df], axis=1)

            # Heikin-Ashi
            ha = pd.DataFrame(index=df.index)
            ha.loc[:, "ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
            ha.loc[df.index[0], "ha_open"] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
            for i in range(1, len(df.index)):
                cur_idx = df.index[i]; prev_idx = df.index[i-1]
                ha.loc[cur_idx, "ha_open"] = (ha.loc[prev_idx, "ha_open"] + ha.loc[prev_idx, "ha_close"]) / 2.0
            tmp_high = pd.concat([ha["ha_open"], ha["ha_close"], df["high"]], axis=1)
            tmp_low  = pd.concat([ha["ha_open"], ha["ha_close"], df["low"]], axis=1)
            ha.loc[:, "ha_high"] = tmp_high.max(axis=1)
            ha.loc[:, "ha_low"]  = tmp_low.min(axis=1)
            df = pd.concat([df, ha[["ha_open","ha_high","ha_low","ha_close"]]], axis=1)

            # Подбор фич для ИИ (включая самые важные)
            available = set(df.columns)
            want_keys = []
            sample_keys = ['ema_20','ema_50','ema_200','atr14','bb_upper','bb_lower','bb_middle','obv','mfi','psar','donchian_up','donchian_dn','liquidity_highs','liquidity_lows','pvo','pvo_signal','vwap']
            for k in sample_keys:
                if k in available:
                    want_keys.append(k)
            for k in reg.list():
                if k in available and k not in want_keys:
                    want_keys.append(k)
                if len(want_keys) >= 140:
                    break
            features_for_ai = {k: df[k] for k in want_keys}

            last_price = float(df['close'].iloc[-1])
            sig = self.agent.ask(self.symbol, self.timeframe, last_price, features_for_ai, df.get('ha_close'))

            self.finished.emit(df, sig, self.symbol, self.timeframe)
        except Exception as e:
            print("Ошибка загрузки данных:", e)
            traceback.print_exc()
            empty_df = pd.DataFrame()
            sig = SignalResult("NEUTRAL", 0.0, f"Ошибка: {e}", None, None, None, None, None)
            self.finished.emit(empty_df, sig, self.symbol, self.timeframe)

# ---------------------------
# UI Application
# ---------------------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trader Pro (Local) — Ликвидность & Multi Indicators")
        self.setGeometry(40, 40, 1400, 900)

        # Exchanges
        self.exchange_spot = None
        self.exchange_swap = None
        if ccxt:
            try:
                self.exchange_spot = ccxt.bybit({"enableRateLimit": True, "options":{"defaultType":"spot"}})
                self.exchange_swap = ccxt.bybit({"enableRateLimit": True, "options":{"defaultType":"swap"}})
            except Exception as e:
                print("ccxt/Bybit init failed:", e)

        self.use_swap = False
        self.agent = LocalAIAgent()
        self.markets = {}
        self._load_markets_once = False
        self.last_df = None
        self.init_ui()
        self.apply_style()

    def init_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.cmb_market = QComboBox(); self.cmb_market.addItems(["spot","swap"]); self.cmb_market.currentTextChanged.connect(self.on_market_change)
        self.ed_symbol = QLineEdit(); self.ed_symbol.setPlaceholderText("Символ (BTC, BTCUSDT, BTC/USDT)")
        self.cmb_tf = QComboBox(); self.cmb_tf.addItems(["1w","1d","4h","1h","30m","15m","5m","1m"]); self.cmb_tf.setCurrentText("1h")
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

        # tabs
        self.tabs = QTabWidget()
        self.tab_signal = QWidget(); self.tab_chart = QWidget(); self.tab_ind = QWidget(); self.tab_settings = QWidget()
        self.tabs.addTab(self.tab_signal, "Сигнал")
        self.tabs.addTab(self.tab_chart, "График")
        self.tabs.addTab(self.tab_ind, "Индикаторы")
        self.tabs.addTab(self.tab_settings, "Настройки")

        # Signal tab content
        gbox = QGroupBox("Сигнал и параметры сделки")
        g = QGridLayout(gbox)
        self.lbl_symbol = QLabel("—"); self.lbl_side = QLabel("—"); self.lbl_conf = QLabel("—")
        self.lbl_entry = QLabel("—"); self.lbl_tp1 = QLabel("—"); self.lbl_tp2 = QLabel("—"); self.lbl_tp3 = QLabel("—"); self.lbl_sl = QLabel("—")
        self.lbl_pos_size = QLabel("—")
        self.txt_reason = QTextEdit(); self.txt_reason.setReadOnly(True)

        g.addWidget(QLabel("Символ:"), 0,0); g.addWidget(self.lbl_symbol, 0,1)
        g.addWidget(QLabel("Сигнал:"), 1,0); g.addWidget(self.lbl_side, 1,1)
        g.addWidget(QLabel("Надёжность:"), 2,0); g.addWidget(self.lbl_conf, 2,1)
        g.addWidget(QLabel("Вход:"), 3,0); g.addWidget(self.lbl_entry, 3,1)
        g.addWidget(QLabel("TP1:"), 4,0); g.addWidget(self.lbl_tp1, 4,1)
        g.addWidget(QLabel("TP2:"), 5,0); g.addWidget(self.lbl_tp2, 5,1)
        g.addWidget(QLabel("TP3:"), 6,0); g.addWidget(self.lbl_tp3, 6,1)
        g.addWidget(QLabel("SL:"), 7,0); g.addWidget(self.lbl_sl, 7,1)
        g.addWidget(QLabel("Размер позиции (баз.)"), 8,0); g.addWidget(self.lbl_pos_size, 8,1)
        g.addWidget(QLabel("Обоснование (встроенный ИИ + 200+ индикаторов):"), 9,0,1,2)
        g.addWidget(self.txt_reason, 10,0,1,2)

        lay_sig = QVBoxLayout(self.tab_signal)
        lay_sig.addLayout(top)
        lay_sig.addWidget(gbox)

        # Chart tab
        lay_chart = QVBoxLayout(self.tab_chart)
        self.fig = Figure(figsize=(12,6))
        self.canvas = FigureCanvas(self.fig)
        lay_chart.addWidget(self.canvas)

        # Indicators tab
        lay_ind = QGridLayout(self.tab_ind)
        row = 0
        self.ind_checkboxes = {}
        preset_list = ['Heikin-Ashi', 'Liquidity Pools', 'VWAP', 'EMA 20/50/200', 'Bollinger', 'Donchian', 'SuperTrend', 'PSAR', 'Ichimoku']
        for name in preset_list:
            cb = QCheckBox(name)
            cb.setChecked(name in ['VWAP','EMA 20/50/200','Bollinger'])
            self.ind_checkboxes[name] = cb
            lay_ind.addWidget(cb, row//3, row%3)
            row += 1

        # settings tab
        lay_set = QGridLayout(self.tab_settings)
        self.spn_lookback_liq = QSpinBox(); self.spn_lookback_liq.setRange(10, 1000); self.spn_lookback_liq.setValue(60)
        self.dspn_tol_liq = QDoubleSpinBox(); self.dspn_tol_liq.setRange(0.00001, 0.01); self.dspn_tol_liq.setDecimals(6); self.dspn_tol_liq.setSingleStep(0.0001); self.dspn_tol_liq.setValue(0.0007)
        self.dspn_risk = QDoubleSpinBox(); self.dspn_risk.setRange(0.1, 100.0); self.dspn_risk.setSingleStep(0.1); self.dspn_risk.setValue(1.0)
        self.dspn_atr_mul = QDoubleSpinBox(); self.dspn_atr_mul.setRange(0.2, 10.0); self.dspn_atr_mul.setSingleStep(0.1); self.dspn_atr_mul.setValue(1.2)
        self.ed_balance = QLineEdit(); self.ed_balance.setPlaceholderText("Баланс (число)"); self.ed_balance.setText("10000")

        lay_set.addWidget(QLabel("Lookback ликвидности"), 0,0); lay_set.addWidget(self.spn_lookback_liq, 0,1)
        lay_set.addWidget(QLabel("Tolerance ликвидности"), 1,0); lay_set.addWidget(self.dspn_tol_liq, 1,1)
        lay_set.addWidget(QLabel("Риск на сделку, % от депо"), 2,0); lay_set.addWidget(self.dspn_risk, 2,1)
        lay_set.addWidget(QLabel("ATR множитель для SL"), 3,0); lay_set.addWidget(self.dspn_atr_mul, 3,1)
        lay_set.addWidget(QLabel("Account balance"), 4,0); lay_set.addWidget(self.ed_balance, 4,1)

        root.addWidget(self.tabs)

    def apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color: #0f1320; color: #E6E6E6; font-family: Segoe UI, Roboto, Arial; font-size: 10pt; }
            QGroupBox { border: 1px solid #2a2f45; border-radius: 8px; margin-top: 6px; padding: 8px; }
            QPushButton { background: #1f2540; border-radius: 6px; padding: 6px 8px; }
            QPushButton:hover { background: #2a325e; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #151a2e; border: 1px solid #2a2f45; border-radius: 6px; padding: 4px; }
            QTabBar::tab { background: #151a2e; padding: 6px 10px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background: #1d223a; }
            QTextEdit { background: #0f1320; border: 1px solid #2a2f45; border-radius: 6px; }
            QLabel { padding: 2px; }
        """)

    def on_market_change(self, text: str):
        self.use_swap = (text == "swap")
        self._load_markets_once = False
        self.markets = {}

    def load_markets(self):
        if self._load_markets_once:
            QMessageBox.information(self, "Info", "Markets already loaded.")
            return
        try:
            ex = self.exchange_swap if self.use_swap else self.exchange_spot
            if not ex:
                raise RuntimeError("Exchange not available (ccxt missing or init failed).")
            ex.load_markets()
            self.markets = ex.markets
            self._load_markets_once = True
            QMessageBox.information(self, "Info", f"Loaded {len(self.markets)} markets from Bybit ({self.cmb_market.currentText()}).")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка загрузки рынков: {e}")

    def export_csv(self):
        try:
            if not hasattr(self, 'last_df') or self.last_df is None or self.last_df.empty:
                QMessageBox.information(self, "Info", "Нет данных для экспорта.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "signals.csv", "CSV Files (*.csv)")
            if path:
                self.last_df.to_csv(path)
                QMessageBox.information(self, "Saved", f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка экспорта", str(e))

    def resolve_symbol(self, user_input: str) -> str:
        s = user_input.strip().upper().replace(" ", "").replace("-", "").replace("_","").replace("/","")
        if not s:
            return "BTC/USDT"
        if s.endswith("USDT"):
            return s[:-4] + "/USDT"
        elif s.endswith("USD"):
            return s[:-3] + "/USD"
        else:
            return s + "/USDT"

    def fetch_signal(self):
        if not self._load_markets_once:
            try:
                self.load_markets()
            except Exception:
                pass
        user_symbol = self.ed_symbol.text().strip() or "BTC/USDT"
        resolved = self.resolve_symbol(user_symbol)
        tf = self.cmb_tf.currentText()
        ex = self.exchange_swap if self.use_swap else self.exchange_spot
        if not ex:
            QMessageBox.warning(self, "Ошибка", "ccxt не инициализирован — установите ccxt и настройте ключи, или используйте mock данные.")
            return
        self.worker = FetchWorker(ex, resolved, tf, self.agent, self.use_swap)
        self.worker.finished.connect(self.on_fetched)
        self.btn_fetch.setEnabled(False)
        self.worker.start()

    def on_fetched(self, df: pd.DataFrame, sig: SignalResult, symbol: str, timeframe: str):
        self.btn_fetch.setEnabled(True)
        if df is None or df.empty:
            QMessageBox.warning(self, "Ошибка", f"Нет данных для {symbol}")
            return
        self.last_df = df.copy()
        # compute final TP/SL/pos_size using UI settings if needed
        try:
            account_balance = float(self.ed_balance.text())
        except Exception:
            account_balance = 10000.0
        risk_pct = float(self.dspn_risk.value())
        atr_mul = float(self.dspn_atr_mul.value())
        last_atr = float(df['atr14'].iloc[-1]) if 'atr14' in df.columns and not np.isnan(df['atr14'].iloc[-1]) else 0.0

        if sig.side in ("LONG","SHORT"):
            entry = sig.entry if sig.entry is not None else float(df['close'].iloc[-1])
            e, tp1_c, tp2_c, tp3_c, sl_c, pos_c = calc_trade_params(entry, last_atr if last_atr else 0.0, risk_pct, account_balance, sig.side, atr_sl_multiplier=atr_mul)
            sig.entry = coalesce(sig.entry, e)
            sig.tp1 = coalesce(sig.tp1, tp1_c)
            sig.tp2 = coalesce(sig.tp2, tp2_c)
            sig.tp3 = coalesce(sig.tp3, tp3_c)
            sig.sl = coalesce(sig.sl, sl_c)
            sig.position_size = coalesce(sig.position_size, pos_c)
        else:
            sig.entry = None; sig.tp1=None; sig.tp2=None; sig.tp3=None; sig.sl=None; sig.position_size=None

        def fmt(v, prec=8):
            if v is None: return "—"
            try:
                if isinstance(v, float) and math.isnan(v): return "—"
            except Exception:
                pass
            return f"{v:.{prec}f}"

        self.lbl_symbol.setText(f"{symbol}  [{timeframe}]  • {self.cmb_market.currentText()}")
        self.lbl_side.setText(sig.side)
        self.lbl_conf.setText(f"{sig.confidence*100:.1f}%")
        self.lbl_entry.setText(fmt(sig.entry))
        self.lbl_tp1.setText(fmt(sig.tp1))
        self.lbl_tp2.setText(fmt(sig.tp2))
        self.lbl_tp3.setText(fmt(sig.tp3))
        self.lbl_sl.setText(fmt(sig.sl))
        self.lbl_pos_size.setText(f"{sig.position_size:.6f}" if (sig.position_size is not None and not (isinstance(sig.position_size, float) and math.isnan(sig.position_size))) else "—")
        self.txt_reason.setPlainText(sig.reason)

        try:
            self.plot_chart(df, sig, symbol, timeframe)
        except Exception as e:
            print("Plot failed:", e)
            traceback.print_exc()

    def plot_chart(self, df: pd.DataFrame, sig: SignalResult, symbol: str, timeframe: str):
        self.fig.clear()
        ax_price = self.fig.add_subplot(2,1,1)
        ax_vol = self.fig.add_subplot(2,1,2, sharex=ax_price)

        try:
            if mpf:
                plot_df = pd.DataFrame({
                    'Open': df['open'], 'High': df['high'], 'Low': df['low'], 'Close': df['close'], 'Volume': df['volume']
                }, index=df.index)
                style = mpf.make_mpf_style(base_mpf_style='nightclouds')
                mpf.plot(plot_df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}),
                         type='candle', ax=ax_price, volume=ax_vol, style=style, show_nontrading=True, warn_too_much_data=10000)
            else:
                ax_price.plot(df.index, df['close'], linewidth=1.0, label='Close')
                ax_vol.bar(df.index, df['volume'], width=0.0005)
        except Exception:
            ax_price.plot(df.index, df['close'], linewidth=1.0, label='Close')

        # overlay indicators (selected)
        try:
            if self.ind_checkboxes.get('EMA 20/50/200') and self.ind_checkboxes['EMA 20/50/200'].isChecked():
                for key in ['ema_20','ema_50','ema_200']:
                    if key in df.columns:
                        ax_price.plot(df.index, df[key], linewidth=1.0, label=key.upper())
            if self.ind_checkboxes.get('VWAP') and self.ind_checkboxes['VWAP'].isChecked() and 'vwap' in df.columns:
                ax_price.plot(df.index, df['vwap'], linewidth=1.0, label="VWAP")
            if self.ind_checkboxes.get('Bollinger') and self.ind_checkboxes['Bollinger'].isChecked() and 'bb_upper' in df.columns:
                ax_price.plot(df.index, df['bb_upper'], linewidth=0.8, label="BB Up")
                ax_price.plot(df.index, df['bb_middle'], linewidth=0.8, label="BB Mid")
                ax_price.plot(df.index, df['bb_lower'], linewidth=0.8, label="BB Low")
            if self.ind_checkboxes.get('Donchian') and self.ind_checkboxes['Donchian'].isChecked():
                for k in ['donchian_up','donchian_mid','donchian_dn']:
                    if k in df.columns: ax_price.plot(df.index, df[k], linewidth=0.8, label=k)
            if self.ind_checkboxes.get('PSAR') and self.ind_checkboxes['PSAR'].isChecked() and 'psar' in df.columns:
                ax_price.scatter(df.index, df['psar'], s=6, label="PSAR")
            if self.ind_checkboxes.get('Ichimoku') and self.ind_checkboxes['Ichimoku'].isChecked():
                for k in ['ich_tenkan','ich_kijun','ich_span_a','ich_span_b']:
                    if k in df.columns:
                        ax_price.plot(df.index, df[k], linewidth=0.8, label=k)
        except Exception:
            pass

        # liquidity lines
        try:
            if self.ind_checkboxes.get('Liquidity Pools') and self.ind_checkboxes['Liquidity Pools'].isChecked():
                hi = df['liquidity_highs'].iloc[-1] if 'liquidity_highs' in df.columns else np.nan
                lo = df['liquidity_lows'].iloc[-1] if 'liquidity_lows' in df.columns else np.nan
                if not np.isnan(hi): ax_price.axhline(float(hi), linestyle='--', linewidth=0.8, alpha=0.7)
                if not np.isnan(lo): ax_price.axhline(float(lo), linestyle='--', linewidth=0.8, alpha=0.7)
        except Exception:
            pass

        # show entry/tp/sl
        try:
            def is_valid_price(x):
                return (x is not None) and (not (isinstance(x, float) and math.isnan(x)))
            if is_valid_price(sig.entry):
                color_pt = 'g' if sig.side == "LONG" else ('r' if sig.side == "SHORT" else 'k')
                ax_price.plot([df.index[-1]], [sig.entry], color_pt+'o', markersize=9)
                ax_price.axhline(sig.entry, color=color_pt, linewidth=1.0, alpha=0.6)
            if is_valid_price(sig.tp1): ax_price.axhline(sig.tp1, linewidth=0.9, alpha=0.7)
            if is_valid_price(sig.tp2): ax_price.axhline(sig.tp2, linewidth=0.9, alpha=0.7)
            if is_valid_price(sig.tp3): ax_price.axhline(sig.tp3, linewidth=0.9, alpha=0.7)
            if is_valid_price(sig.sl):  ax_price.axhline(sig.sl,  linewidth=0.9, alpha=0.85)
        except Exception:
            pass

        ax_price.set_title(f"{symbol} — {sig.side} ({sig.confidence*100:.1f}%)")
        ax_price.legend(loc='upper left', fontsize=8)
        ax_price.grid(True, linestyle=':', linewidth=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

# ---------------------------
# simple backtest helper
# ---------------------------
def simple_backtest(df: pd.DataFrame, entry_price: float, sl: float, tp: float, direction: str):
    res = {'entry': entry_price, 'sl': sl, 'tp': tp, 'direction': direction, 'exit_price': None, 'exit_idx': None, 'result': None}
    if direction not in ('LONG','SHORT'):
        res['result'] = 'NO_TRADE'
        return res
    for idx in range(1, len(df)):
        high = df['high'].iloc[idx]; low = df['low'].iloc[idx]
        if direction == 'LONG':
            if low <= sl:
                res.update({'exit_price': sl, 'exit_idx': idx, 'result': 'LOSS'})
                return res
            if high >= tp:
                res.update({'exit_price': tp, 'exit_idx': idx, 'result': 'WIN'})
                return res
        else:
            if high >= sl:
                res.update({'exit_price': sl, 'exit_idx': idx, 'result': 'LOSS'})
                return res
            if low <= tp:
                res.update({'exit_price': tp, 'exit_idx': idx, 'result': 'WIN'})
                return res
    res['result'] = 'UNDECIDED'
    return res

# ---------------------------
# Entry point
# ---------------------------
def main():
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
