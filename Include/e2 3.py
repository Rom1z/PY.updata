# ai_trader_pro.py
# -*- coding: utf-8 -*-
"""
AI Trader Pro — полнофункциональная версия (GUI, 200+ индикаторов, AIAgent skeleton)
Важно:
 - OpenAI API key НЕ должен быть в коде. Установите переменную окружения OPENAI_API_KEY
   или введите ключ через UI (кнопка "Установить OpenAI ключ").
 - Установка зависимостей (пример):
    pip install PySide6 pandas numpy ccxt mplfinance openai matplotlib
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
    QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox, QInputDialog
)
from PySide6.QtCore import Qt, QThread, Signal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Optional OpenAI
try:
    import openai
except Exception:
    openai = None

# ---------------------------
# Data classes / types
# ---------------------------
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
            # signal computed on the macd line by name
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
        # use functions registered above
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

    # Additional custom indicators for coverage
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

register_bulk_indicators()

# ---------------------------
# Utility: coalesce (treat NaN as missing)
# ---------------------------
def coalesce(existing, computed):
    """Возвращает existing если он не None и не NaN, иначе computed."""
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
    """
    Возвращает (entry, tp1, tp2, tp3, sl, position_size)
    position_size считаем в базовой валюте: risk_amount / abs(entry - sl)

    Поведение:
    - Если ATR > 0: SL = entry +/- atr * atr_sl_multiplier; TP = entry +/- atr * tp_multipliers[i]
    - Если ATR <= 0: используем процентный fallback: SL = entry*(1 - fallback_sl_pct) для LONG,
      SL = entry*(1 + fallback_sl_pct) для SHORT. TP рассчитываем через расстояние entry<->SL * tp_multipliers.
    """
    try:
        if price is None or math.isnan(price):
            return (None, None, None, None, None, None)
        entry = float(price)
        if atr is None or math.isnan(atr) or atr <= 0.0:
            # fallback на % если ATR отсутствует
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
# AI Agent
# ---------------------------
class AIAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", None)
        self.model = model
        if self.api_key and openai:
            openai.api_key = self.api_key

    def build_prompt(self, symbol: str, timeframe: str, price: float, feature_snippet: str, news_snippets: List[str]):
        prompt = f"""Вы — ветеран трейдинга с 60-летним опытом. Проанализируйте символ {symbol} на таймфрейме {timeframe}.
Текущая цена: {price:.8f}
Ключевые признаки (последние значения): {feature_snippet}
Последние новости (коротко): {" ||| ".join(news_snippets)}

Задачи:
1) Итоговая рекомендация: LONG / SHORT / NEUTRAL и надёжность (0-100). Если надёжность < 55 → NEUTRAL.
2) Укажите ENTRY (число) или '-' для NEUTRAL, TP1/TP2/TP3/SL (числа) если даёте сигнал.
3) Дайте 4-6 кратких причин (технические, ликвидность, новости, объём) — короткими пунктами.
4) Если даёте сигнал — кратко объясните расчёт SL/TP (например: SL = entry - 1.2*ATR(14)).

Формат ответа (строго):
RECOMMENDATION: <LONG|SHORT|NEUTRAL>
CONFIDENCE: <число 0-100>
ENTRY: <число или ->
TP1: <число или -> 
TP2: <число или -> 
TP3: <число или -> 
SL: <число или -> 
REASONS:
1) ...
2) ...
..."""
        return prompt

    def ask(self, symbol: str, timeframe: str, price: float, features: Dict[str, pd.Series], news: List[str]) -> SignalResult:
        # Compose feature snippet — show last values for up to N features
        keys = sorted(list(features.keys())) if features else []
        summary_items = []
        for k in keys[-60:]:
            try:
                v = features[k].iloc[-1]
                summary_items.append(f"{k}={float(v):.6f}" if not np.isnan(v) else f"{k}=nan")
            except Exception:
                summary_items.append(f"{k}=nan")
        feature_snippet = "; ".join(summary_items[:60])
        prompt = self.build_prompt(symbol, timeframe, price, feature_snippet, news[:6])

        # If API key available and openai installed -> call
        if self.api_key and openai:
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role":"system","content":"You are a helpful trading expert."},
                              {"role":"user","content":prompt}],
                    temperature=0.0,
                    max_tokens=512
                )
                text = resp['choices'][0]['message']['content'].strip()
                # pass features so parser can compute ATR fallback if needed
                return self.parse_llm_response(text, price, features)
            except Exception as e:
                print("LLM call failed:", e)
                return self.ai_fallback(features, price)
        else:
            # fallback local heuristic
            return self.ai_fallback(features, price)

    def parse_llm_response(self, text: str, price: float, features: Dict[str, pd.Series]=None) -> SignalResult:
        """
        Парсит строго отформатированный ответ LLM. Если TP/SL отсутствует и есть ATR -> дополняет.
        """
        side = "NEUTRAL"; confidence = 0.0; entry=None; tp1=None; tp2=None; tp3=None; sl=None
        lower = text.lower()
        # recommendation parsing
        try:
            import re
            mrec = re.search(r"recommendation:\s*(long|short|neutral)", text, flags=re.IGNORECASE)
            if mrec:
                rec = mrec.group(1).upper()
                if rec in ("LONG","SHORT","NEUTRAL"):
                    side = rec
            # confidence
            m = re.search(r"confidence:\s*([0-9]{1,3}(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
            if m:
                cval = float(m.group(1))
                # normalize to 0..1
                if cval > 1.5:  # likely percent
                    confidence = min(100.0, cval)/100.0
                else:
                    confidence = max(0.0, min(1.0, cval))
            # parse lines for explicit ENTRY/TP/SL
            for line in text.splitlines():
                l = line.strip()
                low = l.lower()
                if low.startswith("entry:"):
                    try: entry = float(l.split(":",1)[1].strip())
                    except: entry = None
                elif low.startswith("tp1:"):
                    try: tp1 = float(l.split(":",1)[1].strip())
                    except: tp1 = None
                elif low.startswith("tp2:"):
                    try: tp2 = float(l.split(":",1)[1].strip())
                    except: tp2 = None
                elif low.startswith("tp3:"):
                    try: tp3 = float(l.split(":",1)[1].strip())
                    except: tp3 = None
                elif low.startswith("sl:"):
                    try: sl = float(l.split(":",1)[1].strip())
                    except: sl = None
        except Exception:
            pass

        # apply neutral rule
        if confidence < 0.55:
            side = "NEUTRAL"

        # if side is not neutral and some TP/SL missing -> compute using ATR if available (or fallback % inside calc_trade_params)
        if side in ("LONG","SHORT"):
            last_atr = None
            try:
                if features and 'atr14' in features and not np.isnan(features['atr14'].iloc[-1]):
                    last_atr = float(features['atr14'].iloc[-1])
            except Exception:
                last_atr = None
            # set entry to current price if LLM didn't provide one
            if entry is None:
                entry = float(price)
            # compute TP/SL if necessary
            if (tp1 is None or tp2 is None or tp3 is None or sl is None):
                e,tp1_c,tp2_c,tp3_c,sl_c,pos = calc_trade_params(entry, last_atr if last_atr else 0.0, 1.0, 10000.0, side)
                entry = coalesce(entry, e)
                tp1 = coalesce(tp1, tp1_c)
                tp2 = coalesce(tp2, tp2_c)
                tp3 = coalesce(tp3, tp3_c)
                sl = coalesce(sl, sl_c)
        else:
            # neutral -> clear
            entry = None; tp1=None; tp2=None; tp3=None; sl=None

        reason = text
        return SignalResult(side, confidence, reason, entry, tp1, tp2, tp3, sl, None)

    def ai_fallback(self, features: Dict[str, pd.Series], price: float) -> SignalResult:
        # Simple heuristic-based fallback (keeps previous logic). Uses available EMA and MACD variants if present.
        if not features:
            last_vals = {}
        else:
            last_vals = {k:(features[k].iloc[-1] if k in features and len(features[k])>0 else float('nan')) for k in features}
        score = 0.0; reasons=[]
        try:
            # use best-guess EMA 50/200 if available
            ema50 = last_vals.get('ema_50', None)
            ema200 = last_vals.get('ema_200', None)
            if ema50 is not None and ema200 is not None and not np.isnan(ema50) and not np.isnan(ema200):
                if ema50 > ema200:
                    score += 0.06; reasons.append("Тренд восходящий (EMA50>EMA200)")
                else:
                    score -= 0.06; reasons.append("Тренд нисходящий (EMA50<EMA200)")

            # search for any macd variant where line > signal
            macd_pos = False; macd_neg = False
            for name in last_vals:
                if name.endswith("_line"):
                    sig_name = name.replace("_line","_signal")
                    if sig_name in last_vals and not np.isnan(last_vals[name]) and not np.isnan(last_vals[sig_name]):
                        if last_vals[name] > last_vals[sig_name]:
                            macd_pos = True
                        else:
                            macd_neg = True
            if macd_pos and not macd_neg:
                score += 0.03; reasons.append("MACD положителен (var)")
            elif macd_neg and not macd_pos:
                score -= 0.03; reasons.append("MACD отрицателен (var)")

            # RSI 14
            rsi14 = None
            for k in last_vals:
                if k.startswith("rsi_") and k.endswith("14"):
                    rsi14 = last_vals[k]; break
            if rsi14 is not None and not np.isnan(rsi14):
                if rsi14 < 30:
                    score += 0.03; reasons.append("RSI перепродан")
                elif rsi14 > 70:
                    score -= 0.03; reasons.append("RSI перекуплен")

            if 'obv' in last_vals and not np.isnan(last_vals.get('obv', np.nan)):
                score += 0.005; reasons.append("OBV поддерживает движение")
        except Exception:
            pass
        score = max(-1.0, min(1.0, score))
        side = "LONG" if score > 0.15 else ("SHORT" if score < -0.15 else "NEUTRAL")
        entry = float(price) if side != "NEUTRAL" else None
        atr = None
        try:
            if features and 'atr14' in features and not np.isnan(features['atr14'].iloc[-1]):
                atr = float(features['atr14'].iloc[-1])
        except Exception:
            atr = None
        tp1 = tp2 = tp3 = sl = None
        pos_size = None
        if entry is not None:
            entry, tp1, tp2, tp3, sl, pos_size = calc_trade_params(entry, atr if atr else 0.0, 1.0, 10000.0, side)
        reason = "; ".join(reasons) if reasons else "Недостаточно данных — fallback"
        confidence = abs(score)
        return SignalResult(side, confidence, reason, entry, tp1, tp2, tp3, sl, pos_size)

# ---------------------------
# Fetch Worker (fetch OHLCV and compute indicators)
# ---------------------------
class FetchWorker(QThread):
    finished = Signal(object, object, str, str)  # df, signal, symbol, timeframe

    def __init__(self, exchange, symbol: str, timeframe: str, agent: AIAgent, use_swap: bool=False):
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
            # concat
            df = pd.concat([df, features_df], axis=1)

            # heikin ashi
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

            # prepare features for AI: choose meaningful subset present in df
            available = set(df.columns)
            want_keys = []
            sample_keys = ['ema_50','ema_200','atr14','bb_upper','bb_lower','obv','mfi','psar','donchian_up','donchian_dn','liquidity_highs','liquidity_lows']
            for k in sample_keys:
                if k in available:
                    want_keys.append(k)
            # add up to ~120 most important from registry that exist
            for k in reg.list():
                if k in available and k not in want_keys:
                    want_keys.append(k)
                if len(want_keys) >= 120:
                    break
            features_for_ai = {k: df[k] for k in want_keys}

            # mock news — in real product, fetch from news APIs
            news = [
                "Mock news: large wallet moved funds to exchange (possible sell pressure).",
                "Mock news: positive developer update about protocol."
            ]

            last_price = float(df['close'].iloc[-1])
            sig = self.agent.ask(self.symbol, self.timeframe, last_price, features_for_ai, news)

            # emit
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
        self.setWindowTitle("AI Trader Pro — Ликвидность & Multi Indicators")
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
        self.agent = AIAgent(api_key=os.environ.get("OPENAI_API_KEY", None))
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
        self.btn_set_key = QPushButton("Установить OpenAI ключ")

        self.btn_load.clicked.connect(self.load_markets)
        self.btn_fetch.clicked.connect(self.fetch_signal)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_set_key.clicked.connect(self.set_openai_key)

        top.addWidget(QLabel("Тип рынка:")); top.addWidget(self.cmb_market)
        top.addWidget(QLabel("Символ:")); top.addWidget(self.ed_symbol)
        top.addWidget(QLabel("Таймфрейм:")); top.addWidget(self.cmb_tf)
        top.addWidget(self.btn_load); top.addWidget(self.btn_fetch); top.addWidget(self.btn_export); top.addWidget(self.btn_set_key)

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
        g.addWidget(QLabel("Обоснование (AI + 200+ индикаторов):"), 9,0,1,2)
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
        # Basic dark theme
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

    def set_openai_key(self):
        text, ok = QInputDialog.getText(self, "OpenAI Key", "Enter OpenAI API key (or leave empty to disable):")
        if ok:
            key = text.strip()
            if key:
                self.agent.api_key = key
                if openai:
                    openai.api_key = key
                QMessageBox.information(self, "OK", "API key set. (Key not saved in file.)")
            else:
                self.agent.api_key = None
                QMessageBox.information(self, "OK", "API key cleared.")

    def resolve_symbol(self, user_input: str) -> str:
        s = user_input.strip().upper().replace(" ", "").replace("-", "").replace("_","").replace("/","")
        if not s:
            return "BTC/USDT"
        if s.endswith("USDT"):
            return s[:-4] + "/USDT"
        elif s.endswith("USD"):
            return s[:-3] + "/USD"
        else:
            # fallback to /USDT
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
        last_atr = None
        if 'atr14' in df.columns and not np.isnan(df['atr14'].iloc[-1]):
            last_atr = float(df['atr14'].iloc[-1])
        else:
            last_atr = 0.0

        # if AI produced entry but no pos_size -> compute (or if AI didn't produce entry compute from last price)
        if sig.side in ("LONG","SHORT"):
            entry = sig.entry if sig.entry is not None else float(df['close'].iloc[-1])
            # compute TP/SL if missing, using last_atr (calc_trade_params now handles fallback percent)
            e, tp1_c, tp2_c, tp3_c, sl_c, pos_c = calc_trade_params(entry, last_atr if last_atr else 0.0, risk_pct, account_balance, sig.side, atr_sl_multiplier=atr_mul)
            sig.entry = coalesce(sig.entry, e)
            sig.tp1 = coalesce(sig.tp1, tp1_c)
            sig.tp2 = coalesce(sig.tp2, tp2_c)
            sig.tp3 = coalesce(sig.tp3, tp3_c)
            sig.sl = coalesce(sig.sl, sl_c)
            sig.position_size = coalesce(sig.position_size, pos_c)
        else:
            # neutral: clear entries
            sig.entry = None; sig.tp1=None; sig.tp2=None; sig.tp3=None; sig.sl=None; sig.position_size=None

        # update UI labels (handle None/NaN)
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

        # plot
        try:
            self.plot_chart(df, sig, symbol, timeframe)
        except Exception as e:
            print("Plot failed:", e)
            traceback.print_exc()

    def plot_chart(self, df: pd.DataFrame, sig: SignalResult, symbol: str, timeframe: str):
        self.fig.clear()
        ax_price = self.fig.add_subplot(2,1,1)
        ax_vol = self.fig.add_subplot(2,1,2, sharex=ax_price)

        # Candles: use mplfinance if available for nicer candles, else simple line
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

        # show entry/tp/sl (check for None/NaN explicitly)
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
    # iterate from second candle (assuming entry executed at index 0)
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
