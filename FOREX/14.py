#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TraderPro AI — Объединённый полный скрипт (исправлённый)
Версия: 2025-09-11 (merged + fixes)
Описание:
    - Универсальный движок сигналов для XAUUSD с расширенными expert60-правилами,
      ML-моделями (если установлены), FVG, Volume Profile, False Breakouts, DXY correlation,
      первом свечном правиле (NY 9:30) и уровнями предыдущего дня/сессии.
    - Поддержка MT5 при наличии модуля MetaTrader5; fallback на CSV или синтетику.
    - News: NewsAPI + RSS + простой лексиконный сентимент.
    - DRY_RUN=True по умолчанию.
Исправления:
    - Нормализация publishedAt дат в NewsFetcher.aggregate_news_sentiment (убраны tzinfo,
      чтобы избежать TypeError при сортировке).
    - freq='min' вместо 'T' в pd.date_range для синтетики.
    - .bfill() вместо fillna(method='bfill') в rolling средних.
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import csv
import logging
import traceback
import threading
import queue
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import feedparser
from dateutil import parser as dateparser
import pytz

# Optional heavy dependencies
try:
    import MetaTrader5 as mt5  # type: ignore
    MT5_AVAILABLE = True
except Exception:
    mt5 = None
    MT5_AVAILABLE = False

try:
    import pandas_ta as ta  # type: ignore
    PANDAS_TA_AVAILABLE = True
except Exception:
    ta = None
    PANDAS_TA_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.neural_network import MLPClassifier  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import accuracy_score, precision_score, recall_score  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# GUI optional
try:
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTextEdit, QComboBox, QSpinBox, QCheckBox, QProgressBar, QFileDialog,
        QLineEdit, QListWidget, QListWidgetItem, QMessageBox
    )
    from PySide6.QtCore import QTimer, Signal, QObject
    PYSIDE_AVAILABLE = True
except Exception:
    PYSIDE_AVAILABLE = False

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    "SYMBOL": "XAUUSD",
    "TIMEFRAME_MINUTES": 1,
    "HISTORY_BARS": 500,
    "DRY_RUN": True,
    "LOG_FILE": "trader_pro_ai_full.log",
    "SIGNAL_OUTPUT_CSV": "signals_history.csv",
    "MT5_LOGIN": None,
    "MT5_PASSWORD": None,
    "MT5_SERVER": None,
    "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY", ""),
    "RSS_SOURCES": [
        "https://www.reuters.com/markets/commodities/rss",
        "https://www.fxstreet.com/rss/news",
        "https://www.investing.com/rss/news_25.rss"
    ],
    # Heuristic weights (legacy)
    "WEIGHTS": {
        "experts_vote": 0.4,
        "market_intent": 0.35,
        "news_sentiment": 0.25
    },
    "MM": {
        "large_volume_multiplier": 5.0,
        "spike_window": 5
    },
    "RISK": {
        "risk_per_trade_usd": 100.0,
        "account_currency": "USD"
    },
    "POLL_SECONDS": 10,
    # ML / Expert60 params (from 11.txt)
    "DEFAULT_TF": "M15",
    "DEFAULT_BARS": 1500,
    "LOOKAHEAD": 6,
    "THR_UP": 0.0018,
    "THR_DOWN": -0.0018,
    "RISK_PER_TRADE_PCT": 0.01,
    "ATR_PERIOD": 14,
    "SL_ATR_MULT": 1.3,
    "TP1_ATR": 1.0,
    "TP2_ATR": 2.0,
    "TP3_ATR": 3.5,
    "RF_ESTIMATORS": 200,
    "MLP_HIDDEN": (96, 48),
    "RANDOM_STATE": 42,
    "MIN_TRAIN_SAMPLES": 160,
    "ML_WEIGHT": 0.55,
    "IND_WEIGHT": 0.35,
    "NEWS_WEIGHT": 0.10,
    "VOLUME_PROFILE_BINS": 50,
    "VALUE_AREA_SHARE": 0.70
}

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    filename=CONFIG["LOG_FILE"],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def debug_log(msg: str):
    logging.info(msg)

# -------------------------
# Utilities
# -------------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat()

def safe_div(a, b, default=0.0):
    try:
        return a / b
    except Exception:
        return default

def safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"

def ensure_float_cols(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').ffill().bfill().fillna(0).astype('float64')

# -------------------------
# Market Data Provider
# -------------------------
class MarketDataProvider:
    """
    Универсальный провайдер данных: MT5 если доступен, иначе CSV или синтетика.
    """
    def __init__(self, symbol: str, timeframe_minutes: int = 1, history_bars: int = 500):
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        self.history_bars = history_bars
        self.connected = False
        if MT5_AVAILABLE:
            logging.info("MetaTrader5 доступен.")
        else:
            logging.info("MetaTrader5 недоступен.")

    def connect_mt5(self, login=None, password=None, server=None) -> bool:
        if not MT5_AVAILABLE:
            logging.warning("MT5 не установлен. Пропускаю подключение.")
            self.connected = False
            return False
        try:
            mt5.initialize()
            if login and password and server:
                authorized = mt5.login(login, password=password, server=server)
                if not authorized:
                    logging.error("Не удалось авторизоваться в MT5: %s", mt5.last_error())
                    self.connected = mt5.account_info() is not None
                else:
                    self.connected = True
            else:
                self.connected = True
            logging.info("MT5 connected: %s", self.connected)
            return self.connected
        except Exception as e:
            logging.exception("Ошибка при подключении MT5: %s", e)
            self.connected = False
            return False

    def get_history_mt5(self) -> Optional[pd.DataFrame]:
        if not MT5_AVAILABLE or not self.connected:
            logging.warning("MT5 не доступен/не подключен.")
            return None
        timeframe = mt5.TIMEFRAME_M1
        utc_from = datetime.utcnow() - timedelta(minutes=self.history_bars * self.timeframe_minutes)
        rates = mt5.copy_rates_from(self.symbol, timeframe, utc_from, self.history_bars)
        if rates is None or len(rates) == 0:
            logging.warning("MT5 вернул 0 баров.")
            return None
        df = pd.DataFrame(rates)
        if 'time' in df.columns:
            try:
                df['time'] = pd.to_datetime(df['time'], unit='s')
            except Exception:
                pass
        df.set_index('time', inplace=True)
        return df

    def get_recent_ticks(self) -> Optional[pd.DataFrame]:
        df = self.get_history_mt5()
        if df is None:
            return None
        return df.tail(100)

    def get_history_from_csv(self, path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(path):
            logging.warning("CSV файл не найден: %s", path)
            return None
        df = pd.read_csv(path, parse_dates=['time'])
        df.set_index('time', inplace=True)
        return df

# -------------------------
# MarketMakerIntent
# -------------------------
class MarketMakerIntent:
    def __init__(self, config_mm: Dict[str, Any]):
        self.large_vol_mult = config_mm.get("large_volume_multiplier", 5.0)
        self.spike_window = config_mm.get("spike_window", 5)

    def analyze(self, df: pd.DataFrame) -> Dict[str, float]:
        res = {'mm': 0.33, 'inst': 0.33, 'retail': 0.34}
        if df is None or len(df) < 10:
            logging.info("Недостаточно данных для MarketMakerIntent, равновесие.")
            return res
        vol = df.get('tick_volume') if 'tick_volume' in df.columns else df.get('real_volume')
        if vol is None:
            vol = pd.Series(np.ones(len(df)), index=df.index)
        avg_vol = vol.rolling(20, min_periods=5).mean().bfill()
        last_vol = vol.iloc[-1]
        mean_recent = avg_vol.iloc[-1]
        vol_ratio = safe_div(last_vol, mean_recent, default=1.0)
        close = df['close']
        returns = close.pct_change().fillna(0)
        recent_returns = returns[-self.spike_window:]
        momentum = recent_returns.sum()
        hl_range = (df['high'] - df['low']).abs()
        avg_range = hl_range.rolling(20, min_periods=5).mean().iloc[-1]
        last_range = hl_range.iloc[-1]
        range_ratio = safe_div(last_range, avg_range, default=1.0)
        score_inst = score_mm = score_retail = 0.0
        if vol_ratio >= self.large_vol_mult and range_ratio > 1.5:
            score_inst += 0.7; score_mm += 0.2
        elif vol_ratio >= self.large_vol_mult and range_ratio <= 1.5:
            score_mm += 0.7; score_inst += 0.2
        else:
            score_retail += 0.4; score_mm += 0.3; score_inst += 0.3
        if momentum > 0.002 and vol_ratio > 2.0:
            score_inst += 0.2
        elif momentum < -0.002 and vol_ratio > 2.0:
            score_inst += 0.2
        bodies = (df['close'] - df['open']).abs()
        last_body = bodies.iloc[-1]
        last_wick = (df['high'] - df['low']).iloc[-1] - last_body
        if last_body > 0:
            wick_ratio = safe_div(last_wick, last_body, default=0.0)
            if wick_ratio > 2.5:
                score_retail += 0.3
        total = score_inst + score_mm + score_retail
        if total <= 0:
            return res
        res = {'inst': score_inst / total, 'mm': score_mm / total, 'retail': score_retail / total}
        logging.info("MarketMakerIntent: inst=%.2f mm=%.2f retail=%.2f", res['inst'], res['mm'], res['retail'])
        return res

# -------------------------
# NewsFetcher + Sentiment (исправлённый)
# -------------------------
class NewsFetcher:
    """
    Забирает новости из NewsAPI и RSS-фидов, делает простой сентимент-скор.
    """
    POSITIVE_WORDS = {"rise", "gain", "bull", "rally", "surge", "higher", "beats", "up"}
    NEGATIVE_WORDS = {"fall", "drop", "bear", "decline", "slump", "lower", "misses", "down", "risk"}
    NEUTRAL_WORDS = {"hold", "unchanged", "mixed"}

    def __init__(self, api_key: str = "", rss_sources: Optional[List[str]] = None):
        self.api_key = api_key
        self.rss_sources = rss_sources or []

    def fetch_newsapi(self, query: str = "gold OR xau OR XAUUSD", pagesize: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            logging.info("NewsAPI ключ не задан — пропускаю NewsAPI.")
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "pageSize": pagesize,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.api_key
        }
        try:
            r = requests.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            articles = data.get("articles", [])
            simplified = []
            for a in articles:
                simplified.append({
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "publishedAt": a.get("publishedAt"),
                    "source": a.get("source", {}).get("name")
                })
            logging.info("NewsAPI fetched %d articles", len(simplified))
            return simplified
        except Exception as e:
            logging.exception("Ошибка при запросе NewsAPI: %s", e)
            return []

    def fetch_rss(self, max_items_per_source=5) -> List[Dict[str, Any]]:
        items = []
        for src in self.rss_sources:
            try:
                feed = feedparser.parse(src)
                entries = feed.entries[:max_items_per_source]
                for e in entries:
                    published = None
                    if hasattr(e, 'published'):
                        published = e.published
                    elif hasattr(e, 'updated'):
                        published = e.updated
                    items.append({
                        "title": e.get("title"),
                        "description": e.get("summary", ""),
                        "publishedAt": published,
                        "source": feed.feed.get("title", src)
                    })
                logging.info("RSS %s: %d items", src, len(entries))
            except Exception:
                logging.exception("Ошибка при парсинге RSS: %s", src)
        return items

    def simple_sentiment(self, text: str) -> float:
        """
        Простой лексиконный сентимент: -1..+1
        """
        if not text or not isinstance(text, str):
            return 0.0
        t = text.lower()
        pos = sum(1 for w in self.POSITIVE_WORDS if w in t)
        neg = sum(1 for w in self.NEGATIVE_WORDS if w in t)
        score = safe_div(pos - neg, max(1, pos + neg))
        if 'risk' in t or 'uncertain' in t or 'volatility' in t:
            score -= 0.2
        return max(-1.0, min(1.0, score))

    def aggregate_news_sentiment(self) -> Dict[str, Any]:
        """
        Возвращает aggregate sentiment score и список новостей.
        """
        articles = []
        if self.api_key:
            articles.extend(self.fetch_newsapi())
        articles.extend(self.fetch_rss())

        # Нормализуем publishedAt → datetime без tzinfo (чтобы избежать смешения aware/naive)
        for a in articles:
            try:
                if a.get('publishedAt'):
                    dt = dateparser.parse(a['publishedAt'])
                    if dt is not None and dt.tzinfo is not None:
                        dt = dt.replace(tzinfo=None)
                    a['publishedAt_dt'] = dt
                else:
                    a['publishedAt_dt'] = None
            except Exception:
                a['publishedAt_dt'] = None

        # сортировка без ошибки (все даты naive или None)
        articles.sort(key=lambda x: x.get('publishedAt_dt') or datetime(1970, 1, 1), reverse=True)

        # Compute sentiment per article
        scores = []
        for a in articles:
            txt = (a.get('title') or "") + " " + (a.get('description') or "")
            s = self.simple_sentiment(txt)
            a['sentiment'] = s
            scores.append(s)

        agg = {
            "avg_sentiment": float(np.mean(scores)) if scores else 0.0,
            "article_count": len(articles),
            "articles": articles[:20]  # limit
        }
        logging.info("News aggregated: avg_sentiment=%.3f count=%d", agg['avg_sentiment'], agg['article_count'])
        return agg

# -------------------------
# NewsWatcher (simple ingest)
# -------------------------
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

NEWS_MANUAL = NewsWatcher()

# -------------------------
# Indicators & Feature Engineering
# -------------------------
def parabolic_sar(high: pd.Series, low: pd.Series, af0=0.02, af_step=0.02, af_max=0.2):
    n = len(high)
    if n == 0:
        return pd.Series([], dtype='float64')
    sar = np.zeros(n, dtype='float64')
    trend = 1
    af = af0
    ep = float(high.iloc[0]) if n>0 else 0.0
    sar[0] = float(low.iloc[0]) if n>0 else 0.0
    for i in range(1, n):
        prev = sar[i-1]
        if trend == 1:
            sar[i] = prev + af * (ep - prev)
            sar[i] = min(sar[i], float(low.iloc[i-1]), float(low.iloc[i-2]) if i>=2 else float(low.iloc[i-1]))
            if float(low.iloc[i]) < sar[i]:
                trend = -1
                sar[i] = ep
                ep = float(low.iloc[i])
                af = af0
            else:
                if float(high.iloc[i]) > ep:
                    ep = float(high.iloc[i])
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev + af * (ep - prev)
            sar[i] = max(sar[i], float(high.iloc[i-1]), float(high.iloc[i-2]) if i>=2 else float(high.iloc[i-1]))
            if float(high.iloc[i]) > sar[i]:
                trend = 1
                sar[i] = ep
                ep = float(high.iloc[i])
                af = af0
            else:
                if float(low.iloc[i]) < ep:
                    ep = float(low.iloc[i])
                    af = min(af + af_step, af_max)
    return pd.Series(sar, index=high.index, dtype='float64')

def add_indicators(df: pd.DataFrame, use_all_strategy: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    ensure_float_cols(df, ['open','high','low','close','volume'])
    close = df['close']
    try:
        df['ema5'] = close.ewm(span=5, adjust=False).mean()
        df['ema8'] = close.ewm(span=8, adjust=False).mean()
        df['ema13'] = close.ewm(span=13, adjust=False).mean()
        df['ema21'] = close.ewm(span=21, adjust=False).mean()
        df['ema50'] = close.ewm(span=50, adjust=False).mean()
        df['ema100'] = close.ewm(span=100, adjust=False).mean()
    except Exception as e:
        debug_log(f"EMA error: {e}")
    try:
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / (down + 1e-12)
        df['rsi14'] = (100 - (100 / (1 + rs))).fillna(50.0)
    except Exception as e:
        debug_log(f"RSI error: {e}")
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
    try:
        ma = close.rolling(20).mean()
        sd = close.rolling(20).std()
        df['bb_mid'] = ma
        df['bb_upper'] = ma + 2 * sd
        df['bb_lower'] = ma - 2 * sd
    except Exception as e:
        debug_log(f"BB error: {e}")
    try:
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(CONFIG.get("ATR_PERIOD",14)).mean().ffill().bfill().fillna(0.0)
    except Exception as e:
        debug_log(f"ATR error: {e}")
    try:
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        vtp = tp * df['volume']
        c_vtp = vtp.cumsum()
        c_vol = df['volume'].cumsum().replace(0, np.nan)
        df['vwap'] = (c_vtp / c_vol).ffill().bfill().fillna(df['close']).astype('float64')
    except Exception as e:
        debug_log(f"VWAP error: {e}")
        df['vwap'] = df['close']
    try:
        df['donch_high'] = df['high'].rolling(20).max().bfill().astype('float64')
        df['donch_low'] = df['low'].rolling(20).min().bfill().astype('float64')
    except Exception as e:
        debug_log(f"Donchian error: {e}")
    try:
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-12)
        df['adl'] = (clv * df['volume']).cumsum()
    except Exception as e:
        debug_log(f"ADL error: {e}")
    try:
        df['psar'] = parabolic_sar(df['high'], df['low'])
    except Exception as e:
        debug_log(f"PSAR error: {e}")
    try:
        if 'bb_mid' in df.columns:
            df['price_vs_bbmid'] = ((df['close'] - df['bb_mid']) / (df['bb_mid'] + 1e-9)).astype('float64')
    except Exception:
        pass
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    return df

# -------------------------
# ML helpers (labels/train/predict) - optional
# -------------------------
def make_labels(df: pd.DataFrame, lookahead: int = None, thr_up: float = None, thr_down: float = None) -> Optional[pd.Series]:
    lookahead = lookahead or CONFIG.get("LOOKAHEAD", 6)
    thr_up = thr_up or CONFIG.get("THR_UP", 0.0018)
    thr_down = thr_down or CONFIG.get("THR_DOWN", -0.0018)
    if df is None or 'close' not in df.columns:
        return None
    if len(df) < max(lookahead + 5, CONFIG.get("MIN_TRAIN_SAMPLES", 160)):
        logging.debug(f"make_labels: insufficient rows {len(df)}")
        return None
    future = df['close'].shift(-lookahead)
    ret = (future - df['close']) / (df['close'] + 1e-12)
    labels = np.where(ret <= thr_down, 0, np.where(ret >= thr_up, 2, 1))
    return pd.Series(labels, index=df.index, dtype='int64')

def train_models(feature_df: pd.DataFrame, labels: pd.Series):
    if not SKLEARN_AVAILABLE:
        return None, None, [], {'error': 'sklearn not installed'}
    feats = [c for c in feature_df.columns if c not in ['open','high','low','close','volume']]
    feats = [f for f in feats if feature_df[f].isnull().sum() == 0]
    valid_idx = labels.dropna().index if labels is not None else []
    if len(valid_idx) == 0:
        return None, None, feats, {'error': 'no valid samples', 'samples': 0}
    X = feature_df.loc[valid_idx, feats].values
    y = labels.loc[valid_idx].values
    if len(y) < CONFIG.get("MIN_TRAIN_SAMPLES", 160):
        return None, None, feats, {'error': 'not enough samples', 'samples': int(len(y))}
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=CONFIG.get("RANDOM_STATE",42), stratify=y)
    except Exception as e:
        logging.debug(f"train_test_split error: {e}")
        return None, None, feats, {'error': 'split error', 'exception': safe_str(e)}
    rf = RandomForestClassifier(n_estimators=CONFIG.get("RF_ESTIMATORS",200), random_state=CONFIG.get("RANDOM_STATE",42), n_jobs=-1)
    mlp = MLPClassifier(hidden_layer_sizes=CONFIG.get("MLP_HIDDEN",(96,48)), max_iter=500, random_state=CONFIG.get("RANDOM_STATE",42))
    try:
        rf.fit(X_train, y_train)
    except Exception as e:
        logging.debug(f"rf.fit error: {e}")
        return None, None, feats, {'error': 'rf fit failed', 'exception': safe_str(e)}
    try:
        mlp.fit(X_train, y_train)
    except Exception as e:
        logging.debug(f"mlp.fit error: {e}")
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
# FVG detection
# -------------------------
def find_fvgs(df: pd.DataFrame, lookback: int = 200) -> List[Tuple[pd.Timestamp, float, float, str]]:
    out = []
    if df is None or len(df) < 3:
        return out
    rr = df[-lookback:] if lookback and len(df) > lookback else df
    idxs = rr.index
    for i in range(2, len(rr)):
        c0 = rr.iloc[i-2]; c1 = rr.iloc[i-1]; c2 = rr.iloc[i]
        if c2['low'] > c0['high']:
            out.append((idxs[i], float(c0['high']), float(c2['low']), 'bull'))
        if c2['high'] < c0['low']:
            out.append((idxs[i], float(c2['high']), float(c0['low']), 'bear'))
    return out

def fvg_vote(df: pd.DataFrame, latest_idx: pd.Timestamp) -> Tuple[float, Optional[str]]:
    try:
        fvgs = find_fvgs(df, lookback=200)
        if not fvgs:
            return 0.0, None
        last_price = float(df['close'].loc[latest_idx])
        for t, low, high, typ in fvgs[-10:]:
            if typ == 'bull' and low <= last_price <= high:
                return -0.6, f"FVG bull пересечена — возможный sell-limit (зона {low:.4f}-{high:.4f})"
            if typ == 'bear' and low <= last_price <= high:
                return 0.6, f"FVG bear пересечена — возможный buy-limit (зона {low:.4f}-{high:.4f})"
        return 0.0, None
    except Exception as e:
        logging.debug(f"fvg_vote error: {e}")
        return 0.0, None

# -------------------------
# Volume Profile (POC/VAH/VAL)
# -------------------------
def compute_volume_profile(df: pd.DataFrame, bins: int = 50):
    try:
        prices = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df['volume'].values
        pmin, pmax = prices.min(), prices.max()
        if pmax <= pmin:
            return None
        edges = np.linspace(pmin, pmax, bins+1)
        vols = np.zeros(bins, dtype=float)
        tp = prices.values
        for i, v in enumerate(vol):
            idx = np.searchsorted(edges, tp[i], side='right') - 1
            if idx < 0:
                idx = 0
            if idx >= bins:
                idx = bins-1
            vols[idx] += v
        if vols.sum() <= 0:
            return None
        poc_idx = int(np.argmax(vols))
        poc = 0.5 * (edges[poc_idx] + edges[poc_idx+1])
        sorted_idx = np.argsort(-vols)
        cum = 0.0
        target = vols.sum() * CONFIG.get("VALUE_AREA_SHARE", 0.7)
        selected = np.zeros(bins, dtype=bool)
        for idx in sorted_idx:
            selected[idx] = True
            cum += vols[idx]
            if cum >= target:
                break
        va_prices = []
        for i in range(bins):
            if selected[i]:
                va_prices.append(0.5*(edges[i]+edges[i+1]))
        if not va_prices:
            return {'poc': poc, 'vah': poc, 'val': poc, 'edges': edges, 'vols': vols}
        vah = max(va_prices); val = min(va_prices)
        return {'poc': poc, 'vah': vah, 'val': val, 'edges': edges, 'vols': vols}
    except Exception as e:
        logging.debug(f"compute_volume_profile error: {e}")
        return None

def vp_vote(df: pd.DataFrame, latest_idx: pd.Timestamp) -> Tuple[float, Optional[str]]:
    try:
        vp = compute_volume_profile(df.tail(500), bins=CONFIG.get("VOLUME_PROFILE_BINS",50))
        if not vp:
            return 0.0, None
        price = float(df['close'].loc[latest_idx])
        poc = vp['poc']; vah = vp['vah']; val = vp['val']
        tol = (vp['edges'][-1] - vp['edges'][0]) / 200.0
        if abs(price - vah) <= tol:
            return -0.5, f"Реакция у VAH ({vah:.4f}) — возможный разворот вниз"
        if abs(price - val) <= tol:
            return 0.5, f"Реакция у VAL ({val:.4f}) — возможный разворот вверх"
        if abs(price - poc) <= tol:
            return 0.2, f"Цена около POC ({poc:.4f}) — нейтральный контекст"
        return 0.0, None
    except Exception as e:
        logging.debug(f"vp_vote error: {e}")
        return 0.0, None

# -------------------------
# False Breakout filter
# -------------------------
def is_valid_breakout(df: pd.DataFrame, level: float, side: str, latest_idx: pd.Timestamp) -> Tuple[bool, Optional[str]]:
    try:
        if latest_idx not in df.index:
            return False, None
        recent = df.loc[:latest_idx].tail(6)
        crossed_idx = None
        for idx in recent.index[::-1]:
            o = recent.at[idx, 'open']; c = recent.at[idx, 'close']
            if side == 'BUY' and c > level and o <= level:
                crossed_idx = idx; break
            if side == 'SELL' and c < level and o >= level:
                crossed_idx = idx; break
        if crossed_idx is None:
            return False, None
        o = df.at[crossed_idx, 'open']; c = df.at[crossed_idx, 'close']
        if side == 'BUY' and c <= level:
            return False, "Свеча не закрылась выше уровня"
        if side == 'SELL' and c >= level:
            return False, "Свеча не закрылась ниже уровня"
        atr_series = df['atr14'].ffill().bfill() if 'atr14' in df.columns else None
        atr_val = float(atr_series.loc[crossed_idx]) if atr_series is not None and crossed_idx in atr_series.index else None
        body = abs(c - o)
        if atr_val is not None and atr_val > 1e-9:
            if body < 0.8 * atr_val:
                return False, f"Импульс слабый (body {body:.5f} < 0.8*ATR {atr_val:.5f})"
        later = df.loc[crossed_idx:].head(3)
        for i, idx in enumerate(later.index[1:3], start=1):
            if idx in later.index:
                rev = df.at[idx, 'close']
                if side == 'BUY' and rev < level - 0.5 * body:
                    return False, "Быстрый разворот после пробоя"
                if side == 'SELL' and rev > level + 0.5 * body:
                    return False, "Быстрый разворот после пробоя"
        return True, f"Чистый пробой с подтверждением закрытием и импульсом (body={body:.5f})"
    except Exception as e:
        logging.debug(f"is_valid_breakout error: {e}")
        return False, None

# -------------------------
# DXY correlation helpers
# -------------------------
GOLD_BASES = [
    "XAUUSD", "XAUUSD+", "XAUUSDm", "XAU/USD", "GOLD", "XAU/EUR", "XAUEUR", "XAUJPY", "XAUJPY+",
    "XAUUSD.mt", "XAUUSD.micro", "XAUUSD.i", "XAUUSD.M", "XAUUSD.M+"
]
CURRENCY_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
    "EURGBP","EURJPY","GBPJPY","AUDJPY"
]

def _symbol_candidates(base: str):
    suf = ["", "+", "m", "M", ".m", ".M", "_i", ".i", ".micro"]
    cand = []
    base_no_slash = base.replace("/", "")
    for b in ( [base, base_no_slash] + GOLD_BASES + CURRENCY_PAIRS ):
        for s in suf:
            cand.append(f"{b}{s}")
            cand.append(f"{b}/{s}" if s else b)
    final = []
    for c in cand:
        final.append(c); final.append(c.upper()); final.append(c.lower())
    seen = set(); out = []
    for x in final:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _normalize_rates_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    lc_map = {c.lower(): c for c in df2.columns}
    out = pd.DataFrame()
    if 'time' in lc_map:
        try:
            out['time'] = pd.to_datetime(df2[lc_map['time']], unit='s', utc=True).dt.tz_convert(None)
        except Exception:
            try:
                out['time'] = pd.to_datetime(df2[lc_map['time']], unit='s')
            except Exception:
                out['time'] = df2[lc_map['time']]
    elif 'datetime' in lc_map:
        out['time'] = pd.to_datetime(df2[lc_map['datetime']])
    for k in ['open','high','low','close','last','bid','ask','tick_volume','real_volume','volume']:
        if k in lc_map and k not in out.columns:
            out[k] = pd.to_numeric(df2[lc_map[k]], errors='coerce')
    if 'close' not in out.columns or out['close'].isnull().all():
        if 'last' in out.columns:
            out['close'] = out['last']
    if 'close' not in out.columns or out['close'].isnull().all():
        if 'bid' in out.columns and 'ask' in out.columns:
            out['close'] = (out['bid'] + out['ask']) / 2.0
    if 'close' not in out.columns or out['close'].isnull().all():
        if all(k in out.columns for k in ['open','high','low']):
            out['close'] = (out['open'] + out['high'] + out['low']) / 3.0
    if 'volume' not in out.columns:
        if 'tick_volume' in out.columns:
            out['volume'] = out['tick_volume']
        elif 'real_volume' in out.columns:
            out['volume'] = out['real_volume']
        else:
            out['volume'] = 0.0
    if 'time' in out.columns:
        out = out.set_index('time')
    out = out[['open','high','low','close','volume']].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().bfill().fillna(0.0)
    for c in out.columns:
        try:
            out[c] = out[c].astype('float64')
        except Exception:
            pass
    return out

def get_rates_mt5(symbol: str, timeframe: int, bars: int, verbose: bool = False) -> Optional[pd.DataFrame]:
    if not MT5_AVAILABLE:
        logging.debug("get_rates: MT5 module not available.")
        return None
    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        pass
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(bars))
        if rates is None or len(rates) == 0:
            now = int(time.time())
            rates = mt5.copy_rates_from(symbol, timeframe, now, int(bars))
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df2 = _normalize_rates_df(df)
        if df2 is None or df2.empty:
            return None
        if len(df2) > bars:
            df2 = df2.iloc[-bars:]
        return df2
    except Exception as e:
        logging.debug(f"get_rates_mt5 exception: {e}")
        return None

def correlation_with_dxy(df: pd.DataFrame, timeframe_val: int, bars: int) -> Tuple[Optional[float], Optional[str]]:
    try:
        if not MT5_AVAILABLE:
            return None, "MT5 недоступен для корреляции"
        candidates = ["DXY", "USDX", "USDINDEX", "USDOLLAR", "DX", "USDX.i"]
        dxy_sym = None
        for c in candidates:
            try:
                for cand in _symbol_candidates(c):
                    try:
                        mt5.symbol_select(cand, True)
                        dxy_sym = cand; break
                    except Exception:
                        continue
                if dxy_sym:
                    break
            except Exception:
                continue
        if not dxy_sym:
            return None, "Не найден символ индекса доллара"
        dxy_df = get_rates_mt5(dxy_sym, timeframe_val, bars)
        if dxy_df is None or dxy_df.empty:
            return None, f"Нет данных для {dxy_sym}"
        merged = pd.concat([df['close'].rename('asset'), dxy_df['close'].rename('dxy')], axis=1, join='inner')
        if merged.shape[0] < 30:
            return None, "Недостаточно общих баров для корреляции"
        corr = merged['asset'].pct_change().rolling(50).corr(merged['dxy'].pct_change()).iloc[-1]
        if pd.isna(corr):
            return None, "Корр не вычислима"
        return float(corr), f"Корреляция с {dxy_sym} = {corr:.3f}"
    except Exception as e:
        logging.debug(f"correlation_with_dxy error: {e}")
        return None, None

# -------------------------
# First candle rule (NY 9:30)
# -------------------------
def first_candle_rule_vote(df: pd.DataFrame, latest_idx: pd.Timestamp) -> Tuple[float, Optional[str]]:
    try:
        if df is None or df.empty:
            return 0.0, None
        idx_local = df.index.tz_localize('UTC').tz_convert('America/New_York')
        df_ny = df.copy(); df_ny.index = idx_local
        last_date = idx_local[-1].date()
        target = pd.Timestamp(year=last_date.year, month=last_date.month, day=last_date.day, hour=9, minute=30, tz='America/New_York')
        cand_idx = None
        for ts in df_ny.index:
            if ts >= target:
                cand_idx = ts; break
        if cand_idx is None:
            return 0.0, None
        orig_idx = df.index[df.index.tz_localize('UTC').tz_convert('America/New_York') == cand_idx]
        if len(orig_idx) == 0:
            return 0.0, None
        orig_idx = orig_idx[0]
        first_o = df.at[orig_idx, 'open']; first_h = df.at[orig_idx, 'high']; first_l = df.at[orig_idx, 'low']
        price = float(df['close'].loc[latest_idx])
        if price > first_h:
            return 0.6, f"Правило первой свечи: пробой high первой свечи (9:30 NY) -> BUY"
        if price < first_l:
            return -0.6, f"Правило первой свечи: пробой low первой свечи (9:30 NY) -> SELL"
        return 0.0, None
    except Exception as e:
        logging.debug(f"first_candle_rule_vote error: {e}")
        return 0.0, None

# -------------------------
# Previous day/session levels
# -------------------------
def levels_prev_day_session(df: pd.DataFrame, latest_idx: pd.Timestamp):
    try:
        if df is None or df.empty:
            return {}
        highs = df['high'].resample('D').max()
        lows = df['low'].resample('D').min()
        last_ts = latest_idx
        last_day = pd.Timestamp(last_ts.date())
        try:
            yday = last_day - pd.Timedelta(days=1)
            y_high = float(highs.loc[yday])
            y_low = float(lows.loc[yday])
        except Exception:
            y_high = None; y_low = None
        try:
            period_start = latest_idx - pd.Timedelta(hours=24)
            sess = df.loc[period_start:latest_idx]
            s_high = float(sess['high'].max()); s_low = float(sess['low'].min())
        except Exception:
            s_high = None; s_low = None
        return {'y_high': y_high, 'y_low': y_low, 's_high': s_high, 's_low': s_low}
    except Exception as e:
        logging.debug(f"levels_prev_day_session error: {e}")
        return {}

def levels_vote(df: pd.DataFrame, latest_idx: pd.Timestamp) -> Tuple[float, Optional[str]]:
    try:
        lv = levels_prev_day_session(df, latest_idx)
        price = float(df['close'].loc[latest_idx])
        if not lv:
            return 0.0, None
        reasons = []
        v = 0.0
        for key, lvl in lv.items():
            if lvl is None:
                continue
            diff = abs(price - lvl)
            tol = (df['high'].max() - df['low'].min()) / 300.0
            if diff <= tol:
                if key.endswith('high'):
                    v -= 0.4; reasons.append(f"Цена у уровня {key} ({lvl:.4f}) — риск разворота вниз")
                else:
                    v += 0.4; reasons.append(f"Цена у уровня {key} ({lvl:.4f}) — возможный отскок вверх")
        if reasons:
            return v, "; ".join(reasons)
        return 0.0, None
    except Exception as e:
        logging.debug(f"levels_vote error: {e}")
        return 0.0, None

# -------------------------
# Expert60 (agg of indicators + new votes)
# -------------------------
def expert60_votes(df: pd.DataFrame, idx: pd.Timestamp, timeframe_val: Optional[int] = None, bars: Optional[int] = None) -> Tuple[float, List[str]]:
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
    # New votes
    try:
        v_fvg, reason_fvg = fvg_vote(df, idx)
        if v_fvg != 0.0:
            votes += v_fvg; reasons.append(reason_fvg)
        v_vp, reason_vp = vp_vote(df, idx)
        if v_vp != 0.0:
            votes += v_vp; reasons.append(reason_vp)
        v_first, reason_first = first_candle_rule_vote(df, idx)
        if v_first != 0.0:
            votes += v_first; reasons.append(reason_first)
        v_lvl, reason_lvl = levels_vote(df, idx)
        if v_lvl != 0.0:
            votes += v_lvl; reasons.append(reason_lvl)
        try:
            corr, corr_reason = correlation_with_dxy(df, timeframe_val or 15, bars or 1500)
            if corr is not None:
                if corr < -0.25:
                    votes += 0.4; reasons.append(f"Отрицательная корреляция с долларом ({corr:.2f}) — поддержка движения золота")
                elif corr > 0.25:
                    votes -= 0.4; reasons.append(f"Положительная корреляция с долларом ({corr:.2f}) — осторожно")
                else:
                    reasons.append(f"Корреляция с долларом нейтральна ({corr:.2f})")
            elif corr_reason:
                reasons.append(corr_reason)
        except Exception as e:
            logging.debug(f"correlation check error: {e}")
    except Exception as e:
        logging.debug(f"expert60 extra votes error: {e}")
    return votes, reasons

# -------------------------
# Order Manager
# -------------------------
class OrderManager:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.orders = []
        self.position_id = 0

    def place_order(self, symbol: str, side: str, volume: float, sl: Optional[float], tp: Optional[float], comment: str = "") -> Dict[str, Any]:
        self.position_id += 1
        order = {
            "id": self.position_id,
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "time": now_iso(),
            "status": "simulated" if self.dry_run else "sent"
        }
        self.orders.append(order)
        logging.info("Order placed: %s", json.dumps(order, ensure_ascii=False))
        return order

    def close_all(self):
        closed = []
        while self.orders:
            o = self.orders.pop()
            o['closed_time'] = now_iso()
            o['status'] = 'closed_sim'
            closed.append(o)
            logging.info("Closed: %s", json.dumps(o, ensure_ascii=False))
        return closed

# -------------------------
# SL/TP/Lot calc
# -------------------------
def calc_sl_tp_and_lot(info, price: float, side: str, atr_val: float, account_balance: float):
    if atr_val is None or atr_val <= 0:
        return None, None, None, None, "N/A"
    sl_dist = CONFIG.get("SL_ATR_MULT",1.3) * atr_val
    if side == 'BUY':
        sl = price - sl_dist
        tp1 = price + CONFIG.get("TP1_ATR",1.0) * atr_val
        tp2 = price + CONFIG.get("TP2_ATR",2.0) * atr_val
        tp3 = price + CONFIG.get("TP3_ATR",3.5) * atr_val
    elif side == 'SELL':
        sl = price + sl_dist
        tp1 = price - CONFIG.get("TP1_ATR",1.0) * atr_val
        tp2 = price - CONFIG.get("TP2_ATR",2.0) * atr_val
        tp3 = price - CONFIG.get("TP3_ATR",3.5) * atr_val
    else:
        return None, None, None, None, "N/A"
    tick_size = getattr(info, 'trade_tick_size', None) or getattr(info, 'point', None) or 0.01
    tick_value = getattr(info, 'trade_tick_value', None)
    price_move = abs(price - sl)
    if tick_value and tick_size:
        try:
            stop_value_per_lot = (price_move / tick_size) * tick_value
        except Exception:
            stop_value_per_lot = price_move * 100.0
    else:
        stop_value_per_lot = price_move * 100.0
    risk_amount = account_balance * CONFIG.get("RISK_PER_TRADE_PCT", 0.01)
    if stop_value_per_lot <= 0:
        lots = 0.0
    else:
        lots = risk_amount / stop_value_per_lot
    vol_step = getattr(info, 'volume_step', 0.01) or 0.01
    lots = max(getattr(info, 'volume_min', 0.01) or 0.01, lots)
    try:
        rounded = math.floor(lots / vol_step) * vol_step
        lots = round(rounded, 2)
    except Exception:
        lots = round(lots, 2)
    return float(sl), float(tp1), float(tp2), float(tp3), lots

# -------------------------
# Signal Engine
# -------------------------
class SignalEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mdp = MarketDataProvider(config["SYMBOL"], config["TIMEFRAME_MINUTES"], config["HISTORY_BARS"])
        self.mm_intent = MarketMakerIntent(config_mm=config.get("MM", {}))
        self.news_fetcher = NewsFetcher(api_key=config.get("NEWSAPI_KEY", ""), rss_sources=config.get("RSS_SOURCES", []))
        self.news_manual = NEWS_MANUAL
        self.order_manager = OrderManager(dry_run=config.get("DRY_RUN", True))
        self.weights = config.get("WEIGHTS", {})
        # CSV header
        if not os.path.exists(config.get("SIGNAL_OUTPUT_CSV")):
            with open(config.get("SIGNAL_OUTPUT_CSV"), "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["time", "symbol", "signal", "confidence", "volume", "sl", "tp", "details"])

    def maybe_connect_mt5(self):
        if MT5_AVAILABLE and (self.config.get("MT5_LOGIN") or self.config.get("MT5_PASSWORD")):
            return self.mdp.connect_mt5(login=self.config.get("MT5_LOGIN"),
                                        password=self.config.get("MT5_PASSWORD"),
                                        server=self.config.get("MT5_SERVER"))
        return False

    def compute_lot_from_risk(self, price: float, sl_pips: float) -> float:
        risk_usd = self.config.get("RISK", {}).get("risk_per_trade_usd", 100.0)
        value_per_pip_per_lot = 1.0
        if sl_pips <= 0:
            return 0.01
        lot = safe_div(risk_usd, abs(sl_pips) * value_per_pip_per_lot)
        lot = max(0.01, min(10.0, round(lot, 2)))
        return lot

    def create_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        df_ind = add_indicators(df)
        latest_idx = df_ind.index[-1]
        expert_score_raw, expert_reasons = expert60_votes(df_ind, latest_idx, timeframe_val=self.config.get("TIMEFRAME_MINUTES",1), bars=self.config.get("HISTORY_BARS",500))
        expert_score = max(-1.0, min(1.0, expert_score_raw / 6.0))
        mm = self.mm_intent.analyze(df_ind)
        expert_score_mapped = safe_div(expert_score + 1.0, 2.0)
        news_agg = self.news_fetcher.aggregate_news_sentiment()
        news_score = safe_div(news_agg['avg_sentiment'] + 1.0, 2.0)
        manual_news_score = self.news_manual.get_sentiment()
        news_combined = (news_score * 0.7) + ((manual_news_score + 1.0)/2.0 * 0.3)
        ml_score = 0.5  # neutral default
        ml_weight = self.config.get("ML_WEIGHT", 0.55)
        ind_weight = self.config.get("IND_WEIGHT", 0.35)
        news_weight = self.config.get("NEWS_WEIGHT", 0.10)
        close = df_ind['close']
        momentum = safe_div(close.iloc[-1] - close.iloc[-5], close.iloc[-5]) if len(close) >= 6 else 0.0
        dir_sign = 1 if momentum > 0 else -1 if momentum < 0 else 0
        total_conf = (ml_weight * ml_score + ind_weight * (expert_score_mapped) + news_weight * news_combined)
        confidence = max(0.0, min(1.0, float(total_conf)))
        if expert_score > 0.15:
            signal = 'buy'
        elif expert_score < -0.15:
            signal = 'sell'
        else:
            if dir_sign > 0 and news_agg['avg_sentiment'] >= 0:
                signal = 'buy'
            elif dir_sign < 0 and news_agg['avg_sentiment'] <= 0:
                signal = 'sell'
            else:
                signal = 'hold'
        explanation = {
            "expert_raw": expert_score_raw,
            "expert_norm": expert_score,
            "expert_reasons": expert_reasons,
            "market_intent": mm,
            "news": {"avg_sentiment": news_agg['avg_sentiment'], "count": news_agg['article_count'], "manual": self.news_manual.get_sentiment()},
            "momentum": float(momentum),
            "raw_confidence": confidence
        }
        logging.info("Signal created: %s (conf=%.3f) : %s", signal, confidence, explanation)
        return {"signal": signal, "confidence": confidence, "explanation": explanation, "df_ind": df_ind}

    def act_on_signal(self, signal_obj: Dict[str, Any]):
        signal = signal_obj.get("signal")
        conf = signal_obj.get("confidence", 0.0)
        explanation = signal_obj.get("explanation", {})
        df_ind = signal_obj.get("df_ind")
        threshold = 0.55
        latest_price = float(df_ind['close'].iloc[-1])
        if conf < threshold or signal == 'hold':
            logging.info("Holding. signal=%s conf=%.2f (threshold=%.2f)", signal, conf, threshold)
            self.log_signal(signal, conf, 0.0, None, None, explanation)
            return None
        atr = float(df_ind['atr14'].iloc[-1]) if 'atr14' in df_ind.columns else 0.5
        sl_distance = max(atr * 0.5, 0.3)
        if signal == 'buy':
            sl_price = round(float(latest_price - sl_distance), 3)
            tp_price = round(float(latest_price + sl_distance * 2.0), 3)
        else:
            sl_price = round(float(latest_price + sl_distance), 3)
            tp_price = round(float(latest_price - sl_distance * 2.0), 3)
        lot = self.compute_lot_from_risk(price=latest_price, sl_pips=sl_distance)
        order = self.order_manager.place_order(self.config['SYMBOL'], side=signal, volume=lot, sl=sl_price, tp=tp_price,
                                               comment=f"auto signal conf={conf:.2f}")
        self.log_signal(signal, conf, lot, sl_price, tp_price, explanation)
        return order

    def log_signal(self, signal: str, conf: float, volume: float, sl: Optional[float], tp: Optional[float], details: Any):
        row = [now_iso(), self.config['SYMBOL'], signal, f"{conf:.4f}", volume, sl, tp, json.dumps(details, ensure_ascii=False)]
        with open(self.config.get("SIGNAL_OUTPUT_CSV"), "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def run_once(self):
        self.maybe_connect_mt5()
        df = None
        if self.mdp.connected:
            df = self.mdp.get_history_mt5()
        if df is None:
            logging.warning("No MT5 data — пытаюсь загрузить из CSV/synthetic.")
            path = f"{self.config['SYMBOL']}_sample.csv"
            df = self.mdp.get_history_from_csv(path)
            if df is None:
                df = self._generate_synthetic()
                logging.info("Использую синтетические данные для теста.")
        sig = self.create_signal(df)
        order = self.act_on_signal(sig)
        return {"signal": sig, "order": order}

    def run_loop(self, iterations: int = 1000000):
        logging.info("Starting main loop. DRY_RUN=%s", self.config.get("DRY_RUN"))
        i = 0
        try:
            while i < iterations:
                res = self.run_once()
                time.sleep(self.config.get("POLL_SECONDS", 10))
                i += 1
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        except Exception:
            logging.exception("Unhandled exception in run_loop")
        finally:
            logging.info("Loop stopped. Closing positions (simulated).")
            self.order_manager.close_all()

    def _generate_synthetic(self) -> pd.DataFrame:
        n = max(200, self.config.get("HISTORY_BARS", 500))
        dt_index = pd.date_range(end=datetime.utcnow(), periods=n, freq=f'{self.config["TIMEFRAME_MINUTES"]}min')
        price = 1900.0 + np.cumsum(np.random.normal(0, 0.5, size=n))
        high = price + np.abs(np.random.normal(0, 0.3, size=n))
        low = price - np.abs(np.random.normal(0, 0.3, size=n))
        openp = price + np.random.normal(0, 0.1, size=n)
        close = price + np.random.normal(0, 0.1, size=n)
        volume = np.random.randint(1, 100, size=n)
        df = pd.DataFrame({'open': openp, 'high': high, 'low': low, 'close': close, 'tick_volume': volume, 'volume': volume}, index=dt_index)
        df.index.name = 'time'
        return df

# -------------------------
# Result dataclass (for analyze wrapper)
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
# Optional analyze() similar to 11.txt
# -------------------------
def analyze(symbol: str, timeframe_value: int, bars: int = None) -> Result:
    bars = bars or CONFIG.get("DEFAULT_BARS", 1500)
    df = None
    if MT5_AVAILABLE:
        df = get_rates_mt5(symbol, timeframe_value, bars)
    if df is None:
        raise RuntimeError(f"Не удалось загрузить данные для {symbol}. Проверь MT5 или CSV.")
    if 'close' not in df.columns or len(df) < CONFIG.get("MIN_TRAIN_SAMPLES", 160):
        raise RuntimeError("Недостаточно данных для анализа/labels.")
    info = None
    try:
        info = mt5.symbol_info(symbol) if MT5_AVAILABLE else None
    except Exception:
        info = None
    df_ind = add_indicators(df)
    labels = make_labels(df_ind)
    rf = mlp = None; feats = []; ml_metrics = {}
    if SKLEARN_AVAILABLE:
        try:
            rf, mlp, feats, ml_metrics = train_models(df_ind, labels)
        except Exception as e:
            logging.debug(f"train_models exception: {e}")
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
                    logging.debug(f"mlp predict_proba error: {e}")
        except Exception as e:
            logging.debug(f"model predict error: {e}")
    expert_score, expert_reasons = expert60_votes(df_ind, latest_idx, timeframe_val=timeframe_value, bars=bars)
    ind_component = max(-1.0, min(1.0, expert_score / 6.0))
    news_component = max(-1.0, min(1.0, NEWS_MANUAL.get_sentiment()))
    ml_score = probs['BUY'] - probs['SELL']
    combined_score = CONFIG.get("ML_WEIGHT",0.55) * ml_score + CONFIG.get("IND_WEIGHT",0.35) * ind_component + CONFIG.get("NEWS_WEIGHT",0.10) * news_component
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
        account = mt5.account_info() if MT5_AVAILABLE else None
    except Exception:
        account = None
    balance = float(getattr(account, 'balance', 1000.0))
    sl = tp1 = tp2 = tp3 = None; lots = None
    try:
        if agg_signal in ("ПОКУПАТЬ","ПРОДАВАТЬ"):
            side = 'BUY' if agg_signal == 'ПОКУПАТЬ' else 'SELL'
            level = price
            if side == 'BUY':
                level = float(df_ind['donch_high'].iloc[-2]) if 'donch_high' in df_ind.columns else price*0.999
            else:
                level = float(df_ind['donch_low'].iloc[-2]) if 'donch_low' in df_ind.columns else price*1.001
            valid, reason = is_valid_breakout(df_ind, level, side, latest_idx)
            if not valid:
                agg_confidence = max(agg_confidence * 0.4, 5.0)
                agg_signal = "ДЕРЖАТЬ"
                expert_reasons.append(f"Проверка пробоя не пройдена: {reason}")
            else:
                expert_reasons.append(f"Проверка пробоя пройдена: {reason}")
                try:
                    sl, tp1, tp2, tp3, lots = calc_sl_tp_and_lot(info, price, side, atr_val, balance)
                except Exception as e:
                    logging.debug(f"calc_sl_tp_and_lot error: {e}")
    except Exception as e:
        logging.debug(f"analyze breakout check error: {e}")
    reasons = [
        f"ML: BUY={probs['BUY']:.3f} HOLD={probs['HOLD']:.3f} SELL={probs['SELL']:.3f}",
        f"Expert60 score={expert_score:.2f} (norm={ind_component:.2f})",
        f"News sentiment={news_component:.2f}"
    ] + expert_reasons
    return Result(symbol=symbol, timeframe=str(timeframe_value), price=price, model_preds=probs, agg_signal=agg_signal,
                  agg_confidence=agg_confidence, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3, lots=lots, ml_metrics=ml_metrics or {}, reasons=reasons)

# -------------------------
# GUI (optional)
# -------------------------
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

if PYSIDE_AVAILABLE:
    class WorkerSignals(QObject):
        finished = Signal(object)
        error = Signal(tuple)
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
            self.setWindowTitle("TraderPro AI — Unified")
            self.resize(1100, 760)
            self.setStyleSheet(APP_QSS)
            self._init_ui()
            try:
                if MT5_AVAILABLE:
                    mt5.initialize()
                    self.statusL.setText("MT5 инициализирован")
                else:
                    self.statusL.setText("MT5 недоступен")
            except Exception as e:
                self.statusL.setText(f"[MT5 error] {e}")
        def _init_ui(self):
            layout = QVBoxLayout()
            top = QHBoxLayout()
            top.addWidget(QLabel("Символ:"))
            self.symbolBox = QComboBox(); self.symbolBox.setEditable(True)
            self.symbolBox.addItems([CONFIG.get("SYMBOL")])
            top.addWidget(self.symbolBox)
            top.addWidget(QLabel("ТФ:"))
            self.tfBox = QComboBox(); self.tfBox.addItems(['M1','M5','M15','M30','H1','H4','D1']); self.tfBox.setCurrentText(CONFIG.get("DEFAULT_TF","M15"))
            top.addWidget(self.tfBox)
            top.addWidget(QLabel("Свечей:"))
            self.barSpin = QSpinBox(); self.barSpin.setRange(300, 8000); self.barSpin.setValue(CONFIG.get("DEFAULT_BARS",1500))
            top.addWidget(self.barSpin)
            self.btnAnalyze = QPushButton("АНАЛИЗ"); top.addWidget(self.btnAnalyze)
            self.auto = QCheckBox("Авто"); top.addWidget(self.auto)
            top.addWidget(QLabel("Интервал (с):"))
            self.intSpin = QSpinBox(); self.intSpin.setRange(5, 3600); self.intSpin.setValue(60); top.addWidget(self.intSpin)
            self.saveBtn = QPushButton("Сохранить логи…"); top.addWidget(self.saveBtn)
            layout.addLayout(top)
            news_layout = QHBoxLayout()
            news_layout.addWidget(QLabel("Новости →"))
            self.newsEdit = QLineEdit(); self.newsEdit.setPlaceholderText("Вставьте текст новости…")
            news_layout.addWidget(self.newsEdit)
            self.btnNews = QPushButton("Учитывать новость"); news_layout.addWidget(self.btnNews)
            layout.addLayout(news_layout)
            srow = QHBoxLayout()
            self.lamp = QLabel(); self.lamp.setObjectName("SignalLamp"); self.lamp.setFixedSize(24, 24); self._setLampColor("gray")
            self.statusL = QLabel("Готово")
            srow.addWidget(QLabel("Сигнал:")); srow.addWidget(self.lamp); srow.addWidget(self.statusL)
            srow.addStretch(1)
            layout.addLayout(srow)
            self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0); layout.addWidget(self.progress)
            self.out = QTextEdit(); self.out.setReadOnly(True); layout.addWidget(self.out)
            layout.addWidget(QLabel("Обоснование сигнала:"))
            self.reasonsList = QListWidget(); layout.addWidget(self.reasonsList)
            opts = QHBoxLayout()
            self.chk_save_df_on_err = QCheckBox("Дамп DF при ошибке"); self.chk_save_df_on_err.setChecked(True); opts.addWidget(self.chk_save_df_on_err)
            opts.addStretch(1); layout.addLayout(opts)
            self.setLayout(layout)
            self.btnAnalyze.clicked.connect(self.on_analyze)
            self.btnNews.clicked.connect(self.on_news)
            self.timer = QTimer(); self.timer.timeout.connect(self.on_analyze)
            self.auto.stateChanged.connect(self._toggle_timer)
            self.saveBtn.clicked.connect(self.save_logs)
        def _setLampColor(self, name: str):
            m = {'green':'#22c55e','red':'#ef4444','yellow':'#facc15','gray':'#6b7280'}
            col = m.get(name, '#6b7280')
            self.lamp.setStyleSheet(f"#SignalLamp {{ background: {col}; }}")
        def _toggle_timer(self):
            if self.auto.isChecked():
                self.timer.start(self.intSpin.value() * 1000)
                self.out.append("Автоматический режим включён")
            else:
                self.timer.stop()
                self.out.append("Автоматический режим выключен")
        def save_logs(self):
            if not os.path.exists("signals_log.csv"):
                self.out.append("Нет логов для сохранения.")
                return
            fname, _ = QFileDialog.getSaveFileName(self, "Сохранить логи как…", "signals_log.csv", "CSV Files (*.csv)")
            if fname:
                try:
                    import shutil
                    shutil.copyfile("signals_log.csv", fname)
                    self.out.append(f"Логи сохранены в {fname}")
                except Exception as e:
                    self.out.append(f"Ошибка при сохранении: {e}")
        def on_news(self):
            txt = self.newsEdit.text().strip()
            if txt:
                NEWS_MANUAL.ingest_text(txt)
                self.out.append(f"Новости учтены. sentiment={NEWS_MANUAL.get_sentiment():+.2f}")
                self.newsEdit.clear()
        def on_analyze(self):
            self.progress.setValue(5)
            user_symbol = self.symbolBox.currentText().strip()
            if not user_symbol:
                self.out.append("Введите символ.")
                return
            self.progress.setValue(10)
            self.out.append(f"Запуск анализа: {user_symbol} TF={self.tfBox.currentText()} bars={self.barSpin.value()}")
            signals = WorkerSignals()
            signals.finished.connect(self._on_worker_finished)
            signals.error.connect(self._on_worker_error)
            tf_map = {'M1':1,'M5':5,'M15':15,'M30':30,'H1':60,'H4':240,'D1':1440}
            tf_val = tf_map.get(self.tfBox.currentText(), 15)
            worker = AnalyzerWorker(user_symbol, tf_val, int(self.barSpin.value()), signals)
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
                    lines.append(f"TP1: {res.tp1:.4f} (ATR x{CONFIG.get('TP1_ATR')})")
                    lines.append(f"TP2: {res.tp2:.4f} (ATR x{CONFIG.get('TP2_ATR')})")
                    lines.append(f"TP3: {res.tp3:.4f} (ATR x{CONFIG.get('TP3_ATR')})")
                    lines.append(f"Реком. лот: {res.lots}")
                lines.append("\nМетрики модели:")
                for k, v in res.ml_metrics.items():
                    lines.append(f"  {k}: {v}")
                lines.append("\nОбоснование:")
                for r in res.reasons:
                    lines.append(" - " + (r or ""))
                self.out.append("\n".join(lines))
                self.reasonsList.clear()
                for r in res.reasons:
                    self.reasonsList.addItem(QListWidgetItem(r))
                if res.agg_signal == "ПОКУПАТЬ":
                    self._setLampColor("green")
                elif res.agg_signal == "ПРОДАВАТЬ":
                    self._setLampColor("red")
                else:
                    self._setLampColor("yellow")
                self.progress.setValue(100)
            except Exception as e:
                self._on_worker_error((e, traceback.format_exc()))
        def _on_worker_error(self, err):
            e, tb = err
            self.out.append(f"Error: {e}\n{tb}")
            self._setLampColor("gray")
            self.progress.setValue(0)

# -------------------------
# Main entry
# -------------------------
def main():
    logging.info("TraderPro AI full starting...")
    se = SignalEngine(CONFIG)
    res = se.run_once()
    logging.info("Single-run finished.")
    # For continuous operation uncomment:
    # se.run_loop()

if __name__ == "__main__":
    main()
