# -*- coding: utf-8 -*-
"""
Crypto TA Desktop App + AI confirmation (PyQt5/PySide6 + ccxt + scikit-learn)

✅ Что внутри:
- Универсальный импорт Qt (PyQt5 -> PySide6 fallback)
- Биржи: Binance / Bybit / OKX (через ccxt)
- Таймфреймы: 1m, 5m, 10m, 15m, 30m, 1h, 4h, 1d
- Индикаторы: EMA50/EMA200, MACD, RSI(14), ATR(14), OBV (+ наклон)
- Эвристический сигнал (trend/momentum/RSI) + ML (LogisticRegression) подтверждение
- TP1/TP2/TP3 + SL корректно для LONG и SHORT (и порядок целей выровнен)
- Нормализация и резолвинг символа из "кривого" ввода

Установить зависимости:
pip install ccxt numpy pandas scikit-learn
(и либо PyQt5, либо PySide6)

Запуск:
python app_ai_ta.py
"""

import sys
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import ccxt

# --- Универсальный импорт Qt ---
try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
        QGroupBox, QGridLayout
    )
    qt_lib = "PyQt5"
except ImportError:
    from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
        QGroupBox, QGridLayout
    )
    qt_lib = "PySide6"

# --- ML ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------- Data classes ----------

@dataclass
class SignalResult:
    side: str                 # "LONG" | "SHORT" | "NEUTRAL"
    confidence: float         # 0..1
    reason: str               # текстовое объяснение
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    sl: Optional[float]


# ---------- Exchanges ----------

SUPPORTED_EXCHANGES = ["binance", "bybit", "okx"]


def create_exchange(name: str, market_type: str):
    """
    Создаёт экземпляр ccxt-экчейнджа с нужным defaultType: spot/swap.
    """
    if name not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Биржа '{name}' не поддерживается.")
    klass = getattr(ccxt, name)
    ex = klass({
        "enableRateLimit": True,
        "options": {"defaultType": market_type}  # важно для bybit/okx/binance
    })
    return ex


# ---------- Indicators ----------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = ema(macd_line, 9)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(span=period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = (df["high"] - df["low"]).abs()
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).fillna(0.0).cumsum()

def swing_levels(df: pd.DataFrame, lookback: int = 14) -> Tuple[Optional[float], Optional[float]]:
    if len(df) < lookback:
        return None, None
    recent = df.tail(lookback)
    hi = float(recent["high"].max()) if not recent["high"].isna().all() else None
    lo = float(recent["low"].min()) if not recent["low"].isna().all() else None
    return hi, lo


# ---------- Heuristic signal ----------

def heuristic_signal(df: pd.DataFrame) -> SignalResult:
    last = df.iloc[-1]; prev = df.iloc[-2]
    score, reasons = 0.0, []

    # Trend bias
    if last["ema50"] > last["ema200"]:
        score += 0.25; reasons.append("EMA50 > EMA200 (uptrend)")
    elif last["ema50"] < last["ema200"]:
        score -= 0.25; reasons.append("EMA50 < EMA200 (downtrend)")

    # MACD cross
    macd_cross_up = prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]
    macd_cross_dn = prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]
    if macd_cross_up:
        score += 0.30; reasons.append("MACD bullish cross")
    if macd_cross_dn:
        score -= 0.30; reasons.append("MACD bearish cross")

    # MACD momentum slope
    hist_slope = (last["macd"] - last["macd_signal"]) - (prev["macd"] - prev["macd_signal"])
    if hist_slope > 0:
        score += 0.15; reasons.append("MACD momentum rising")
    else:
        score -= 0.15; reasons.append("MACD momentum falling")

    # RSI regime
    if last["rsi14"] > 55:
        score += 0.20; reasons.append("RSI > 55 (bullish regime)")
    elif last["rsi14"] < 45:
        score -= 0.20; reasons.append("RSI < 45 (bearish regime)")

    # OBV slope confirmation (за последние 5 свечей)
    if "obv" in df.columns:
        obv_slope = df["obv"].diff().rolling(5).mean().iloc[-1]
        if obv_slope > 0:
            score += 0.1; reasons.append("OBV rising")
        elif obv_slope < 0:
            score -= 0.1; reasons.append("OBV falling")

    score = max(-1.0, min(1.0, score))
    side = "LONG" if score > 0.15 else ("SHORT" if score < -0.15 else "NEUTRAL")
    confidence = float(abs(score))

    # Risk levels
    atr_val = float(last.get("atr14", np.nan))
    price = float(last["close"])
    swing_hi, swing_lo = swing_levels(df, lookback=14)

    tp1 = tp2 = tp3 = sl = None
    if side == "LONG":
        sl = max(price - atr_val * 1.2, (swing_lo or price - atr_val * 1.2))
        tp1 = price + atr_val * 1.0
        tp2 = price + atr_val * 1.5
        tp3 = min(price + atr_val * 2.0, (swing_hi or price + atr_val * 2.0))
        # гарантируем возрастание целей
        tps = sorted([tp for tp in [tp1, tp2, tp3] if tp is not None])
        tp1, tp2, tp3 = (tps + [None, None, None])[:3]
    elif side == "SHORT":
        sl = min(price + atr_val * 1.2, (swing_hi or price + atr_val * 1.2))
        # цели вниз
        tp1 = price - atr_val * 1.0
        tp2 = price - atr_val * 1.5
        tp3 = max(price - atr_val * 2.0, (swing_lo or price - atr_val * 2.0))
        # гарантируем убывание целей (для шорта TP1 > TP2 > TP3 по числу)
        tps = sorted([tp for tp in [tp1, tp2, tp3] if tp is not None], reverse=True)
        tp1, tp2, tp3 = (tps + [None, None, None])[:3]

    return SignalResult(side, confidence, "; ".join(reasons), tp1, tp2, tp3, sl)


# ---------- AI model (ML) ----------

class AIMarketModel:
    """
    Простая ML-модель (LogisticRegression) для подтверждения направления.
    Признаки: ema50, ema200, macd, macd_signal, rsi14, atr14, obv_slope(5)
    Разметка: next_close > close -> 1 (LONG), иначе 0 (SHORT)
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=200, solver="lbfgs")
        self.is_trained = False

    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["ema50"] = df["ema50"]
        X["ema200"] = df["ema200"]
        X["macd"] = df["macd"]
        X["macd_signal"] = df["macd_signal"]
        X["rsi14"] = df["rsi14"]
        X["atr14"] = df["atr14"]
        if "obv" in df.columns:
            X["obv_slope_5"] = df["obv"].diff().rolling(5).mean()
        else:
            X["obv_slope_5"] = 0.0
        return X

    def train(self, df: pd.DataFrame) -> bool:
        # нужно достаточное кол-во баров и заполненные индикаторы
        if len(df) < 210:
            self.is_trained = False
            return False
        X = self._make_features(df).copy()
        y = (df["close"].shift(-1) > df["close"]).astype(int)
        data = pd.concat([X, y.rename("y")], axis=1).dropna()
        if len(data) < 100:
            self.is_trained = False
            return False

        X_clean = data.drop(columns=["y"]).values
        y_clean = data["y"].values

        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y_clean)
        self.is_trained = True
        return True

    def predict_proba(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Возврат вероятностей (p_short, p_long)."""
        if not self.is_trained:
            return 0.5, 0.5
        X = self._make_features(df).iloc[[-1]].fillna(0.0).values
        X_scaled = self.scaler.transform(X)
        proba_long = float(self.model.predict_proba(X_scaled)[0, 1])
        return 1.0 - proba_long, proba_long

    def predict_side(self, df: pd.DataFrame, threshold: float = 0.6) -> str:
        p_short, p_long = self.predict_proba(df)
        if p_long >= threshold:
            return "LONG"
        if p_short >= threshold:
            return "SHORT"
        return "NEUTRAL"


ai_model = AIMarketModel()


def combined_signal(df: pd.DataFrame) -> SignalResult:
    """
    Комбинируем эвристику + AI:
    - обучаем модель на текущем df (локально)
    - получаем эвристику
    - получаем предсказание ИИ (направление + уверенность)
    - если совпало — повышаем confidence и отмечаем в reason
    - если расходится — отмечаем предупреждение
    - если эвристика NEUTRAL, но AI уверен > 0.6 — принимаем сторону AI
    """
    # обучаем AI (молчаливо, если данных мало — is_trained=False)
    ai_model.train(df)

    heur = heuristic_signal(df)
    p_short, p_long = ai_model.predict_proba(df)
    ai_side = ai_model.predict_side(df, threshold=0.60)

    if heur.side == "NEUTRAL" and ai_side in ("LONG", "SHORT"):
        heur.side = ai_side
        heur.confidence = max(heur.confidence, 0.55)
        heur.reason += f"; AI override: {ai_side} (p_long={p_long:.2f}, p_short={p_short:.2f})"
    elif ai_side == heur.side and heur.side != "NEUTRAL":
        heur.confidence = min(1.0, heur.confidence + 0.30)
        heur.reason += f"; AI confirms {ai_side} (p_long={p_long:.2f}, p_short={p_short:.2f})"
    else:
        heur.reason += f"; AI suggests {ai_side} (p_long={p_long:.2f}, p_short={p_short:.2f})"

    # Пересчёт TP/SL после финального side (чтобы порядок был корректный)
    last = df.iloc[-1]
    price = float(last["close"])
    atr_val = float(last.get("atr14", np.nan))
    swing_hi, swing_lo = swing_levels(df, lookback=14)

    if heur.side == "LONG":
        sl = max(price - atr_val * 1.2, (swing_lo or price - atr_val * 1.2))
        tps = [price + atr_val * 1.0, price + atr_val * 1.5, min(price + atr_val * 2.0, (swing_hi or price + atr_val * 2.0))]
        tps = sorted([t for t in tps if t is not None])
        heur.tp1, heur.tp2, heur.tp3 = (tps + [None, None, None])[:3]
        heur.sl = sl
    elif heur.side == "SHORT":
        sl = min(price + atr_val * 1.2, (swing_hi or price + atr_val * 1.2))
        tps = [price - atr_val * 1.0, price - atr_val * 1.5, max(price - atr_val * 2.0, (swing_lo or price - atr_val * 2.0))]
        tps = sorted([t for t in tps if t is not None], reverse=True)  # убывание чисел для шорта
        heur.tp1, heur.tp2, heur.tp3 = (tps + [None, None, None])[:3]
        heur.sl = sl

    return heur


# ---------- Symbol normalization / resolve ----------

KNOWN_QUOTES = ["USDT", "USD", "USDC", "BTC", "ETH", "BUSD", "FDUSD", "EUR", "TRY", "RUB"]

def normalize_symbol(user_input: str) -> str:
    """
    Убираем точки/дефисы/подчеркивания, приводим к верхнему регистру.
    Если нет '/', пытаемся добавить '/USDT'.
    Также лечим популярные окончания (PERP, SWAP).
    """
    s = user_input.strip().upper()
    s = re.sub(r"\.[A-Z0-9]+$", "", s)        # обрезаем .P/.U и т.п. хвост
    s = s.replace("-", "/").replace("_", "/").replace(".", "/")
    if "/" not in s:
        # пробуем распарсить BASE + QUOTE
        base, quote = split_base_quote_guess(s)
        if quote:
            s = f"{base}/{quote}"
        else:
            s = f"{s}/USDT"
    s = re.sub(r"(PERP|SWAP)$", "", s)
    return s

def split_base_quote_guess(raw: str) -> Tuple[str, Optional[str]]:
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    s = s.replace("USDU", "USD")  # лечим ETHUSDU
    for q in sorted(KNOWN_QUOTES, key=len, reverse=True):
        if s.endswith(q):
            return s[:-len(q)], q
    for q in sorted(KNOWN_QUOTES, key=len, reverse=True):
        pos = s.rfind(q)
        if pos > 0:
            return s[:pos], q
    return s, None

def resolve_symbol(user_input: str, markets: Dict[str, dict]) -> Optional[str]:
    """
    Пытаемся найти лучший символ из markets под пользовательский ввод.
    """
    if not markets:
        return None

    cleaned = normalize_symbol(user_input)
    if cleaned in markets:
        return cleaned

    # пробуем с подменой USD <-> USDT
    if "/USD" in cleaned and cleaned.replace("/USD", "/USDT") in markets:
        return cleaned.replace("/USD", "/USDT")
    if "/USDT" in cleaned and cleaned.replace("/USDT", "/USD") in markets:
        return cleaned.replace("/USDT", "/USD")

    # более мягкий поиск по BASE
    base = cleaned.split("/")[0]
    cand = [m for m in markets.keys() if m.startswith(base + "/")]
    return cand[0] if cand else None


# ---------- Worker Thread ----------

class FetchWorker(QThread):
    finished = pyqtSignal(pd.DataFrame, SignalResult, str)

    def __init__(self, exchange: ccxt.Exchange, symbol: str, timeframe: str):
        super().__init__()
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

    def run(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=500)
            if not ohlcv:
                raise RuntimeError("Пустые OHLCV")
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
            # индикаторы
            df["ema50"] = ema(df["close"], 50)
            df["ema200"] = ema(df["close"], 200)
            df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
            df["rsi14"] = rsi(df["close"], 14)
            df["atr14"] = atr(df, 14)
            df["obv"] = obv(df["close"], df["volume"])

            sig = combined_signal(df)
            self.finished.emit(df, sig, self.symbol)
        except Exception as e:
            # В случае ошибки — пустой результат, но с reason
            df = pd.DataFrame()
            err_sig = SignalResult(
                side="NEUTRAL", confidence=0.0,
                reason=f"ERROR: {e}", tp1=None, tp2=None, tp3=None, sl=None
            )
            self.finished.emit(df, err_sig, self.symbol)


# ---------- Main Window ----------

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Crypto TA + AI ({qt_lib})")
        self.setGeometry(200, 200, 760, 520)

        self.markets: Dict[str, dict] = {}
        self.current_exchange: Optional[ccxt.Exchange] = None

        vbox = QVBoxLayout(self)

        # --- Controls ---
        controls = QHBoxLayout()
        self.cmb_exchange = QComboBox(); self.cmb_exchange.addItems(SUPPORTED_EXCHANGES)
        self.cmb_tf = QComboBox(); self.cmb_tf.addItems(["1m","5m","10m","15m","30m","1h","4h","1d"])
        self.cmb_market_type = QComboBox(); self.cmb_market_type.addItems(["spot","swap"])
        self.ed_symbol = QLineEdit(); self.ed_symbol.setPlaceholderText("Введите символ (e.g., BTCUSDT, ETH-USD, XAUUSD)")
        self.btn_load = QPushButton("Загрузить рынки")
        self.btn_fetch = QPushButton("Сканировать (TA+AI)")

        controls.addWidget(QLabel("Биржа:")); controls.addWidget(self.cmb_exchange)
        controls.addWidget(QLabel("TF:")); controls.addWidget(self.cmb_tf)
        controls.addWidget(QLabel("Тип:")); controls.addWidget(self.cmb_market_type)
        controls.addWidget(self.ed_symbol)
        controls.addWidget(self.btn_load); controls.addWidget(self.btn_fetch)
        vbox.addLayout(controls)

        # --- Signal box ---
        box = QGroupBox("Результат сигнала"); grid = QGridLayout(box)
        self.lbl_resolved = QLabel("—")
        self.lbl_side = QLabel("—")
        self.lbl_conf = QLabel("—")
        self.lbl_tp1 = QLabel("—")
        self.lbl_tp2 = QLabel("—")
        self.lbl_tp3 = QLabel("—")
        self.lbl_sl = QLabel("—")
        self.lbl_reason = QTextEdit(); self.lbl_reason.setReadOnly(True)

        grid.addWidget(QLabel("Символ:"), 0, 0); grid.addWidget(self.lbl_resolved, 0, 1)
        grid.addWidget(QLabel("Сторона:"), 1, 0); grid.addWidget(self.lbl_side, 1, 1)
        grid.addWidget(QLabel("Уверенность:"), 2, 0); grid.addWidget(self.lbl_conf, 2, 1)
        grid.addWidget(QLabel("TP1:"), 3, 0); grid.addWidget(self.lbl_tp1, 3, 1)
        grid.addWidget(QLabel("TP2:"), 4, 0); grid.addWidget(self.lbl_tp2, 4, 1)
        grid.addWidget(QLabel("TP3:"), 5, 0); grid.addWidget(self.lbl_tp3, 5, 1)
        grid.addWidget(QLabel("SL:"), 6, 0); grid.addWidget(self.lbl_sl, 6, 1)
        grid.addWidget(QLabel("Обоснование:"), 7, 0); grid.addWidget(self.lbl_reason, 7, 1, 3, 1)

        vbox.addWidget(box)

        # --- Events ---
        self.btn_load.clicked.connect(self.load_markets)
        self.btn_fetch.clicked.connect(self.fetch_signal)

    def load_markets(self):
        exch_name = self.cmb_exchange.currentText()
        mtype = self.cmb_market_type.currentText()
        try:
            self.current_exchange = create_exchange(exch_name, mtype)
            self.current_exchange.load_markets()
            self.markets = self.current_exchange.markets
            self.lbl_reason.setPlainText(f"✅ Загружено рынков: {len(self.markets)} ({exch_name}, {mtype})")
        except Exception as e:
            self.lbl_reason.setPlainText(f"Ошибка загрузки рынков: {e}")

    def fetch_signal(self):
        if not self.current_exchange:
            self.lbl_reason.setPlainText("Сначала нажмите 'Загрузить рынки'.")
            return
        sym = self.ed_symbol.text().strip()
        if not sym:
            self.lbl_reason.setPlainText("Введите символ (например, BTCUSDT).")
            return

        resolved = resolve_symbol(sym, self.markets)
        if not resolved:
            self.lbl_resolved.setText("—")
            self.lbl_reason.setPlainText(f"Не найден символ для ввода: {sym}")
            return

        tf = self.cmb_tf.currentText()
        self.lbl_resolved.setText(resolved)
        self.lbl_reason.setPlainText(f"Загружаю {resolved} {tf} ...")

        self.worker = FetchWorker(self.current_exchange, resolved, tf)
        self.worker.finished.connect(self.on_fetched)
        self.worker.start()

    def on_fetched(self, df: pd.DataFrame, sig: SignalResult, resolved_symbol: str):
        self.lbl_resolved.setText(resolved_symbol or "—")
        self.lbl_side.setText(sig.side)
        self.lbl_conf.setText(f"{sig.confidence:.2f}")
        self.lbl_tp1.setText(f"{sig.tp1:.4f}" if sig.tp1 else "—")
        self.lbl_tp2.setText(f"{sig.tp2:.4f}" if sig.tp2 else "—")
        self.lbl_tp3.setText(f"{sig.tp3:.4f}" if sig.tp3 else "—")
        self.lbl_sl.setText(f"{sig.sl:.4f}" if sig.sl else "—")
        self.lbl_reason.setPlainText(sig.reason)


# ---------- Entry Point ----------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    if qt_lib == "PyQt5":
        sys.exit(app.exec_())
    else:
        sys.exit(app.exec())
