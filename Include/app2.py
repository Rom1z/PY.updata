"""
Crypto TA Desktop App — fixed symbol resolver
--------------------------------------
Новое:
1) Dropdown 'Тип рынка' (spot/swap) для корректной загрузки рынков.
2) Кнопка 'Загрузить рынки' — подтягивает markets с биржи и заполняет список символов.
3) Авто-резолвер символов из пользовательского ввода:
   - убирает .P/.U/точки/дефисы,
   - добавляет или убирает слэш,
   - лечит опечатки типа ETHUSDU -> ETH/USD,
   - подменяет USD <-> USDT, если нужно,
   - ищет лучший матч в ex.load_markets() и логирует, во что преобразовал.

Поддержка твоих записей:
- 1000PEPEUSDT.P
- XAUUSD
- ETHUSDU
- HYPERUSDT.P
и любых других, если они есть на выбранной бирже/типе рынка.

Не является инвестсоветом.
"""

from __future__ import annotations

import sys, io, math, traceback, re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QSpinBox, QDoubleSpinBox, QTabWidget,
    QFormLayout, QProgressBar, QMessageBox, QTextEdit, QGroupBox, QSplitter
)

# Optional deps
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SUPPORTED_EXCHANGES = {
    "bybit": {"timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]},
    "binance": {"timeframes": ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]},
    "okx": {"timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]},
}

KNOWN_QUOTES = ["USDT", "USD", "USDC", "BTC", "ETH", "BUSD", "FDUSD", "EUR", "TRY", "RUB"]


@dataclass
class SignalResult:
    side: str
    confidence: float
    reason: str
    tp: Optional[float]
    sl: Optional[float]


# ---------- Helpers: exchange + symbol resolving ----------

def create_exchange(exchange_name: str, market_type: str):
    if ccxt is None:
        raise RuntimeError("ccxt не установлен. Установите: pip install ccxt")
    if exchange_name not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Биржа '{exchange_name}' не поддерживается.")
    ex_class = getattr(ccxt, exchange_name)
    # 'defaultType': 'spot' | 'swap' (для bybit/okx/binance актуально в ccxt unified)
    opts = {"enableRateLimit": True, "options": {"defaultType": market_type}}
    ex = ex_class(opts)
    return ex


def normalize_symbol(s: str) -> str:
    """Uppercase + только буквы/цифры (без слешей, точек и т.д.)."""
    return "".join(ch for ch in s.upper() if ch.isalnum())


def split_base_quote_guess(raw: str) -> Tuple[str, Optional[str]]:
    """
    Пытаемся угадать BASE/QUOTE из строк без слеша.
    Возвращает (base, quote | None).
    """
    s = normalize_symbol(raw)
    # лечим популярные артефакты
    s = s.replace("USDU", "USD")  # ETHUSDU -> ETHUSD
    # найдем известный квот в конце или внутри
    for q in sorted(KNOWN_QUOTES, key=len, reverse=True):
        if s.endswith(q):
            return s[:-len(q)], q
    # если не нашли на конце — ищем последнюю позицию любого квота
    for q in sorted(KNOWN_QUOTES, key=len, reverse=True):
        pos = s.rfind(q)
        if pos > 0:
            return s[:pos], q
    return s, None


def best_symbol_match(user_input: str, markets: Dict[str, dict]) -> Tuple[Optional[str], List[str]]:
    """
    Находит лучший доступный символ из markets под пользовательский ввод.
    Возвращает (лучший_символ | None, список_кандидатов_для_подсказки)
    """
    if not markets:
        return None, []

    cleaned = user_input.strip().upper()
    # Убираем .P, .U и прочие хвосты вида .XYZ
    cleaned = re.sub(r"\.[A-Z0-9]+$", "", cleaned)
    # Заменим дефисы и точки на слэш, если он отсутствует
    cleaned = cleaned.replace("-", "/").replace(".", "/")
    # Быстрый путь: точное совпадение ключа рынка
    if cleaned in markets:
        return cleaned, []

    # Если пользователь ввёл без слеша — попробуем угадать BASE/QUOTE
    if "/" not in cleaned:
        base, quote = split_base_quote_guess(cleaned)  # ETHUSDU -> ('ETH', 'USD')
        # Попробуем набор вариантов:
        candidates_try = []
        if quote:
            candidates_try += [f"{base}/{quote}"]
            # подмена USD<->USDT
            if quote == "USD":
                candidates_try.append(f"{base}/USDT")
            if quote == "USDT":
                candidates_try.append(f"{base}/USD")
        else:
            # если квот неизвестен — попробуем самые частые
            candidates_try += [f"{base}/USDT", f"{base}/USD"]

        # Плюс деривативные формы (без слеша)
        if quote:
            candidates_try += [f"{base}{quote}", f"{base}{quote}"]
            if quote == "USD":
                candidates_try.append(f"{base}USDT")
            if quote == "USDT":
                candidates_try.append(f"{base}USD")

        for c in candidates_try:
            if c in markets:
                return c, []

    # Фазза: нормализуем и ищем в markets по подстрокам BASE & QUOTE
    n_user = normalize_symbol(cleaned)
    base, quote = split_base_quote_guess(cleaned)
    n_base = normalize_symbol(base)
    n_quote = normalize_symbol(quote or "")

    scored: List[Tuple[int, str]] = []  # (score, symbol)
    for m in markets.keys():
        n_m = normalize_symbol(m)
        score = 0
        if n_m == n_user:
            score += 100
        if n_base and n_base in n_m:
            score += 10
        if n_quote and n_quote in n_m:
            score += 5
        # подмена USD<->USDT даёт +3
        if n_quote in ("USD", "USDT"):
            if ("USD" in n_m and "USDT" in n_user) or ("USDT" in n_m and "USD" in n_user):
                score += 3
        # небольшой бонус, если формат со слэшем совпал
        if ("/" in m) == ("/" in cleaned):
            score += 1
        if score > 0:
            scored.append((score, m))

    scored.sort(reverse=True)
    if scored:
        best = scored[0][1]
        suggestions = [m for _, m in scored[:10]]
        return best, suggestions

    return None, []


# ---------- Data / TA ----------

def fetch_ohlcv(exchange_name: str, market_type: str, user_symbol: str,
                timeframe: str, limit: int, log: Optional[QTextEdit] = None) -> Tuple[pd.DataFrame, str]:
    ex = create_exchange(exchange_name, market_type)
    markets = ex.load_markets()
    # Попробуем резолв
    symbol, suggestions = best_symbol_match(user_symbol, markets)

    if symbol is None:
        hint = "\n".join(suggestions) if suggestions else "совпадений не найдено"
        raise RuntimeError(
            f"Не удалось найти символ '{user_symbol}' на {exchange_name} ({market_type}).\nПодсказки:\n{hint}"
        )

    if log is not None and symbol != user_symbol:
        log.append(f"🔎 Символ '{user_symbol}' преобразован → <b>{symbol}</b>")

    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not data:
        raise RuntimeError(f"Пустые OHLCV для {symbol}. Проверь таймфрейм/лимит/тип рынка.")
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df, symbol


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).ewm(span=14, adjust=False).mean()
    roll_down = pd.Series(down, index=df.index).ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=14, adjust=False).mean()

    df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df


def swing_levels(df: pd.DataFrame, lookback: int = 14) -> Tuple[Optional[float], Optional[float]]:
    lows = df["low"].rolling(window=lookback).min()
    highs = df["high"].rolling(window=lookback).max()
    return float(highs.iloc[-1]) if not math.isnan(highs.iloc[-1]) else None, \
           float(lows.iloc[-1]) if not math.isnan(lows.iloc[-1]) else None


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
    if macd_cross_up: score += 0.3; reasons.append("MACD bullish cross")
    if macd_cross_dn: score -= 0.3; reasons.append("MACD bearish cross")

    # Momentum
    hist_slope = (last["macd"] - last["macd_signal"]) - (prev["macd"] - prev["macd_signal"])
    if hist_slope > 0:
        score += 0.15; reasons.append("MACD momentum rising")
    else:
        score -= 0.15; reasons.append("MACD momentum falling")

    # RSI regime
    if last["rsi14"] > 55:
        score += 0.2; reasons.append("RSI > 55")
    elif last["rsi14"] < 45:
        score -= 0.2; reasons.append("RSI < 45")

    score = max(-1.0, min(1.0, score))
    side = "LONG" if score > 0.15 else ("SHORT" if score < -0.15 else "NEUTRAL")
    confidence = float(abs(score))

    atr = float(last.get("atr14", np.nan))
    price = float(last["close"])
    swing_hi, swing_lo = swing_levels(df, lookback=14)
    tp = sl = None
    if side == "LONG":
        sl = max(price - atr * 1.2, swing_lo or price - atr * 1.2)
        tp = min(price + atr * 2.0, swing_hi or price + atr * 2.0)
    elif side == "SHORT":
        sl = min(price + atr * 1.2, swing_hi or price + atr * 1.2)
        tp = max(price - atr * 2.0, swing_lo or price - atr * 2.0)

    return SignalResult(side=side, confidence=confidence, reason="; ".join(reasons), tp=tp, sl=sl)


def fig_to_qpixmap(fig) -> QPixmap:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.read(), 'PNG')
    return pixmap


def plot_chart(df: pd.DataFrame, title: str = "") -> QPixmap:
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["close"], label="Close")
    ax.plot(df.index, df["ema50"], label="EMA50")
    ax.plot(df.index, df["ema200"], label="EMA200")
    ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("Time"); ax.set_ylabel("Price")
    fig.autofmt_xdate()
    pm = fig_to_qpixmap(fig)
    plt.close(fig)
    return pm


def analyze_image_trendlines(path: str) -> Tuple[Optional[QPixmap], str]:
    if cv2 is None:
        return None, "OpenCV не установлен. Установите opencv-python для анализа изображений."
    img = cv2.imread(path)
    if img is None:
        return None, "Не удалось загрузить изображение."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=80, maxLineGap=10)

    overlay, count = img.copy(), 0
    if lines is not None:
        for l in lines[:50]:
            x1, y1, x2, y2 = l[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count += 1
    blended = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    is_success, buffer = cv2.imencode(".png", blended)
    if not is_success:
        return None, "Ошибка кодирования изображения."
    pm = QPixmap(); pm.loadFromData(buffer.tobytes(), 'PNG')
    note = f"Найдено линий: {count}. Это лишь подсказки — проверяйте вручную."
    return pm, note


# ---------- Threads ----------

class FetchThread(QThread):
    progress = Signal(int)
    finished_df = Signal(object, str, str)  # df, status, resolved_symbol

    def __init__(self, exchange_name: str, market_type: str, symbol: str, timeframe: str, limit: int, log: Optional[QTextEdit]):
        super().__init__()
        self.exchange_name = exchange_name
        self.market_type = market_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.log = log

    def run(self):
        try:
            self.progress.emit(10)
            df, resolved = fetch_ohlcv(self.exchange_name, self.market_type, self.symbol, self.timeframe, self.limit, self.log)
            self.progress.emit(60)
            df = compute_indicators(df)
            self.progress.emit(100)
            self.finished_df.emit(df, "OK", resolved)
        except Exception as e:
            self.finished_df.emit(None, f"ERROR: {e}\n{traceback.format_exc()}", "")


# ---------- UI ----------

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto TA Desktop — heuristic AI")
        self.resize(1280, 860)

        self.tabs = QTabWidget()
        self.tab_market = QWidget()
        self.tab_image = QWidget()
        self.tab_log = QWidget()

        self.tabs.addTab(self.tab_market, "Рынок (OHLCV)")
        self.tabs.addTab(self.tab_image, "Изображение")
        self.tabs.addTab(self.tab_log, "Лог")

        root = QVBoxLayout(self)
        root.addWidget(self.tabs)

        self._build_market_tab()
        self._build_image_tab()
        self._build_log_tab()

        self.current_df: Optional[pd.DataFrame] = None
        self.current_image_path: Optional[str] = None
        self.loaded_markets: Dict[str, dict] = {}

    def _build_market_tab(self):
        main = QVBoxLayout(self.tab_market)

        form = QFormLayout()

        self.cb_exchange = QComboBox(); self.cb_exchange.addItems(list(SUPPORTED_EXCHANGES))
        self.cb_market_type = QComboBox(); self.cb_market_type.addItems(["spot", "swap"])

        # Символ: editable ComboBox (с предзаполнением твоих вариантов)
        self.cb_symbol = QComboBox(); self.cb_symbol.setEditable(True)
        self.cb_symbol.addItems([
            "BTC/USDT",
            "ETH/USDT",
            "1000PEPEUSDT.P",
            "HYPERUSDT.P",
            "XAUUSD",
            "ETHUSDU"
        ])

        self.cb_timeframe = QComboBox(); self.cb_timeframe.addItems(["1m","5m","15m","30m","1h","4h","1d"])
        self.sb_limit = QSpinBox(); self.sb_limit.setRange(100, 10000); self.sb_limit.setValue(500)

        form.addRow("Биржа:", self.cb_exchange)
        form.addRow("Тип рынка:", self.cb_market_type)
        form.addRow("Пара:", self.cb_symbol)
        form.addRow("Таймфрейм:", self.cb_timeframe)
        form.addRow("Свечей (limit):", self.sb_limit)

        # Кнопки
        btns = QHBoxLayout()
        self.btn_load_markets = QPushButton("Загрузить рынки")
        self.btn_load_markets.clicked.connect(self.on_load_markets)
        self.btn_fetch = QPushButton("Загрузить данные")
        self.btn_fetch.clicked.connect(self.on_fetch)
        self.pb = QProgressBar(); self.pb.setValue(0)
        btns.addWidget(self.btn_load_markets)
        btns.addWidget(self.btn_fetch)
        btns.addWidget(self.pb)

        # Чарт + Сигнал
        split = QSplitter(Qt.Vertical)
        top = QWidget(); top_l = QHBoxLayout(top)
        self.lbl_chart = QLabel("График появится здесь"); self.lbl_chart.setAlignment(Qt.AlignCenter)
        top_l.addWidget(self.lbl_chart)

        bottom = QWidget(); bottom_l = QHBoxLayout(bottom)
        self.group_signal = QGroupBox("Сигнал и риск-менеджмент")
        gl = QFormLayout(self.group_signal)
        self.lbl_resolved = QLabel("—")  # показывает, во что резолвнулся реальный символ
        self.lbl_side = QLabel("—")
        self.lbl_conf = QLabel("—")
        self.lbl_reason = QTextEdit(); self.lbl_reason.setReadOnly(True)
        self.dsb_atr_tp_mult = QDoubleSpinBox(); self.dsb_atr_tp_mult.setRange(0.5, 10.0); self.dsb_atr_tp_mult.setSingleStep(0.1); self.dsb_atr_tp_mult.setValue(2.0)
        self.dsb_atr_sl_mult = QDoubleSpinBox(); self.dsb_atr_sl_mult.setRange(0.2, 5.0); self.dsb_atr_sl_mult.setSingleStep(0.1); self.dsb_atr_sl_mult.setValue(1.2)
        self.lbl_tp = QLabel("—")
        self.lbl_sl = QLabel("—")
        self.btn_recalc_risk = QPushButton("Пересчитать TP/SL")
        self.btn_recalc_risk.clicked.connect(self.recalc_tp_sl)

        gl.addRow("Реальный символ:", self.lbl_resolved)
        gl.addRow("Сторона:", self.lbl_side)
        gl.addRow("Уверенность:", self.lbl_conf)
        gl.addRow("Обоснование:", self.lbl_reason)
        gl.addRow("ATR TP×:", self.dsb_atr_tp_mult)
        gl.addRow("ATR SL×:", self.dsb_atr_sl_mult)
        gl.addRow("TP:", self.lbl_tp)
        gl.addRow("SL:", self.lbl_sl)
        gl.addRow(self.btn_recalc_risk)

        bottom_l.addWidget(self.group_signal)
        split.addWidget(top); split.addWidget(bottom)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 1)

        main.addLayout(form)
        main.addLayout(btns)
        main.addWidget(split)

    def _build_image_tab(self):
        main = QVBoxLayout(self.tab_image)
        row = QHBoxLayout()
        self.btn_open_img = QPushButton("Открыть изображение...")
        self.btn_open_img.clicked.connect(self.on_open_image)
        self.btn_analyze_img = QPushButton("Анализ линий")
        self.btn_analyze_img.clicked.connect(self.on_analyze_image)
        row.addWidget(self.btn_open_img); row.addWidget(self.btn_analyze_img); row.addStretch(1)
        self.lbl_img = QLabel("Здесь будет изображение"); self.lbl_img.setAlignment(Qt.AlignCenter)
        self.txt_img_note = QTextEdit(); self.txt_img_note.setReadOnly(True)
        main.addLayout(row); main.addWidget(self.lbl_img); main.addWidget(self.txt_img_note)

    def _build_log_tab(self):
        lay = QVBoxLayout(self.tab_log)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)

    # ===== Handlers =====

    def on_load_markets(self):
        ex_name = self.cb_exchange.currentText()
        mtype = self.cb_market_type.currentText()
        try:
            ex = create_exchange(ex_name, mtype)
            self.loaded_markets = ex.load_markets()
            self.log.append(f"✅ Загружено рынков: {len(self.loaded_markets)} ({ex_name}, {mtype})")
            # Заполним список символов несколькими популярными и найденными по ключевым словам:
            self.cb_symbol.clear()
            # Базовые пресеты (как ты просил):
            presets = ["BTC/USDT", "ETH/USDT", "1000PEPEUSDT.P", "HYPERUSDT.P", "XAUUSD", "ETHUSDU"]
            for p in presets:
                self.cb_symbol.addItem(p)

            # Добавим реальные совпадения по ключевым словам, если есть
            def add_if_contains(substrs: List[str], limit=10):
                cnt = 0
                for s in self.loaded_markets.keys():
                    up = s.upper()
                    if all(sub in up for sub in substrs):
                        self.cb_symbol.addItem(s)
                        cnt += 1
                        if cnt >= limit:
                            break

            add_if_contains(["1000PEPE"])
            add_if_contains(["HYPER"])   # если вдруг есть с таким названием
            add_if_contains(["XAU"])
            self.log.append("ℹ️ Список символов обновлён. Можно вводить вручную — автоисправление сработает при загрузке данных.")

        except Exception as e:
            msg = f"Ошибка загрузки рынков: {e}\n{traceback.format_exc()}"
            self.log.append(f"<span style='color:red'>{msg}</span>")
            QMessageBox.critical(self, "Ошибка", str(e))

    def on_fetch(self):
        ex = self.cb_exchange.currentText()
        mtype = self.cb_market_type.currentText()
        sym = self.cb_symbol.currentText().strip()
        tf = self.cb_timeframe.currentText()
        limit = int(self.sb_limit.value())
        self.pb.setValue(0)
        self.log.append(f"Fetching {ex} [{mtype}] {sym} {tf} limit={limit} ...")
        self.thread = FetchThread(ex, mtype, sym, tf, limit, self.log)
        self.thread.progress.connect(self.pb.setValue)
        self.thread.finished_df.connect(self.on_fetched)
        self.thread.start()

    def on_fetched(self, df: Optional[pd.DataFrame], status: str, resolved_symbol: str):
        if df is None:
            self.log.append(f"<span style='color:red'>{status}</span>")
            QMessageBox.critical(self, "Ошибка", status)
            return
        self.log.append("OK. Считаю сигнал...")
        self.current_df = df
        self.lbl_chart.setPixmap(plot_chart(df, title="Close + EMA50/EMA200"))
        sig = heuristic_signal(df)
        self.lbl_resolved.setText(resolved_symbol or "—")
        self.lbl_side.setText(sig.side)
        self.lbl_conf.setText(f"{sig.confidence:.2f}")
        self.lbl_reason.setPlainText(sig.reason)
        self.lbl_tp.setText(f"{sig.tp:.2f}" if sig.tp else "—")
        self.lbl_sl.setText(f"{sig.sl:.2f}" if sig.sl else "—")
        self.log.append(f"Сигнал: {sig.side} (conf={sig.confidence:.2f}) | TP={sig.tp} SL={sig.sl}")

    def recalc_tp_sl(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные на вкладке 'Рынок'.")
            return
        df = self.current_df
        sig = heuristic_signal(df)
        last = df.iloc[-1]
        price = float(last["close"])
        atr = float(last.get("atr14", np.nan))
        if math.isnan(atr) or atr <= 0:
            QMessageBox.warning(self, "ATR недоступен", "Не удалось вычислить ATR.")
            return
        tp_mult = float(self.dsb_atr_tp_mult.value())
        sl_mult = float(self.dsb_atr_sl_mult.value())
        swing_hi, swing_lo = swing_levels(df, lookback=14)
        tp = sl = None
        if sig.side == "LONG":
            sl = max(price - atr * sl_mult, swing_lo or price - atr * sl_mult)
            tp = min(price + atr * tp_mult, swing_hi or price + atr * tp_mult)
        elif sig.side == "SHORT":
            sl = min(price + atr * sl_mult, swing_hi or price + atr * sl_mult)
            tp = max(price - atr * tp_mult, swing_lo or price - atr * tp_mult)
        self.lbl_tp.setText(f"{tp:.2f}" if tp else "—")
        self.lbl_sl.setText(f"{sl:.2f}" if sl else "—")
        self.log.append(f"Пересчитано: TP={tp} SL={sl} (TP×={tp_mult}, SL×={sl_mult})")

    def on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", filter="Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self.current_image_path = path
        pm = QPixmap(path)
        self.lbl_img.setPixmap(pm.scaled(self.lbl_img.size()*0.98, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.txt_img_note.setPlainText("Изображение загружено. Нажмите 'Анализ линий'.")

    def on_analyze_image(self):
        if not self.current_image_path:
            QMessageBox.information(self, "Нет изображения", "Сначала откройте изображение графика.")
            return
        pm, note = analyze_image_trendlines(self.current_image_path)
        if pm is not None:
            self.lbl_img.setPixmap(pm.scaled(self.lbl_img.size()*0.98, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.txt_img_note.setPlainText(note)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = App()
    ui.show()
    sys.exit(app.exec())
