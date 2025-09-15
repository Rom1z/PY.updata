"""
Crypto TA Desktop App
--------------------------------------
• PySide6 desktop GUI that can:
  1) Fetch OHLCV data for crypto pairs from public exchanges via ccxt.
  2) Compute classic TA (EMA50/200, RSI, MACD, ATR, pivots, volatility).
  3) Produce a heuristic AI-like signal (LONG/SHORT/NEUTRAL) with confidence.
  4) Propose TP/SL using ATR multiples and recent swing levels.
  5) Load a chart IMAGE (PNG/JPG) and run basic edge/line detection to hint trendlines.

⚠️ ВАЖНО: 100% точность невозможна. Этот софт — учебный и не является инвестсоветом.

Install (Windows/macOS/Linux):
--------------------------------------
python -m venv .venv && .venv/Scripts/activate  # Windows
# or: source .venv/bin/activate                 # macOS/Linux
pip install PySide6 ccxt pandas numpy pandas_ta matplotlib opencv-python scikit-learn

Run:
--------------------------------------
python main.py

Notes:
- Public endpoints are used; no API keys needed for "bybit", "binance", etc., for OHLCV.
- If an exchange/timeframe is unsupported or rate-limited, try another one.
- Image analysis is basic (Canny + Hough). It only suggests potential lines; verify manually.
"""
from __future__ import annotations

import sys
import io
import math
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QSpinBox, QDoubleSpinBox, QTabWidget,
    QFormLayout, QProgressBar, QMessageBox, QTextEdit, QGroupBox, QSplitter
)

# Optional imports guarded
try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None

try:
    import pandas_ta as ta  # type: ignore
except Exception:
    ta = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
except Exception:
    RandomForestClassifier = None

import matplotlib
matplotlib.use('Agg')  # Offscreen backend; we save plots as PNG and display in QLabel
import matplotlib.pyplot as plt


SUPPORTED_EXCHANGES = {
    "bybit": {"timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]},
    "binance": {"timeframes": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]},
    "okx": {"timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]},
}


@dataclass
class SignalResult:
    side: str  # "LONG", "SHORT", or "NEUTRAL"
    confidence: float  # 0..1
    reason: str
    tp: Optional[float]
    sl: Optional[float]


def fetch_ohlcv(exchange_name: str, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed. Run: pip install ccxt")
    if exchange_name not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Exchange '{exchange_name}' not in supported list: {list(SUPPORTED_EXCHANGES)}")

    ex_class = getattr(ccxt, exchange_name)
    ex = ex_class({"enableRateLimit": True})

    if timeframe not in ex.timeframes if hasattr(ex, 'timeframes') and ex.timeframes else SUPPORTED_EXCHANGES[exchange_name]["timeframes"]:
        # Some ccxt drivers expose timeframes; otherwise fall back to our list
        pass  # we'll try anyway; exchange may still accept it

    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not data:
        raise RuntimeError("Empty OHLCV returned. Check symbol/timeframe.")

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic indicators even without pandas_ta
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).ewm(span=14, adjust=False).mean()
    roll_down = pd.Series(down, index=df.index).ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # ATR (Wilder)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=14, adjust=False).mean()

    # Pivots (simple)
    df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3.0

    return df


def swing_levels(df: pd.DataFrame, lookback: int = 10) -> Tuple[Optional[float], Optional[float]]:
    lows = df["low"].rolling(window=lookback).min()
    highs = df["high"].rolling(window=lookback).max()
    return float(highs.iloc[-1]) if not math.isnan(highs.iloc[-1]) else None, \
           float(lows.iloc[-1]) if not math.isnan(lows.iloc[-1]) else None


def heuristic_signal(df: pd.DataFrame) -> SignalResult:
    """Rule-based ensemble to emulate an AI classifier.
    Combines:
      - EMA50 vs EMA200 trend
      - MACD cross and histogram slope
      - RSI 14 position
    Produces LONG/SHORT/NEUTRAL with confidence 0..1.
    Suggests TP/SL using ATR and swing levels.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0.0
    reasons: List[str] = []

    # Trend bias
    if last["ema50"] > last["ema200"]:
        score += 0.25; reasons.append("EMA50 > EMA200 (uptrend)")
    elif last["ema50"] < last["ema200"]:
        score -= 0.25; reasons.append("EMA50 < EMA200 (downtrend)")

    # MACD cross
    macd_cross_up = prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]
    macd_cross_dn = prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]
    if macd_cross_up:
        score += 0.3; reasons.append("MACD bullish cross")
    if macd_cross_dn:
        score -= 0.3; reasons.append("MACD bearish cross")

    # MACD momentum
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

    # Normalize to [-1,1]
    score = max(-1.0, min(1.0, score))
    side = "LONG" if score > 0.15 else ("SHORT" if score < -0.15 else "NEUTRAL")
    confidence = float(abs(score))

    # TP/SL via ATR & swings
    atr = float(last.get("atr14", np.nan))
    price = float(last["close"])

    swing_hi, swing_lo = swing_levels(df, lookback=14)
    tp = sl = None

    # default ATR multiples
    atr_mult_tp = 2.0
    atr_mult_sl = 1.2

    if side == "LONG":
        sl = max(price - atr * atr_mult_sl, swing_lo or price - atr * atr_mult_sl)
        tp = min(price + atr * atr_mult_tp, swing_hi or price + atr * atr_mult_tp)
    elif side == "SHORT":
        sl = min(price + atr * atr_mult_sl, swing_hi or price + atr * atr_mult_sl)
        tp = max(price - atr * atr_mult_tp, swing_lo or price - atr * atr_mult_tp)

    reason = "; ".join(reasons)
    return SignalResult(side=side, confidence=confidence, reason=reason, tp=tp, sl=sl)


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
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
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

    overlay = img.copy()
    count = 0
    if lines is not None:
        for l in lines[:50]:
            x1, y1, x2, y2 = l[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count += 1
    alpha = 0.7
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # Convert to QPixmap
    is_success, buffer = cv2.imencode(".png", blended)
    if not is_success:
        return None, "Ошибка кодирования изображения."
    pm = QPixmap()
    pm.loadFromData(buffer.tobytes(), 'PNG')
    note = f"Найдено линий: {count}. Эти линии могут намекать на тренды/уровни, но проверяйте вручную."
    return pm, note


class FetchThread(QThread):
    progress = Signal(int)
    finished_df = Signal(object, str)

    def __init__(self, exchange_name: str, symbol: str, timeframe: str, limit: int):
        super().__init__()
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit

    def run(self):
        try:
            self.progress.emit(10)
            df = fetch_ohlcv(self.exchange_name, self.symbol, self.timeframe, self.limit)
            self.progress.emit(60)
            df = compute_indicators(df)
            self.progress.emit(100)
            self.finished_df.emit(df, "OK")
        except Exception as e:
            self.finished_df.emit(None, f"ERROR: {e}\n{traceback.format_exc()}")


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto TA Desktop — heuristic AI")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.tab_market = QWidget()
        self.tab_image = QWidget()
        self.tab_log = QWidget()

        self.tabs.addTab(self.tab_market, "Рынок (OHLCV)")
        self.tabs.addTab(self.tab_image, "Изображение (чертеж)")
        self.tabs.addTab(self.tab_log, "Лог")

        root = QVBoxLayout(self)
        root.addWidget(self._build_menu_bar())
        root.addWidget(self.tabs)

        self._build_market_tab()
        self._build_image_tab()
        self._build_log_tab()

        self.current_df: Optional[pd.DataFrame] = None

    def _build_menu_bar(self) -> QWidget:
        # Simple toolbar-like row
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_about = QPushButton("О программе")
        self.btn_about.clicked.connect(self.show_about)
        layout.addWidget(self.btn_about)

        layout.addStretch(1)
        return bar

    def show_about(self):
        QMessageBox.information(self, "О программе",
                                "Учебное приложение для базового теханализа и эвристического сигнала.\n"
                                "Не является инвестсоветом. 100% точность невозможна.")

    def _build_market_tab(self):
        w = self.tab_market
        main = QVBoxLayout(w)

        form = QFormLayout()
        self.cb_exchange = QComboBox(); self.cb_exchange.addItems(list(SUPPORTED_EXCHANGES))
        self.le_symbol = QLineEdit("BTC/USDT")
        self.cb_timeframe = QComboBox(); self.cb_timeframe.addItems(["1m","5m","15m","30m","1h","4h","1d"])
        self.sb_limit = QSpinBox(); self.sb_limit.setRange(100, 10000); self.sb_limit.setValue(500)

        form.addRow("Биржа:", self.cb_exchange)
        form.addRow("Пара:", self.le_symbol)
        form.addRow("Таймфрейм:", self.cb_timeframe)
        form.addRow("Свечей (limit):", self.sb_limit)

        btn_row = QHBoxLayout()
        self.btn_fetch = QPushButton("Загрузить данные")
        self.btn_fetch.clicked.connect(self.on_fetch)
        self.pb = QProgressBar(); self.pb.setValue(0)
        btn_row.addWidget(self.btn_fetch)
        btn_row.addWidget(self.pb)

        # Chart + Signal area
        split = QSplitter(Qt.Vertical)
        top = QWidget(); top_l = QHBoxLayout(top)
        self.lbl_chart = QLabel("График появится здесь"); self.lbl_chart.setAlignment(Qt.AlignCenter)
        top_l.addWidget(self.lbl_chart)

        bottom = QWidget(); bottom_l = QHBoxLayout(bottom)
        self.group_signal = QGroupBox("Сигнал и риск-менеджмент")
        gl = QFormLayout(self.group_signal)
        self.lbl_side = QLabel("—")
        self.lbl_conf = QLabel("—")
        self.lbl_reason = QTextEdit(); self.lbl_reason.setReadOnly(True)
        self.dsb_atr_tp_mult = QDoubleSpinBox(); self.dsb_atr_tp_mult.setRange(0.5, 10.0); self.dsb_atr_tp_mult.setSingleStep(0.1); self.dsb_atr_tp_mult.setValue(2.0)
        self.dsb_atr_sl_mult = QDoubleSpinBox(); self.dsb_atr_sl_mult.setRange(0.2, 5.0); self.dsb_atr_sl_mult.setSingleStep(0.1); self.dsb_atr_sl_mult.setValue(1.2)
        self.lbl_tp = QLabel("—")
        self.lbl_sl = QLabel("—")

        gl.addRow("Сторона:", self.lbl_side)
        gl.addRow("Уверенность:", self.lbl_conf)
        gl.addRow("Обоснование:", self.lbl_reason)
        gl.addRow("ATR TP×:", self.dsb_atr_tp_mult)
        gl.addRow("ATR SL×:", self.dsb_atr_sl_mult)
        gl.addRow("TP:", self.lbl_tp)
        gl.addRow("SL:", self.lbl_sl)

        self.btn_recalc_risk = QPushButton("Пересчитать TP/SL")
        self.btn_recalc_risk.clicked.connect(self.recalc_tp_sl)
        gl.addRow(self.btn_recalc_risk)

        bottom_l.addWidget(self.group_signal)

        split.addWidget(top)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        main.addLayout(form)
        main.addLayout(btn_row)
        main.addWidget(split)

    def _build_image_tab(self):
        w = self.tab_image
        main = QVBoxLayout(w)

        row = QHBoxLayout()
        self.btn_open_img = QPushButton("Открыть изображение...")
        self.btn_open_img.clicked.connect(self.on_open_image)
        self.btn_analyze_img = QPushButton("Анализ линий")
        self.btn_analyze_img.clicked.connect(self.on_analyze_image)
        row.addWidget(self.btn_open_img)
        row.addWidget(self.btn_analyze_img)
        row.addStretch(1)

        self.lbl_img = QLabel("Здесь будет изображение"); self.lbl_img.setAlignment(Qt.AlignCenter)
        self.txt_img_note = QTextEdit(); self.txt_img_note.setReadOnly(True)

        main.addLayout(row)
        main.addWidget(self.lbl_img)
        main.addWidget(self.txt_img_note)

        self.current_image_path: Optional[str] = None

    def _build_log_tab(self):
        w = self.tab_log
        lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)

    # ===== Handlers =====
    def on_fetch(self):
        ex = self.cb_exchange.currentText()
        sym = self.le_symbol.text().strip()
        tf = self.cb_timeframe.currentText()
        limit = int(self.sb_limit.value())
        self.pb.setValue(0)
        self.log.append(f"Fetching {ex} {sym} {tf} limit={limit}...")
        self.thread = FetchThread(ex, sym, tf, limit)
        self.thread.progress.connect(self.pb.setValue)
        self.thread.finished_df.connect(self.on_fetched)
        self.thread.start()

    def on_fetched(self, df: Optional[pd.DataFrame], status: str):
        if df is None:
            self.log.append(f"<span style='color:red'>{status}</span>")
            QMessageBox.critical(self, "Ошибка", status)
            return
        self.log.append("Данные получены. Рассчитываю сигнал...")
        self.current_df = df
        pm = plot_chart(df, title="Close + EMA50/EMA200")
        self.lbl_chart.setPixmap(pm)

        sig = heuristic_signal(df)
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
        # override ATR multiples from UI
        last = df.iloc[-1]
        price = float(last["close"])  # baseline
        atr = float(last["atr14"]) if not math.isnan(last["atr14"]) else None
        if atr is None or atr <= 0:
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
