import sys
import ccxt
import pandas as pd
import numpy as np
import talib
import mplfinance as mpf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib.pyplot as plt

# === Класс для получения данных ===
class DataFetcher(QThread):
    data_fetched = pyqtSignal(pd.DataFrame, str)

    def __init__(self, symbol, timeframe):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.bybit()

    def run(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.data_fetched.emit(df, self.symbol)
        except Exception as e:
            print("Ошибка:", e)

# === Основное приложение ===
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Трейдинг Ассистент (60 лет опыта)")
        self.setGeometry(200, 200, 1000, 800)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Выберите криптовалюту и таймфрейм:")
        self.layout.addWidget(self.label)

        self.symbol_box = QComboBox()
        self.symbol_box.addItems(["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","ADA/USDT"])
        self.layout.addWidget(self.symbol_box)

        self.timeframe_box = QComboBox()
        self.timeframe_box.addItems(["1d","4h","1h","15m","5m"])
        self.layout.addWidget(self.timeframe_box)

        self.btn_load = QPushButton("Загрузить данные")
        self.btn_load.clicked.connect(self.load_data)
        self.layout.addWidget(self.btn_load)

        self.signal_label = QLabel("Сигнал: ---")
        self.layout.addWidget(self.signal_label)

    def load_data(self):
        symbol = self.symbol_box.currentText()
        timeframe = self.timeframe_box.currentText()
        self.fetcher = DataFetcher(symbol, timeframe)
        self.fetcher.data_fetched.connect(self.on_fetched)
        self.fetcher.start()

    def on_fetched(self, df, symbol):
        # === Индикаторы ===
        df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
        df['SMA50'] = talib.SMA(df['close'], timeperiod=50)
        df['EMA14'] = talib.EMA(df['close'], timeperiod=14)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(df['close'], 12, 26, 9)
        df['MACD'] = macd - macdsignal
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        df['STOCH'] = slowk
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = upper, middle, lower
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        last = df.iloc[-1]
        signal, entry, tp, sl = self.generate_signal(last)

        self.signal_label.setText(f"Сигнал: {signal}\nТочка входа: {entry:.2f}\nTP: {tp:.2f}\nSL: {sl:.2f}")

        # === Построение графика ===
        apds = [
            mpf.make_addplot(df['SMA20'], color='blue'),
            mpf.make_addplot(df['SMA50'], color='red')
        ]
        fig, ax = plt.subplots(figsize=(10,6))
        mpf.plot(df, type='candle', ax=ax, volume=True, style='yahoo', addplot=apds, show_nontrading=True)
        ax.set_title(f"{symbol} — {signal}")
        plt.show()

    def generate_signal(self, last):
        score = 0
        if last['SMA20'] > last['SMA50']: score += 1
        if last['EMA14'] > last['SMA20']: score += 1
        if last['RSI'] < 30: score += 1
        if last['RSI'] > 70: score -= 1
        if last['MACD'] > 0: score += 1
        if last['STOCH'] > 80: score -= 1
        if last['ADX'] > 25: score += 1
        if last['close'] < last['BB_lower']: score += 1
        if last['close'] > last['BB_upper']: score -= 1
        if last['CCI'] > 100: score += 1
        if last['CCI'] < -100: score -= 1

        entry = last['close']
        atr = last['ATR'] if last['ATR'] > 0 else 50
        tp = entry + atr * 2
        sl = entry - atr * 2

        if score >= 3:
            return "BUY", entry, tp, sl
        elif score <= -3:
            return "SELL", entry, entry - (tp-entry), entry + (tp-entry)
        else:
            return "HOLD", entry, tp, sl

# === Запуск приложения ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec_())
