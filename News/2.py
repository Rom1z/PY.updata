# -*- coding: utf-8 -*-
import sys
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from plyer import notification

FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

class FedScheduleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Расписание заседаний ФРС")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("""
            QWidget {background-color: #1E1E2F; color: #FFFFFF; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
            QTableWidget {background-color: #2C2C3E; gridline-color: #555;}
            QTableWidget::item {padding: 5px;}
            QTableWidget::item:selected {background-color: #5A5A7A;}
            QHeaderView::section {background-color: #3C3C5E; font-weight: bold; color: #FFF;}
            QPushButton {background-color: #0078D4; color: #FFF; border-radius: 6px; padding: 8px;}
            QPushButton:hover {background-color: #005A9E;}
        """)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Дата UTC", "Время МСК", "Событие"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.table)

        self.btn_update = QPushButton("Обновить расписание")
        self.btn_update.clicked.connect(self.refresh_schedule)
        self.layout.addWidget(self.btn_update)

        self.previous_events = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_schedule)
        self.timer.start(3600000)  # обновление каждый час

        self.refresh_schedule()

    def fetch_fomc_events(self):
        try:
            response = requests.get(FOMC_URL)
            response.raise_for_status()
        except Exception as e:
            print(f"Ошибка загрузки FOMC: {e}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        events = []

        # Парсим все блоки с датами и названиями заседаний
        for div in soup.select('div.fomc-calender table tr'):
            cols = div.find_all('td')
            if len(cols) >= 2:
                date_text = cols[0].get_text(strip=True)
                event_text = cols[1].get_text(strip=True)
                try:
                    # Преобразуем дату в datetime
                    event_date = datetime.strptime(date_text, "%B %d, %Y")
                    events.append({"dt_utc": event_date, "event": event_text})
                except:
                    continue
        return events

    def update_table(self, events):
        self.table.setRowCount(len(events))
        now = datetime.utcnow()
        for i, ev in enumerate(events):
            utc_str = ev["dt_utc"].strftime("%Y-%m-%d")
            msk_str = (ev["dt_utc"] + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
            self.table.setItem(i, 0, QTableWidgetItem(utc_str))
            self.table.setItem(i, 1, QTableWidgetItem(msk_str))
            self.table.setItem(i, 2, QTableWidgetItem(ev["event"]))
            for col in range(3):
                self.table.item(i, col).setTextAlignment(Qt.AlignCenter)

            diff = (ev["dt_utc"] - now).total_seconds()
            color = None
            if 0 <= diff <= 86400:
                color = QColor("#FFD700")
            elif 0 <= diff <= 172800:
                color = QColor("#FFA500")
            else:
                color = QColor("#2C2C3E")
            for col in range(3):
                self.table.item(i, col).setBackground(color)

    def refresh_schedule(self):
        events = self.fetch_fomc_events()
        if not events:
            return
        new_events = [ev for ev in events if ev not in self.previous_events]
        for ev in new_events:
            notification.notify(
                title='Новое событие ФРС',
                message=f"{ev['event']} - {ev['dt_utc'].strftime('%Y-%m-%d')}",
                timeout=10
            )
        self.previous_events = events
        self.update_table(events)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FedScheduleApp()
    window.show()
    sys.exit(app.exec())
