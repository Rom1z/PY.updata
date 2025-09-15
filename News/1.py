# -*- coding: utf-8 -*-
import sys
from datetime import datetime, timezone, timedelta
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QPushButton, QFileDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont

# --- Встроенный список событий ---
events = [
    {"summary": "FOMC Statement (Sep 2025)", "dt_utc": datetime(2025,9,17,18,0, tzinfo=timezone.utc), "comment": "Решение ФРС, высокая волатильность"},
    {"summary": "Jerome Powell Press Conference (Sep 2025)", "dt_utc": datetime(2025,9,17,18,30, tzinfo=timezone.utc), "comment": "Q&A после FOMC"},
    {"summary": "FOMC Statement (Oct 2025)", "dt_utc": datetime(2025,10,29,18,0, tzinfo=timezone.utc), "comment": "Решение ФРС"},
    {"summary": "Jerome Powell Press Conference (Oct 2025)", "dt_utc": datetime(2025,10,29,18,30, tzinfo=timezone.utc), "comment": "Q&A после FOMC"},
    {"summary": "FOMC Statement (Dec 2025)", "dt_utc": datetime(2025,12,10,19,0, tzinfo=timezone.utc), "comment": "Решение ФРС"},
    {"summary": "Jerome Powell Press Conference (Dec 2025)", "dt_utc": datetime(2025,12,10,19,30, tzinfo=timezone.utc), "comment": "Q&A после FOMC"}
]

# --- Функция конвертации UTC в МСК ---
def to_msk(dt_utc):
    return dt_utc + timedelta(hours=3)

# --- Главное окно ---
class PowellScheduleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Расписание Джерома Пауэлла")
        self.setMinimumSize(800, 400)
        self.setStyleSheet("""
            QWidget {background-color: #1E1E2F; color: #FFFFFF;}
            QTableWidget {background-color: #2C2C3E; gridline-color: #555;}
            QHeaderView::section {background-color: #3C3C5E; font-weight: bold; color: #FFF;}
            QPushButton {background-color: #6A5ACD; border-radius: 6px; padding: 6px;}
            QPushButton:hover {background-color: #7B68EE;}
            QLabel {font-size: 16px;}
        """)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        title = QLabel("Расписание выступлений Джерома Пауэлла (MSK)")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        self.layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Событие", "Дата UTC", "Время МСК", "Комментарий"])
        self.layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Импорт .ics")
        self.btn_load.clicked.connect(self.load_ics)
        btn_layout.addWidget(self.btn_load)
        self.layout.addLayout(btn_layout)

        self.load_events(events)
        self.highlight_upcoming()
        self.timer = QTimer()
        self.timer.timeout.connect(self.highlight_upcoming)
        self.timer.start(60 * 1000)  # обновление каждую минуту

    # --- Загрузка событий в таблицу ---
    def load_events(self, events_list):
        self.table.setRowCount(len(events_list))
        for row, ev in enumerate(events_list):
            utc_str = ev["dt_utc"].strftime("%Y-%m-%d %H:%M")
            msk_str = to_msk(ev["dt_utc"]).strftime("%Y-%m-%d %H:%M")
            self.table.setItem(row, 0, QTableWidgetItem(ev["summary"]))
            self.table.setItem(row, 1, QTableWidgetItem(utc_str))
            self.table.setItem(row, 2, QTableWidgetItem(msk_str))
            self.table.setItem(row, 3, QTableWidgetItem(ev["comment"]))
            for col in range(4):
                self.table.item(row, col).setTextAlignment(Qt.AlignCenter)

    # --- Подсветка ближайших событий ---
    def highlight_upcoming(self):
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        for row in range(self.table.rowCount()):
            dt_utc = events[row]["dt_utc"]
            diff = (dt_utc - now).total_seconds()
            for col in range(4):
                item = self.table.item(row, col)
                if 0 <= diff <= 86400:  # событие сегодня
                    item.setBackground(QColor("#FFD700"))  # золотой
                elif 0 <= diff <= 172800:  # завтра
                    item.setBackground(QColor("#FFA500"))  # оранжевый
                else:
                    item.setBackground(QColor("#2C2C3E"))

    # --- Загрузка .ics (опционально) ---
    def load_ics(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите .ics файл", "", "ICS Files (*.ics)")
        if path:
            from icalendar import Calendar
            with open(path, 'rb') as f:
                cal = Calendar.from_ical(f.read())
            new_events = []
            for component in cal.walk():
                if component.name == "VEVENT":
                    dt = component.get('dtstart').dt
                    if isinstance(dt, datetime):
                        dt = dt.replace(tzinfo=timezone.utc)
                    new_events.append({
                        "summary": str(component.get('summary')),
                        "dt_utc": dt,
                        "comment": str(component.get('description') or "")
                    })
            global events
            events = new_events
            self.load_events(events)
            self.highlight_upcoming()

# --- Запуск приложения ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PowellScheduleApp()
    window.show()
    sys.exit(app.exec())
