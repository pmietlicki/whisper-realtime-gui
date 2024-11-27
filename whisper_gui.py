import sys
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
import torch
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve, QPointF, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QLinearGradient, QRadialGradient, QPainterPath, QTextCharFormat, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QTextEdit, QPushButton, QComboBox, QLabel, QHBoxLayout, QFrame)


class WaveformWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.waves = []
        self.target_waves = []
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_waves)
        self.animation_timer.start(30)  # Faster updates for smoother animation
        self.is_recording = False
        self.transition_speed = 0.15  # Controls how fast waves transition

    def start_animation(self):
        self.is_recording = True
        self.waves = [0.1] * 30  # Start with small waves

    def stop_animation(self):
        self.is_recording = False
        self.target_waves = [0] * 30

    def update_audio_data(self, data):
        if len(data) > 0:
            normalized = np.abs(data) / np.max(np.abs(data) + 1e-10)
            if self.is_recording:
                # Create more varied wave heights with higher amplitude
                chunk_size = len(normalized) // 30
                self.target_waves = [
                    # Increased amplitude
                    normalized[i:i + chunk_size].mean() * 1.2
                    for i in range(0, len(normalized), chunk_size)
                ][:30]
                # Add more randomness for dynamic look (±30% variation)
                self.target_waves = [
                    w * (1 + np.random.uniform(-0.3, 0.3)) for w in self.target_waves]
            else:
                self.target_waves = [0] * 30
            self.update()

    def update_waves(self):
        if not self.waves:
            self.waves = [0] * 30
            self.target_waves = [0] * 30

        # More responsive wave transitions
        for i in range(len(self.waves)):
            if self.is_recording:
                # Enhanced dynamic variation
                target = self.target_waves[i] * \
                    (1 + np.sin(time.time() * 6 + i) * 0.15)
                target *= 1 + np.cos(time.time() * 4) * \
                    0.1  # Additional wave motion
            else:
                target = 0

            # Faster transitions
            self.waves[i] += (target - self.waves[i]) * 0.2

        self.update()

    def paintEvent(self, event):
        if not self.waves:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        try:
            width = self.width()
            height = self.height()
            center_y = height / 2
            bar_width = width / (len(self.waves) * 1.5)
            max_height = height * 0.85  # Slightly higher bars

            # Green theme gradient
            gradient = QLinearGradient(0, 0, 0, height)
            if self.is_recording:
                # Vibrant green colors during recording
                gradient.setColorAt(0, QColor(46, 204, 113))  # Bright green
                gradient.setColorAt(0.5, QColor(39, 174, 96))  # Medium green
                gradient.setColorAt(1, QColor(33, 150, 83))  # Dark green
            else:
                # Subtle green when not recording
                gradient.setColorAt(0, QColor(46, 204, 113, 200))
                gradient.setColorAt(1, QColor(33, 150, 83, 200))

            painter.setPen(Qt.NoPen)
            painter.setBrush(gradient)

            for i, amplitude in enumerate(self.waves):
                x = width * i / len(self.waves)

                # Enhanced wave motion
                wave_effect = np.sin(time.time() * 4 + i * 0.5) * 0.08
                bar_height = max_height * (amplitude + wave_effect)

                if self.is_recording:
                    # Enhanced pulsing effect
                    pulse = 1 + np.sin(time.time() * 5) * 0.08
                    bar_height *= pulse

                rect = QRectF(
                    x + bar_width/2,
                    center_y - bar_height/2,
                    bar_width,
                    bar_height
                )

                # Green glow effect when recording
                if self.is_recording and amplitude > 0.1:
                    glow = QPainterPath()
                    glow.addRoundedRect(rect, bar_width/2, bar_width/2)
                    painter.fillPath(glow, QColor(46, 204, 113, 40))

                painter.drawRoundedRect(rect, bar_width/2, bar_width/2)

        finally:
            painter.end()


class WhisperGUI(QMainWindow):
    update_text = Signal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_whisper()

    def init_ui(self):
        self.setWindowTitle('Whisper Realtime Transcription')
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QTextEdit {
                background-color: #262626;
                color: #ffffff;
                border: none;
                border-radius: 15px;
                padding: 15px;
                font-size: 14px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto;
                selection-background-color: #404040;
            }
            QPushButton {
                background-color: #34c759;  /* Apple green */
                color: white;
                border: none;
                border-radius: 20px;
                padding: 12px 25px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2fb350;
            }
            QPushButton:pressed {
                background-color: #2aa147;
            }
            QComboBox {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 5px 10px;
                min-width: 100px;
                font-size: 13px;
                font-weight: 500;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QComboBox:hover {
                background-color: rgba(255, 255, 255, 0.15);
            }
            QComboBox QAbstractItemView {
                background-color: #262626;
                color: white;
                selection-background-color: #34c759;
                border: none;
                border-radius: 6px;
                padding: 5px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
                font-weight: 500;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #262626;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #404040;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 20, 30, 30)

        # Top toolbar with controls
        toolbar = QWidget()
        toolbar.setStyleSheet("""
            QWidget {
                background-color: rgba(45, 45, 45, 0.7);
                border-radius: 10px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(15, 10, 15, 10)
        toolbar_layout.setSpacing(15)

        # Model selection with icon
        model_layout = QHBoxLayout()
        model_layout.setSpacing(8)
        model_icon = QLabel("🎯")
        model_icon.setStyleSheet("background: transparent;")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("small")
        model_layout.addWidget(model_icon)
        model_layout.addWidget(self.model_combo)
        toolbar_layout.addLayout(model_layout)

        # Vertical separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("background-color: rgba(255, 255, 255, 0.1);")
        toolbar_layout.addWidget(separator)

        # Language selection with icon
        lang_layout = QHBoxLayout()
        lang_layout.setSpacing(8)
        lang_icon = QLabel("🌍")
        lang_icon.setStyleSheet("background: transparent;")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["fr", "en", "vi", "auto"])
        lang_layout.addWidget(lang_icon)
        lang_layout.addWidget(self.lang_combo)
        toolbar_layout.addLayout(lang_layout)

        # Add stretch to push controls to the left
        toolbar_layout.addStretch()

        # Recording button in toolbar
        self.toggle_button = QPushButton("Start Recording")
        self.toggle_button.setMinimumWidth(150)
        self.toggle_button.clicked.connect(self.toggle_recording)
        toolbar_layout.addWidget(self.toggle_button)

        layout.addWidget(toolbar)

        # Waveform
        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)

        # Text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setMinimumHeight(250)
        layout.addWidget(self.text_display)

        # Connect signal
        self.update_text.connect(self.update_display)

    def init_whisper(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.samplerate = 16000
        self.channels = 1
        self.blocksize = 3 * self.samplerate
        self.model = None
        self.process_thread = None

    def load_model(self):
        model_name = self.model_combo.currentText()
        try:
            # Thử sử dụng MPS
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
                self.model = whisper.load_model(model_name)
                # Chuyển model sang MPS một cách an toàn hơn
                try:
                    self.model = self.model.to(device)
                    print("Đang sử dụng Apple GPU (MPS)")
                except Exception as e:
                    print(f"Không thể sử dụng GPU, chuyển sang CPU: {str(e)}")
                    self.model = self.model.to("cpu")
            else:
                self.model = whisper.load_model(model_name, device="cpu")
                print("Sử dụng CPU vì không có GPU")
        except Exception as e:
            print(f"Lỗi khi tải model: {str(e)}")
            self.model = whisper.load_model(model_name, device="cpu")
            print("Sử dụng CPU do lỗi")

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.model:
            self.load_model()

        self.recording = True
        self.toggle_button.setText("Stop Recording")
        self.waveform.start_animation()

        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

        # Start audio input stream
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=self.blocksize
        )
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.toggle_button.setText("Start Recording")
        self.waveform.stop_animation()

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        if self.process_thread:
            self.process_thread.join()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())
        self.waveform.update_audio_data(indata.copy())

    def process_audio(self):
        while self.recording:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_data = audio_data.flatten().astype(np.float32)

                try:
                    result = self.model.transcribe(
                        audio_data,
                        language=self.lang_combo.currentText(),
                        fp16=False,
                        condition_on_previous_text=True,
                        best_of=1,
                        beam_size=1,
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.6
                    )

                    if result["text"].strip():
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        text = f"[{timestamp}] {result['text']}\n"
                        self.update_text.emit(text)
                except Exception as e:
                    print(f"Error in transcription: {str(e)}")

    def update_display(self, text):
        cursor = self.text_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)

        # Add two newlines before new timestamp for better readability
        if self.text_display.toPlainText():  # If there's existing text
            cursor.insertText("\n\n")  # Add extra newlines between entries

        # Split timestamp and content
        timestamp, content = text.split("]", 1)
        timestamp += "]"

        # Set format for timestamp (gray color)
        timestamp_format = QTextCharFormat()
        timestamp_format.setForeground(
            QColor(128, 128, 128))  # Gray color for timestamp
        timestamp_format.setFontWeight(QFont.Bold)

        # Set format for content (white color)
        content_format = QTextCharFormat()
        content_format.setForeground(
            QColor(255, 255, 255))  # White color for content

        # Insert timestamp with gray format
        cursor.insertText(timestamp, timestamp_format)

        # Insert content with white format
        cursor.insertText(content, content_format)

        # Scroll to the new text
        self.text_display.setTextCursor(cursor)
        self.text_display.ensureCursorVisible()


def main():
    app = QApplication(sys.argv)
    window = WhisperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
