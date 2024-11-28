import sys
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
import torch
import traceback
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve, QPointF, QRectF
from PySide6.QtGui import (QPainter, QColor, QPen, QLinearGradient, QRadialGradient, 
                          QPainterPath, QTextCharFormat, QFont, QTextCursor)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QTextEdit, QPushButton, QComboBox, QLabel, QHBoxLayout, QFrame)
from transformers import WhisperProcessor, WhisperForConditionalGeneration


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
    add_newline = Signal()

    def __init__(self):
        super().__init__()
        self.current_transcription = ""  # Current transcription text
        self.history_text = []  # Array to store history
        self.init_ui()
        self.init_whisper()
        self.last_buffer_reset = time.time()
        self.update_text.connect(self.update_display)
        self.add_newline.connect(self._add_newline)

    def on_language_change(self, language):
        print(f"Changing language to: {language}")
        if self.model is not None and self.processor is not None:
            # Reset các biến streaming
            self.stable_tokens = None
            self.unstable_tokens = None
            self.eos_token = None

            # Xóa text hiện tại
            self.text_display.clear()
            print(f"Language updated to {language}")

    def init_ui(self):
        # Layout chính
        main_layout = QVBoxLayout()

        # Frame cho controls
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        controls_layout = QHBoxLayout()

        # Model selection
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.currentTextChanged.connect(self.load_model)

        # Language selection
        lang_label = QLabel("Language:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["english", "vietnamese", "french"])
        self.lang_combo.currentTextChanged.connect(self.on_language_change)

        # Record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)

        # Add controls to layout
        controls_layout.addWidget(model_label)
        controls_layout.addWidget(self.model_combo)
        controls_layout.addWidget(lang_label)
        controls_layout.addWidget(self.lang_combo)
        controls_layout.addWidget(self.record_button)
        controls_frame.setLayout(controls_layout)

        # Text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        # Set dark theme
        self.text_display.setStyleSheet(
            "QTextEdit { background-color: #2b2b2b; color: white; }")

        # Waveform
        self.waveform = WaveformWidget()

        # Add everything to main layout
        main_layout.addWidget(controls_frame)
        main_layout.addWidget(self.waveform)
        main_layout.addWidget(self.text_display)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Window properties
        self.setWindowTitle("Whisper GUI")
        self.setGeometry(100, 100, 800, 600)

    def init_whisper(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.blocksize = int(self.sample_rate * 0.3)  # 0.3 giây mỗi chunk
        self.model = None
        self.processor = None
        self.process_thread = None
        self.stable_tokens = None
        self.unstable_tokens = None
        self.eos_token = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        # Load model ngay khi khởi tạo
        self.load_model()

    def load_model(self):
        model_name = self.model_combo.currentText()
        print(f"Loading model {model_name}...")
        try:
            # Đổi tên model để phù hợp với transformers
            if model_name == "tiny":
                model_name = "openai/whisper-tiny"
            elif model_name == "base":
                model_name = "openai/whisper-base"
            elif model_name == "small":
                model_name = "openai/whisper-small"
            elif model_name == "medium":
                model_name = "openai/whisper-medium"
            elif model_name == "large":
                model_name = "openai/whisper-large"

            # Load processor and model with specific configuration
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            ).to(self.device)
            
            # Set initial language
            current_lang = self.lang_combo.currentText().lower()
            print(f"Setting initial language to: {current_lang}")

            # Reset các biến streaming
            self.stable_tokens = None
            self.unstable_tokens = None
            self.eos_token = None

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def process_audio(self):
        """Process audio data from the queue"""
        if self.model is None or self.processor is None:
            print("Model not loaded. Please load the model first.")
            return

        audio_buffer = np.array([], dtype=np.float32)
        try:
            while self.recording:
                # Get audio data from queue
                if self.audio_queue.empty():
                    time.sleep(0.1)
                    continue

                # Lấy audio data mới và thêm vào buffer
                audio_data = self.audio_queue.get()
                audio_data = audio_data.flatten().astype(np.float32)
                audio_buffer = np.concatenate([audio_buffer, audio_data])

                # Giới hạn độ dài buffer để tránh quá tải
                max_buffer_size = self.sample_rate * 15  # 15 giây
                current_time = time.time()
                buffer_reset_time = 15
                # Reset buffer sau mỗi 15 giây
                if current_time - self.last_buffer_reset > buffer_reset_time:
                    # Save current transcription to history before reset
                    if self.current_transcription.strip():
                        self.history_text.append(self.current_transcription.strip())
                    self.current_transcription = ""
                    self.last_buffer_reset = current_time
                    self.add_newline.emit()  # Emit signal instead of direct modification
                    audio_buffer = audio_data  # Reset buffer
                elif len(audio_buffer) > max_buffer_size:
                    audio_buffer = audio_buffer[-max_buffer_size:]

                try:
                    # Xử lý audio thành features
                    inputs = self.processor(
                        audio_buffer,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    )
                    input_features = inputs.input_features.to(self.device)

                    # Tạo attention mask
                    attention_mask = torch.ones_like(
                        input_features, dtype=torch.long, device=self.device)

                    # Generate token ids
                    predicted_ids = self.model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        task="transcribe",
                        language=self.lang_combo.currentText().lower(),
                        return_timestamps=False,
                        max_new_tokens=128,
                        num_beams=1,  # Giảm số beam để tăng tốc độ
                        forced_decoder_ids=None  # Disable forced decoder ids
                    )

                    # Decode token ids to text
                    transcription = self.processor.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True
                    )[0]

                    # Update the display with new text
                    self.current_transcription = transcription
                    self.update_text.emit(transcription)

                except Exception as e:
                    print(f"Error in transcription: {str(e)}")
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"Error in process_audio: {str(e)}")
            traceback.print_exc()

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.model:
            self.load_model()

        self.recording = True
        self.record_button.setText("Stop Recording")
        self.waveform.start_animation()

        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

        # Start audio input stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=self.blocksize
        )
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.record_button.setText("Start Recording")
        self.waveform.stop_animation()

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        if self.process_thread:
            self.process_thread.join()

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(status)
        # Add the new audio data to the queue
        self.audio_queue.put(indata.copy())
        self.waveform.update_audio_data(indata.copy())

    def merge_text(self, text1, text2):
        """
        Merge two texts intelligently, keeping the context and avoiding duplicates
        """
        if not text1:
            return text2

        # Tìm phần chung dài nhất giữa cuối text1 và đầu text2
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        max_overlap = 0
        overlap_pos = 0

        # Tìm vị trí overlap tốt nhất
        for i in range(len(words1)):
            for j in range(len(words2)):
                k = 0
                while (i + k < len(words1) and
                       k < len(words2) and
                       words1[i + k] == words2[k]):
                    k += 1
                if k > max_overlap:
                    max_overlap = k
                    overlap_pos = i

        if max_overlap > 0:
            # Kết hợp text với phần overlap
            result = " ".join(words1[:overlap_pos] + words2)
        else:
            # Nếu không có overlap, nối trực tiếp
            result = text1 + " " + text2

        return result

    def _add_newline(self):
        if len(self.text_display.toPlainText().strip()) > 0:
            self.text_display.append("")

    def update_display(self, text):
        # Create display text with history above and current transcription below
        display_text = ""
        if self.history_text:
            display_text = "\n".join(self.history_text) + "\n\n"
        display_text += "Current: " + text
        self.text_display.setPlainText(display_text)
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_display.setTextCursor(cursor)


def main():
    app = QApplication(sys.argv)
    window = WhisperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
