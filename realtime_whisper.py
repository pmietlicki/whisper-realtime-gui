import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
from datetime import datetime

# Khởi tạo model Whisper (có thể chọn các model khác như "base", "small", "medium", "large")
model = whisper.load_model("base")

# Cấu hình thu âm
samplerate = 16000  # Whisper yêu cầu 16kHz
channels = 1        # Mono
blocksize = 30 * samplerate  # Buffer cho mỗi 30 giây
audio_queue = queue.Queue()
recording = True

def audio_callback(indata, frames, time, status):
    """Callback function để nhận audio data"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def process_audio():
    """Xử lý audio và thực hiện nhận dạng"""
    while recording:
        if not audio_queue.empty():
            # Lấy audio data từ queue
            audio_data = audio_queue.get()
            
            # Chuẩn bị audio data cho Whisper
            audio_data = audio_data.flatten().astype(np.float32)
            
            try:
                # Thực hiện nhận dạng
                result = model.transcribe(audio_data, language="fr")
                
                # In kết quả với timestamp
                if result["text"].strip():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] {result['text']}")
            except Exception as e:
                print(f"Lỗi khi nhận dạng: {str(e)}")

def main():
    global recording
    
    print("Bắt đầu thu âm từ system audio... (Nhấn Ctrl+C để dừng)")
    
    # Tạo thread xử lý audio
    process_thread = threading.Thread(target=process_audio)
    process_thread.start()
    
    try:
        # Bắt đầu thu âm
        with sd.InputStream(samplerate=samplerate,
                          channels=channels,
                          callback=audio_callback,
                          blocksize=blocksize):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDừng thu âm...")
        recording = False
        process_thread.join()
    except Exception as e:
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()
