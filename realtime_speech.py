import speech_recognition as sr
import time

def listen_microphone():
    # Khởi tạo recognizer
    r = sr.Recognizer()
    
    print("Bắt đầu nghe... (Nhấn Ctrl+C để dừng)")
    
    while True:
        try:
            # Sử dụng microphone làm nguồn âm thanh
            with sr.Microphone() as source:
                print("\nĐang lắng nghe...")
                
                # Điều chỉnh nhiễu môi trường
                r.adjust_for_ambient_noise(source)
                
                # Lắng nghe âm thanh
                audio = r.listen(source)
                
                try:
                    # Sử dụng Google Speech Recognition
                    text = r.recognize_google(audio, language="vi-VN")
                    print("Bạn nói:", text)
                    
                except sr.UnknownValueError:
                    print("Không thể nhận dạng giọng nói")
                except sr.RequestError as e:
                    print("Lỗi từ Google Speech Recognition service; {0}".format(e))
                    
        except KeyboardInterrupt:
            print("\nĐã dừng nghe.")
            break
        except Exception as e:
            print("Lỗi:", str(e))
            continue

if __name__ == "__main__":
    listen_microphone()
