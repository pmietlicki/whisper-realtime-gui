import whisper
import json
from datetime import timedelta

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds // 60) % 60
    seconds = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def transcribe_with_timestamps(audio_path, model="medium"):
    # Load the larger model for improved accuracy
    model = whisper.load_model(model)
    
    # Transcribe with word timestamps and adjust parameters
    result = model.transcribe(
        audio_path, 
        language="fr", 
        word_timestamps=True,
        condition_on_previous_text=False,  # Reduces dependency on previous text, reducing cumulative errors
        no_speech_threshold=0.5,           # Adjusts silence detection sensitivity
        logprob_threshold=-1.0             # Accepts lower confidence predictions
    )
    
    # Format segments
    formatted_segments = []
    for segment in result["segments"]:
        formatted_segment = {
            "start_time": format_timestamp(segment["start"]),
            "end_time": format_timestamp(segment["end"]),
            "text": segment["text"].strip(),
        }
        formatted_segments.append(formatted_segment)
    
    # Save to JSON file
    with open("transcription.json", "w", encoding="utf-8") as f:
        json.dump(formatted_segments, f, ensure_ascii=False, indent=2)
    
    # Save as formatted text (more readable)
    with open("transcription.txt", "w", encoding="utf-8") as f:
        for segment in formatted_segments:
            f.write(f'[{segment["start_time"]} -> {segment["end_time"]}]\n')
            f.write(f'{segment["text"]}\n\n')
    
    # Create markdown table format with translation column
    with open("transcription_table.md", "w", encoding="utf-8") as f:
        f.write("| Thời gian bắt đầu | Thời gian kết thúc | Nội dung | Dịch tiếng Việt |\n")
        f.write("|-------------------|-------------------|----------|----------------|\n")
        for segment in formatted_segments:
            f.write(f'| {segment["start_time"]} | {segment["end_time"]} | {segment["text"]} | |\n')
    
    # Create a Word file with a 4-column table
    try:
        from docx import Document
        doc = Document()
        table = doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'
        
        # Add header
        header_cells = table.rows[0].cells
        header_cells[0].text = 'Thời gian bắt đầu'
        header_cells[1].text = 'Thời gian kết thúc'
        header_cells[2].text = 'Nội dung'
        header_cells[3].text = 'Dịch tiếng Việt'
        
        # Add data
        for segment in formatted_segments:
            row_cells = table.add_row().cells
            row_cells[0].text = segment["start_time"]
            row_cells[1].text = segment["end_time"]
            row_cells[2].text = segment["text"]
            row_cells[3].text = ""  # Translation column left blank
        
        doc.save('transcription.docx')
        print("Đã tạo file Word thành công!")
    except ImportError:
        print("Để tạo file Word, hãy cài đặt python-docx: pip install python-docx")
    
    return formatted_segments

# Usage
audio_file = "demo.WAV"  # Change to your audio file path
segments = transcribe_with_timestamps(audio_file)

# Print results for verification
for segment in segments:
    print(f'[{segment["start_time"]} -> {segment["end_time"]}]')
    print(segment["text"])
    print()
