import gradio as gr
import cv2
import numpy as np

def detect_face_center(frame, width):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        return fx + fw // 2
    return width // 2

def create_short_video(video_path, caption_text, caption_style, start_time, duration):
    if not video_path:
        return None, "Kripya video upload karein!"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = int(height * (9 / 16))
    if target_w % 2 != 0:
        target_w -= 1

    output_path = "output_9_16.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, height))

    start_frame = int(start_time * fps)
    max_frames = int(duration * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_cx = width // 2
    frames_written = 0

    while cap.isOpened() and frames_written < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        detected_cx = detect_face_center(frame, width)
        current_cx = int(0.85 * current_cx + 0.15 * detected_cx)

        x1 = max(0, min(width - target_w, current_cx - target_w // 2))
        cropped = frame[:, x1:x1 + target_w]

        if caption_text:
            text = caption_text.upper()
            color = (0, 255, 255) if caption_style == "Hormozi (Yellow)" else (255, 255, 255)
            cv2.putText(cropped, text, (40, height - 160), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(cropped, text, (40, height - 160), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

        out.write(cropped)
        frames_written += 1

    cap.release()
    out.release()
    return output_path, f"Shorts video tayyar! ({frames_written/fps:.1f} seconds)"

with gr.Blocks(title="AI Shorts Generator") as demo:
    gr.Markdown("# 🎬 AI Shorts Generator (Running on GitHub)")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Long Video")
            caption_input = gr.Textbox(label="Short Caption / Hook", value="VIRAL MOMENT 🔥")
            caption_style = gr.Dropdown(choices=["Hormozi (Yellow)", "Clean (White)"], value="Hormozi (Yellow)", label="Caption Style")
            start_slider = gr.Slider(0, 300, value=0, step=1, label="Start Time (Seconds)")
            duration_slider = gr.Slider(5, 60, value=30, step=1, label="Short Duration (Seconds)")
            btn = gr.Button("🚀 Generate 9:16 Short", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Generated 9:16 Short")
            status = gr.Textbox(label="Status")

    btn.click(create_short_video, inputs=[video_input, caption_input, caption_style, start_slider, duration_slider], outputs=[video_output, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
