import cv2
import gradio as gr
from ultralytics import YOLO
import numpy as np

MODEL_PATH = "./best.pt"
model = None

try:
    model = YOLO(MODEL_PATH)
    print(f"Модель успешно загружена из {MODEL_PATH}")
except Exception as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель из {MODEL_PATH}. Ошибка: {e}")

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Обрабатывает один кадр видеопотока с помощью модели YOLO.
    frame: входной кадр в формате NumPy array (RGB от Gradio).
    Возвращает: обработанный кадр в формате NumPy array (RGB для Gradio).
    """
    if model is None:
        error_text = "Ошибка: Модель не загружена"
        cv2.putText(frame, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return frame

    try:
        results = model(frame)[0]
        annotated_rgb = results.plot()
        return annotated_rgb
    except Exception as e:
        print(f"Ошибка при обработке кадра: {e}")
        error_text = "Ошибка обработки кадра"
        cv2.putText(frame, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return frame

with gr.Blocks() as demo:
    gr.Markdown("## Real-time Video Processing")
    
    with gr.Row():
        input_img = gr.Image(sources=["webcam"], streaming=True, label="Веб-камера")
        output_img = gr.Image(label="Обработанное видео")
    
    input_img.stream(
        fn=process_frame,
        inputs=input_img,
        outputs=output_img,
        # every=0.1
    )

if __name__ == '__main__':
    demo.launch()