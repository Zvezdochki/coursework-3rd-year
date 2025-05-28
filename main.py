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

def process_frame(frame: np.ndarray, confidence_threshold: float) -> np.ndarray:
    """
    Обрабатывает один кадр видеопотока с помощью модели YOLO.
    frame: входной кадр в формате NumPy array (RGB от Gradio).
    confidence_threshold: минимальный порог уверенности для обнаружений (0.0 - 1.0).
    Возвращает: обработанный кадр в формате NumPy array (RGB для Gradio).
    """
    if model is None:
        error_text = "Ошибка: Модель не загружена"
        cv2.putText(frame, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return frame

    try:
        # Передаем порог уверенности в модель
        results = model(frame, conf=confidence_threshold)[0]
        annotated_rgb = results.plot()
        return annotated_rgb
    except Exception as e:
        print(f"Ошибка при обработке кадра: {e}")
        error_text = "Ошибка обработки кадра"
        cv2.putText(frame, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return frame

with gr.Blocks() as demo:
    gr.Markdown("## Распознавание дорожных знаков в реальном времени")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["webcam"], streaming=True, label="Веб-камера")
            # Добавляем ползунок для управления порогом уверенности
            confidence_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.6,  # Значение по умолчанию
                step=0.05,
                label="Порог уверенности",
                info="Минимальная уверенность для отображения обнаружений"
            )
        
        output_img = gr.Image(label="Обработанное видео")
    
    # Теперь передаем и кадр, и значение ползунка в функцию
    input_img.stream(
        fn=process_frame,
        inputs=[input_img, confidence_slider],
        outputs=output_img,
        # every=0.1
    )

if __name__ == '__main__':
    demo.launch()