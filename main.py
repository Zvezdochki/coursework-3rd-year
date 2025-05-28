import cv2
import gradio as gr
from ultralytics import YOLO
import numpy as np
from styles import css
import os

MODEL_PATH = "./best.pt"
model = None

try:
    model = YOLO(MODEL_PATH)
    print(f"Модель успешно загружена из {MODEL_PATH}")
except Exception as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель из {MODEL_PATH}. Ошибка: {e}")
    os._exit(1)  # Завершаем программу с ошибкой

def process_frame(frame: np.ndarray, confidence_threshold: float) -> np.ndarray:
    """
    Обрабатывает один кадр видеопотока с помощью модели YOLO.
    frame: входной кадр в формате NumPy array (RGB от Gradio).
    confidence_threshold: минимальный порог уверенности для обнаружений (0.0 - 1.0).
    """
    if model is None:
        error_text = "Ошибка: Модель не загружена"
        cv2.putText(frame, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return frame

    # Исправляем отзеркаливание изображения с веб-камеры
    frame = cv2.flip(frame, 1)

    try:
        # Передаем порог уверенности в модель
        results = model(frame, conf=confidence_threshold)[0]
        return results.plot()
            
    except Exception as e:
        print(f"Ошибка при обработке кадра: {e}")
        error_text = "Ошибка обработки кадра"
        cv2.putText(frame, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return frame


with gr.Blocks(css=css, title="🚦 AI Детектор", theme=gr.themes.Soft()) as demo:
    # Главный заголовок
    gr.HTML("""
        <div class="main-header">
            <h1>🚦 Распознавание дорожных знаков в реальном времени</h1>
        </div>
    """)
    
    # Основной интерфейс
    with gr.Row():
        # Левая панель - управление
        with gr.Column(scale=1, elem_classes=["control-panel"]):
            gr.HTML("<h3 class='section-title'>🎮 Панель управления</h3>")
            
            input_img = gr.Image(
                sources=["webcam"], 
                streaming=True, 
                label="📹 Веб-камера",
                elem_classes=["video-container"]
            )
            
            # Ползунок уверенности
            confidence_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.6,
                step=0.05,
                label="🎯 Порог уверенности",
                info="Настройте чувствительность детекции",
                elem_classes=["confidence-slider"]
            )
        
        # Правая панель - результат
        with gr.Column(scale=2, elem_classes=["control-panel"]):
            gr.HTML("<h3 class='section-title'>🖥️ Обработанное видео</h3>")
            output_img = gr.Image(
                label="🎬 Результат детекции",
                elem_classes=["video-container"]
            )
            # Футер с дополнительной информацией
            gr.HTML("""
                <div class="info-panel">
                    <h4>💡 Советы по использованию:</h4>
                    <p>• Убедитесь в хорошем освещении для лучшего качества детекции</p>
                    <p>• Настройте порог уверенности в зависимости от ваших потребностей</p>
                    <p>• Держите камеру устойчиво для стабильной работы</p>
                </div>
            """)
    
    # Подключение обработки
    input_img.stream(
        fn=process_frame,
        inputs=[input_img, confidence_slider],
        outputs=output_img,
    )

if __name__ == '__main__':
    demo.launch(inbrowser=True)