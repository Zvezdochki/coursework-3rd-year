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

# Минималистичные CSS стили
css = """
/* Основные стили */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background: #f8f9fa !important;
    color: #212529 !important;
}

/* Заголовок */
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid #e9ecef;
}

.main-header h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #495057;
    margin: 0 0 0.5rem 0;
}

.main-header p {
    font-size: 1rem;
    color: #6c757d;
    margin: 0;
}

/* Панели */
.control-panel {
    background: white !important;
    border: 1px solid #e9ecef !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    margin: 0 !important;
}

/* Видео контейнеры */
.video-container {
    border: 1px solid #e9ecef !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    background: white !important;
}

/* Ползунок */
.confidence-slider {
    margin: 1rem 0 !important;
}

/* Информационная панель */
.info-panel {
    background: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
}

.info-panel h4 {
    font-size: 0.875rem;
    font-weight: 600;
    color: #495057;
    margin: 0 0 0.5rem 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.info-panel ul {
    margin: 0;
    padding-left: 1rem;
}

.info-panel li {
    font-size: 0.875rem;
    color: #6c757d;
    margin-bottom: 0.25rem;
}

/* Статистика */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}

.stat-card {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.stat-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: #495057;
    display: block;
}

.stat-label {
    font-size: 0.75rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
}

/* Секции */
.section-title {
    font-size: 1.125rem;
    font-weight: 600;
    color: #495057;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e9ecef;
}
"""

with gr.Blocks(css=css, title="🚦 AI Детектор", theme=gr.themes.Soft()) as demo:
    # Главный заголовок
    gr.HTML("""
        <div class="main-header">
            <h1>🚦 Распознавание дорожных знаков в реальном времени</h1>
            <p>Интеллектуальная система детекции с использованием YOLO</p>
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
            
            # Информационная панель
            gr.HTML("""
                <div class="info-panel">
                    <h4>📊 Информация</h4>
                    <ul>
                        <li><strong>0.1-0.3:</strong> Высокая чувствительность</li>
                        <li><strong>0.4-0.6:</strong> Сбалансированный режим</li>
                        <li><strong>0.7-0.9:</strong> Только точные детекции</li>
                    </ul>
                </div>
            """)
        
        # Правая панель - результат
        with gr.Column(scale=2):
            gr.HTML("<h3 class='section-title'>🖥️ Обработанное видео</h3>")
            output_img = gr.Image(
                label="🎬 Результат детекции",
                elem_classes=["video-container"]
            )
            
            # Статистика
            gr.HTML("""
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">Real-time</div>
                        <div class="stat-label">Обработка</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">YOLO v8</div>
                        <div class="stat-label">Модель</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">Live</div>
                        <div class="stat-label">Статус</div>
                    </div>
                </div>
            """)
    
    # Футер с дополнительной информацией
    gr.HTML("""
        <div class="info-panel" style="margin-top: 30px;">
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
    demo.launch()