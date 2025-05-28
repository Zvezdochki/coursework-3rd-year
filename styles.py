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
    border: 1px solid #e9ecef !important;
    border-radius: 8px !important;
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