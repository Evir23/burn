from ultralytics import YOLO

# Загрузка модели
model = YOLO("yolo_best.pt")  # загрузка модели (более легкая версия YOLO для ускорения)

# Параметры тренировки
model.train(
    data='yolov11_project/data.yaml',  # Путь к файлу data.yaml
    epochs=50,                  # Уменьшено количество эпох для предварительного теста
    project='ds0/train',       # Папка для сохранения результатов
    name='exp',                # Имя папки с результатами
    exist_ok=True,             # Перезаписывать папку, если она уже существует
    cos_lr=True,               # Косинусный график скорости обучения для улучшения результата
    patience=50,               # Раннее завершение при отсутствии улучшений
    save_period=1              # Сохранение результатов после каждой эпохи
)

print("Обучение завершено.")