.PHONY: install train train-docker api-build api-run train-build train-run clean logs

# Установка зависимостей на хосте
install:
	pip install -r app/requirements.txt

# Обучение модели на хосте
train:
	python training/train_call_model.py

# Сборка образа API
api-build:
	docker build -f Dockerfile.api -t call-classifier-api .

# Запуск API-сервиса
api-run:
	docker run -p 8000:8000 call-classifier-api

# Сборка образа для обучения модели
train-build:
	docker build -f Dockerfile.train -t call-classifier-train .

# Запуск обучения модели в Docker с переносом кэша Whisper
train-run:
	docker run --rm \
		-v ~/.cache/whisper:/root/.cache/whisper \
		-v $(PWD)/app/models:/app/app/models \
		-v $(PWD)/conversations:/app/conversations \
		-v $(PWD)/conversations_augmented:/app/conversations_augmented \
		call-classifier-train

# Просмотр логов API
logs:
	tail -f app/service.log

# Очистка временных и модельных файлов
clean:
	rm -rf __pycache__ */__pycache__ *.log *.tmp *.wav *.pkl *.joblib conversations_augmented/ app/models/*.pkl app/models/*.joblib

