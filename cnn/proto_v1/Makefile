DATA_DIR=data/coco128
CHECKPOINT_DIR=checkpoints
EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=1e-4
VAL_SPLIT=0.1
NUM_WORKERS=8
PRINT_INTERVAL=10
SAVE_INTERVAL=10
SAMPLE_IMAGE=$(DATA_DIR)/images/train2017/000000000009.jpg
MODEL_PATH=$(CHECKPOINT_DIR)/model_final.pt
OUTPUT_DIR=outputs
VISUALIZE_OUTPUT=$(OUTPUT_DIR)/visualization_test.jpg
INFERENCE_OUTPUT=$(OUTPUT_DIR)/inference_result.jpg
COCO_ZIP=data/coco128.zip

all: train

data: $(DATA_DIR)/images $(DATA_DIR)/labels

$(DATA_DIR)/images $(DATA_DIR)/labels: $(COCO_ZIP)
	@echo "Extracting COCO128 dataset..."
	unzip -o $(COCO_ZIP) -d data/

train: data
	@echo "Training model with optimized configuration..."
	python train.py \
		--data-dir $(DATA_DIR) \
		--checkpoint-dir $(CHECKPOINT_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE) \
		--val-split $(VAL_SPLIT) \
		--num-workers $(NUM_WORKERS) \
		--print-interval $(PRINT_INTERVAL) \
		--save-interval $(SAVE_INTERVAL)

train-basic: data
	@echo "Training model with basic configuration..."
	python train.py \
		--data-dir $(DATA_DIR) \
		--checkpoint-dir $(CHECKPOINT_DIR) \
		--epochs $(EPOCHS) \
		--batch-size 8 \
		--learning-rate $(LEARNING_RATE) \
		--val-split $(VAL_SPLIT) \
		--num-workers 2 \
		--print-interval $(PRINT_INTERVAL) \
		--save-interval $(SAVE_INTERVAL) \
		--no-mixed-precision \
		--no-augmentation

inference: data
	@echo "Running inference on sample image..."
	mkdir -p $(OUTPUT_DIR)
	python main.py inference $(SAMPLE_IMAGE) \
		--model-path $(MODEL_PATH) \
		--output-path $(INFERENCE_OUTPUT)

visualize: data
	@echo "Visualizing predictions on dataset..."
	mkdir -p $(OUTPUT_DIR)
	python main.py visualize \
		--data-dir $(DATA_DIR) \
		--model-path $(MODEL_PATH) \
		--output-path $(VISUALIZE_OUTPUT)

clean:
	@echo "Cleaning generated files..."
	rm -rf outputs/
	rm -rf checkpoints/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf *.log

fclean: clean
	@echo "Full cleaning - removing all data..."
	rm -rf data/coco128/
	rm -rf coco128/
	rm -rf data/images_backup/

re: fclean all

activate:
	@echo "To activate the virtual environment, run:"
	@echo "source ../../cnn/.venv/bin/activate.fish" 

.PHONY: all train train-basic inference visualize clean fclean re activate
