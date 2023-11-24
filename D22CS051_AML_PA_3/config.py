# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 101
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 500

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = "16-mixed"