import os

# Paths
ROOT_DIR = "D:\\mentalHealth"
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Data files
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "Combined Data.csv")
PROCESSED_TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, "train.csv")
PROCESSED_VAL_FILE = os.path.join(PROCESSED_DATA_DIR, "val.csv")
PROCESSED_TEST_FILE = os.path.join(PROCESSED_DATA_DIR, "test.csv")

# ============================================================
# OPTIMIZED TRAINING CONFIGURATION FOR 90%+ ACCURACY
# ============================================================

# Model
MODEL_NAME = "bert-base-uncased"

# Training hyperparameters - OPTIMIZED
MAX_LENGTH = 256                     # DOUBLED from 128 (handles longer texts)
BATCH_SIZE = 8                       # REDUCED from 16 (allows larger sequences on 4GB GPU)
LEARNING_RATE = 8e-6                 # REDUCED from 2e-5 (more stable convergence)
EPOCHS = 6                           # INCREASED from 3 (more learning)
RANDOM_SEED = 42

# Advanced training parameters
WARMUP_RATIO = 0.15                  # 15% warmup for stability
DROPOUT = 0.4                        # INCREASED from 0.3 (better generalization)
WEIGHT_DECAY = 0.01                  # L2 regularization
GRADIENT_ACCUMULATION_STEPS = 2      # Effective batch size = 8 * 2 = 16
MAX_GRAD_NORM = 1.0                  # Gradient clipping
LABEL_SMOOTHING = 0.1                # Prevent overconfidence

# Early stopping
PATIENCE = 3                         # Stop if no improvement for 3 epochs

# Class labels
LABELS = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 
          'Personality disorder', 'Stress', 'Suicidal']
NUM_LABELS = len(LABELS)

# Risk level mapping for inference
RISK_LEVELS = {
    'Normal': 'No Risk',
    'Anxiety': 'Mild Distress',
    'Stress': 'Mild Distress',
    'Depression': 'Moderate Risk',
    'Bipolar': 'Moderate Risk',
    'Personality disorder': 'Moderate Risk',
    'Suicidal': 'High/Suicidal Risk'
}
