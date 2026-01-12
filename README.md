# 🧠 Mental Health Classification System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36.0-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based mental health classification system using BERT for identifying 7 mental health conditions from text data.

## 🎯 Project Overview

This system classifies text into one of seven mental health categories:
- **Normal** (No mental health concerns)
- **Anxiety** (Excessive worry, nervousness)
- **Depression** (Persistent sadness, hopelessness)
- **Bipolar** (Mood swings, manic/depressive episodes)
- **Stress** (Overwhelming pressure, tension)
- **Suicidal** (Self-harm ideation - **HIGH RISK**)
- **Personality Disorder** (Behavioral/thinking patterns)

### 🏆 Current Performance
- **Test Accuracy:** 83.00%
- **Weighted F1 Score:** 83.10%
- **Macro F1 Score:** 78.88%
- **Target:** 90%+ accuracy

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/mentalHealth.git
cd mentalHealth
```

2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```


3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Prepare Data
```bash
# Place your dataset in data/raw/Combined Data.csv
python src/preprocess.py
```
5. Train Model
```bash
python src/train.py
```
6. Visualize Results
```
python src/visualize.py
```
7. Make Predictions
```
python src/predict.py --text "Your input text here"
```

🎓 Model Architecture
Base Model: bert-base-uncased (110M parameters)

Classification Head: Linear layer with dropout

Max Sequence Length: 256 tokens

Batch Size: 8 (effective: 16 with gradient accumulation)

Optimizer: AdamW with learning rate 8e-6

Loss Function: Label Smoothing Cross-Entropy with class weights

Advanced Features
✅ Gradient Accumulation - Effective larger batch size
✅ Class Weighting - Handles 17.91:1 class imbalance
✅ Label Smoothing - Prevents overconfidence
✅ Early Stopping - Prevents overfitting
✅ Learning Rate Warmup - Stable training

📊 Training Details
Hyperparameters
```
MAX_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 8e-6
EPOCHS = 6
DROPOUT = 0.4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
GRADIENT_ACCUMULATION = 2
```

Hardware Requirements
GPU: NVIDIA RTX 3050 (4GB VRAM) or better

RAM: 16GB recommended

Storage: 5GB (including models and data)

📈 Performance Analysis
Per-Class F1 Scores

| Class                | Precision | Recall | F1-Score | Support |
| -------------------- | --------- | ------ | -------- | ------- |
| Normal               | 96.3%     | 94.7%  | 95.5%    | 2405    |
| Anxiety              | 83.5%     | 87.8%  | 85.6%    | 543     |
| Bipolar              | 85.7%     | 81.6%  | 83.6%    | 375     |
| Depression           | 80.3%     | 76.7%  | 78.5%    | 2263    |
| Suicidal             | 72.7%     | 75.6%  | 74.1%    | 1596    |
| Stress               | 68.9%     | 75.9%  | 72.2%    | 344     |
| Personality disorder | 56.6%     | 70.1%  | 62.7%    | 134     |


Key Insights
Strengths: Excellent on Normal/Anxiety/Bipolar classes

Challenges: Depression-Suicidal confusion (semantic overlap)

Bottleneck: Personality disorder (only 134 samples)

🛠️ Technologies Used
Deep Learning: PyTorch 2.1.0

NLP: Hugging Face Transformers 4.36.0

Data Processing: Pandas, NumPy, Scikit-learn

Visualization: Matplotlib, Seaborn

Explainability: SHAP

📝 Usage Examples
Batch Prediction
```
from src.predict import MentalHealthPredictor

predictor = MentalHealthPredictor('models/best_model.pt')
texts = [
    "I feel so sad and hopeless lately",
    "Everything is going great, I'm so happy!"
]
results = predictor.predict_batch(texts)
```

Single Prediction with Risk Level
```
result = predictor.predict("I can't take this anymore")
print(f"Condition: {result['prediction']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.2%}")
```
🤝 Contributing
Contributions are welcome! Please:

Fork the repository

Create a feature branch (git checkout -b feature/improvement)

Commit changes (git commit -am 'Add new feature')

Push to branch (git push origin feature/improvement)

Open a Pull Request

⚠️ Disclaimer
This system is for research and educational purposes only. It should NOT replace professional mental health diagnosis or treatment. If you or someone you know is experiencing mental health issues, please contact:

National Suicide Prevention Lifeline (US): 988

Crisis Text Line: Text HOME to 741741

International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

📄 License
This project is licensed under the MIT License - see LICENSE file for details.

Acknowledgments
Hugging Face for the Transformers library

PyTorch team for the deep learning framework

Mental health dataset contributors