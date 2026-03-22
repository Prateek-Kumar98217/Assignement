# Assignment End to End ML Pipeline

## Overview

This project implements an end-to-end machine learning pipeline and inference API designed to analyze personal journal entries and biometric metadata (sleep, energy levels, stress levels, ambience). It predicts a user's emotional state and intensity, and subsequently routes them through a simple **Decision Engine** to recommend safe, low-friction mental health interventions.

## Quick Start (How to Run)

### 1. Clone the repository

```bash
git clone https://github.com/Prateek-Kumar98217/Assignement.git
cd Assignement
```

### 2. Create a virtual environment and install python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate the final `predictions` in the final-prediction directory for both training data and test data

```bash
# Ensure you are in the project root with data/test.csv available
python3 predict.py
```

This will load the saved XGBoost pipelines, process the test data, apply the Decision Engine, and output `final-prediction/test_predictions.csv` and `final-prediction/train_predictions.csv`.

### 4. Run the realtime prediction demo

#### 1. Backend

```bash
cd demo/backend
fastapi dev main.py
```

#### 2. Frontend

```bash
cd demo/frontend
npm install
npm run dev
```

Once running, simply open `http://localhost:3000` in your browser to test the end-to-end system and view the real-time uncertainty UI.

## Architecture & Model Selection

### The Ablation Study

Extensive architecture searches were conducted across multiple paradigms (found in the `models/` directory) to find the optimal deployment model:

- **Tree-Based Models:** XGBoost (Parallel & Sequential predictions)

`Tested with considering the intensity prediction as both regression and classification problem`

- **Deep Learning:** PyTorch Dual-Target MLP

`Tested two models with textual feature modeled using TF-IDF and Embedding using light weight sentence transformer(All-MiniLM-L6-v2)`

- **Feature Sets:** Text-Only vs. Text + Metadata.

### The Findings (The Data Ceiling)

Testing across models (with evaluation scripts mapped in `error-analysis/` and `eval.py`) revealed a hard accuracy ceiling across all architectures. The Deep Learning Transformer and PyTorch MLP models(smaller models were considered due to the constraint of the assignment) performed identically to, or slightly worse than, traditional tree-based models.(see `error-analysis/` directory for detailed error analysis)

Furthermore, the ablation study showed no statistically significant performance leap between the Text-Only and Text + Metadata models.

**Conclusion:** The bottleneck is not algorithmic; it is the irreducible error (aleatoric uncertainty) inherent in subjective human emotional labeling without historical user baselines. It was observed that when using TF-IDF the number of features expanded to >316 feature while on using a embedding model the features expanded to 405, which for a dataset of 1200 samples is too high and leads to overfitting.

### The Chosen Model: Parallel XGBoost Classifier

Because complex deep learning models yielded no performance gain, the Parallel XGBoost Classifier with TF-IDF Vectorization (`xgboost_clf`) was selected for production.

- **Why:** It operates with sub-millisecond latency, requires negligible RAM, boots instantly via `joblib`, and natively handles sparse text matrices alongside dense metadata without the overhead of PyTorch execution in a lightweight web environment. It also offered the most consistent performance accross multiple training runs`(Emotional state f1-score: 56-57%, Intensity f1-score: 22-23%)`

## Decision Engine

Because the dataset exhibits high aleatoric uncertainty, the system is designed with a **Safety-First Product Philosophy**. We do not blindly trust low-confidence ML predictions to prescribe heavy interventions.

Instead, predictions are passed into a deterministic Decision Engine (present in `predict.py` and `demo/backend/main.py`) that evaluates:

- Predicted Emotional State
- Predicted Emotional Intensity
- Model Confidence & Uncertainty Flags

- **Why:** The decision engine was kept deterministic, due to absence of any data regarding the problem. Considering that the problem is a a "make or break" feature it would be better to model it properly after data is collected and the user's response to the interventions are recorded.

### Core Logic Principles:

- **The Uncertainty Fallback:** If the model's confidence drops below 50%, the `uncertain_flag` is triggered. The engine defaults to a safe, low-friction prompt (`prompt_manual_reflection` | `end_of_day`) rather than risking an incorrect or tone-deaf intervention.
- **Biological / Severity Context:** A restless/overwhelmed state with high intensity (>= 4) triggers an immediate physical response (`suggest_short_break` | `immediate`). A milder presentation (< 4) triggers a softer intervention (`review_daily_schedule` | `next_transition`).
- **Minimizing Notification Fatigue:** If the user is in a stable, positive state (`calm`, `focused`, `neutral`) with low intensity, the engine prescribes `no_action_needed` to preserve user trust and avoid annoying disruptions.

### Project Structure

- `artifacts/`: Contains all the plots and tables for ease of analysis or errors and model performance. It is model seprated and populated using the error-analysis scripts.
- `data/`: Contains the raw dataset for training and testing.
- `demo/`: The interactive full-stack application containing `frontend/` (Next.js) and `backend/main.py` (FastAPI utilizing lifespan context managers for memory-efficient model loading).
- `error-analysis/`: Contains the scripts to analyze the error distribution and confusion matrix for deep error distribution analysis.
- `models/`: Defines some of the models tested, include thier definations, training scripts, cross-validation scripts and saving scripts.
- `predictions-test/`: All the predictions generated during the cross validation predictions of the models.
- `saved-models`: All the trained models are stores in this directory with thier preprocessors
- `predict.py`: The lightweight batch-inference script for generating the final CSV deliverables.
- `preprocess.py`: Contains the preprocessor definitions (TF-IDF for text, StandardScaling, encoders) to assure identical processing logic across training and inference with zero data-leakage.
- `eval.py`: Defines the function to generate the plots and metrics for evaluation the model performance.
