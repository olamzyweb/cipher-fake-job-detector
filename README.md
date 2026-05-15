# 🛡️ FAKE JOB POSTING DETECTION SYSTEM
### LASUSTECH Final Year Project 2026
**Student:** Olagunju Basheer Olaniyi | **Matric:** 220303010022  
**Supervisor:** Mr. Adewale Yekini | **Department:** Computer Science, Basic Sciences

---

## 📌 What This System Does

This is a **real AI-powered web application** that uses a **Bi-LSTM (Bidirectional Long Short-Term Memory) Neural Network** trained on the EMSCAD dataset to detect whether a job advertisement is fraudulent or legitimate.

It is NOT a simple rule-based system. It is a **genuine deep learning model** trained on real data.

---

## 📁 File Structure

```
fake_job_detector/
│
├── preprocess.py          ← STEP 1: Cleans and prepares text for the AI
├── model_training.py      ← STEP 2: Builds and trains the Bi-LSTM model
├── app.py                 ← STEP 3: Runs the website and serves predictions
├── test_system.py         ← Verifies everything works correctly
├── requirements.txt       ← Python libraries to install
├── README.md              ← This file
│
├── data/                  ← PUT THE DATASET HERE
│   └── fake_job_postings.csv    (download from Kaggle)
│
├── models/                ← SAVED MODEL (created after training)
│   ├── fake_job_model.keras     (the trained neural network)
│   ├── tokenizer.pkl            (the word vocabulary)
│   ├── model_config.json        (model settings)
│   └── evaluation_results.json  (test accuracy results)
│
└── templates/             ← Web pages
    ├── index.html               (main analysis page)
    └── about.html               (project info page)
```

---

## 🚀 HOW TO RUN — Step by Step

Follow these steps **in order**. Do not skip any step.

---

### ✅ STEP 1 — Install Python

If you don't have Python installed:

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow **"Download Python"** button
3. Run the installer
4. ⚠️ **CRITICAL:** On the first screen, tick **"Add Python to PATH"** before clicking Install
5. Restart your computer

Verify by opening **Command Prompt** (Windows) or **Terminal** (Mac/Linux) and typing:
```bash
python --version
```
Expected output: `Python 3.9.x` or newer

---

### ✅ STEP 2 — Download and Extract This Project

If you received a ZIP file:
1. Right-click the ZIP → **Extract All** (Windows) or double-click (Mac)
2. Note the folder location (e.g., `C:\Users\YourName\Downloads\fake_job_detector`)

---

### ✅ STEP 3 — Open Command Prompt in the Project Folder

**Windows:**
1. Open File Explorer and navigate to the `fake_job_detector` folder
2. Click the address bar at the top, type `cmd`, press Enter

OR:
1. Press `Windows + R`, type `cmd`, press Enter
2. Type: `cd C:\Users\YourName\Downloads\fake_job_detector`

**Mac/Linux:**
1. Open Terminal
2. Type: `cd /Users/YourName/Downloads/fake_job_detector`

---

### ✅ STEP 4 — Install Required Libraries

Type this command and press Enter:
```bash
pip install -r requirements.txt
```

This will install:
- **TensorFlow** — the deep learning framework
- **Flask** — the web framework
- **NLTK** — natural language processing tools
- **scikit-learn** — machine learning utilities
- **imbalanced-learn** — SMOTE for handling imbalanced data
- **pandas, numpy** — data manipulation

⏳ This may take **5-10 minutes** depending on your internet speed.

---

### ✅ STEP 5 — Download the Training Dataset

The AI model is trained on the **EMSCAD (Employment Scam Aegean Dataset)**.

1. Go to: **https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction**
2. Click **"Download"** (requires a free Kaggle account)
3. Extract the downloaded ZIP
4. Find the file named **`fake_job_postings.csv`**
5. Copy it into the **`data/`** folder inside your project:
   ```
   fake_job_detector/
   └── data/
       └── fake_job_postings.csv   ← place it here
   ```

---

### ✅ STEP 6 — Train the AI Model

This is the step that **teaches** the AI to detect fraud.

```bash
python model_training.py
```

What happens during training:
1. **Loads** the dataset (~17,880 job postings)
2. **Cleans** all text (lowercase, removes symbols, lemmatizes)
3. **Tokenizes** text → converts words to numbers
4. **Applies SMOTE** → balances the 95%/5% class imbalance
5. **Builds** the Bi-LSTM neural network
6. **Trains** for up to 20 epochs (EarlyStopping prevents overfitting)
7. **Evaluates** on 20% test data it has never seen
8. **Saves** the trained model to `models/fake_job_model.keras`

⏳ Training time: **10-30 minutes** on CPU, **2-5 minutes** with GPU.

You will see output like:
```
Epoch 1/20 - loss: 0.4521 - accuracy: 0.8234 - val_loss: 0.3812 - val_accuracy: 0.8756
Epoch 2/20 - loss: 0.3012 - accuracy: 0.8891 - val_loss: 0.2934 - val_accuracy: 0.9023
...
✅ TRAINING COMPLETE!
```

---

### ✅ STEP 7 — Run the System Tests

Verify everything is working:

```bash
python test_system.py
```

All tests should show `✅ PASS`.

---

### ✅ STEP 8 — Start the Web Application

```bash
python app.py
```

You should see:
```
============================================================
  FAKE JOB POSTING DETECTION SYSTEM
  By: Olagunju Basheer Olaniyi (220303010022)
  LASUSTECH Final Year Project 2026
============================================================
  Server starting at: http://127.0.0.1:5000
============================================================
✅ Model loaded successfully
✅ Tokenizer loaded successfully
✅ System ready for predictions!
```

---

### ✅ STEP 9 — Open in Browser

1. Open Chrome, Firefox, or Edge
2. Type in the address bar: **`http://127.0.0.1:5000`**
3. Press Enter

🎉 **The website is now live!**

---

## 🖥️ HOW TO USE THE WEBSITE

1. You will see a form with multiple fields
2. **Option A:** Click **"LOAD FAKE JOB EXAMPLE"** or **"LOAD REAL JOB EXAMPLE"** to test
3. **Option B:** Paste a real job posting you want to check
4. Click **"▶ RUN FRAUD ANALYSIS"**
5. The AI will process the text and show:
   - The **verdict** (Fraudulent / Legitimate)
   - A **fraud probability score** (0–100%)
   - **Advice** on what to do next

---

## 🧠 HOW THE AI PIPELINE WORKS

```
USER PASTES JOB TEXT
        │
        ▼ preprocess.py
┌─────────────────────────────┐
│  STEP 1: TEXT CLEANING      │
│  - lowercase                │
│  - remove HTML/symbols      │
│  - lemmatize words          │
│  - remove stop words        │
└──────────────┬──────────────┘
               │
               ▼ model_training.py
┌─────────────────────────────┐
│  STEP 2: TOKENIZATION       │
│  "urgent apply" → [342, 17] │
│  Pad to length 300          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  STEP 3: Bi-LSTM NETWORK    │
│  → Embedding (100-dim)      │
│  → Bidirectional LSTM (64)  │
│  → GlobalMaxPooling1D       │
│  → Dense (32) + Dropout     │
│  → Sigmoid output (0–1)     │
└──────────────┬──────────────┘
               │
               ▼ app.py
┌─────────────────────────────┐
│  STEP 4: PREDICTION         │
│  0.87 → "87% FRAUDULENT"   │
│  0.12 → "12% likely real"  │
└─────────────────────────────┘
```

---

## 🔧 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| `python` not recognised | Reinstall Python with "Add to PATH" ticked, then restart computer |
| `pip install` fails | Try `pip3 install -r requirements.txt` |
| Dataset not found error | Place `fake_job_postings.csv` in the `data/` folder |
| Model not found error | Run `python model_training.py` first |
| Website doesn't load | Keep the `python app.py` terminal open; don't close it |
| Port 5000 in use | Edit last line of `app.py` to use port `5001`, visit `localhost:5001` |
| TensorFlow install error | Try `pip install tensorflow-cpu` instead |

---

## 🛑 How to Stop the Application

Press `CTRL + C` in the terminal running `python app.py`.

---

## 📚 Research Basis

This implementation is grounded in findings from **30 academic papers (2020–2025)**:

- **Dataset:** EMSCAD — Vidros et al. (2021): 17,880 job postings, ~5% fraudulent
- **Architecture:** Bi-LSTM — superior to Naive Bayes, SVM for sequential text (Raza et al., 2022)
- **Imbalance Fix:** SMOTE — prevents the model from ignoring fake jobs (standard practice)
- **Embeddings:** 100-dimensional word vectors capture semantic meaning
- **Threshold:** 0.5 sigmoid output (probability > 50% → flagged as fraud)

---

*Made for LASUSTECH Final Year Project 2026 — Department of Computer Science*
