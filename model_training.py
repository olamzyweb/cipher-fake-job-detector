"""
=============================================================
FILE: model_training.py
STEP 2 OF THE AI PIPELINE — BUILDING & TRAINING THE BI-LSTM MODEL

PURPOSE:
    This file builds and trains the actual AI brain of the system.
    It reads the EMSCAD dataset, trains a Bi-LSTM neural network,
    then saves the trained model so app.py can use it.

SIMPLE ANALOGY:
    Think of this like training a new security guard.
    
    1. You show the guard thousands of REAL job ads and FAKE job ads.
    2. The guard studies the patterns (what makes each type different).
    3. After training, the guard can now spot new fake ads on their own.
    4. We "save" the trained guard's knowledge to a file (the .keras model).
    5. Later, app.py "loads" the guard when a user visits the website.

KEY CONCEPTS:
    - Word Embeddings : Converting words to meaningful number vectors
    - Bi-LSTM         : Neural network that reads text both forwards AND backwards
    - SMOTE           : Technique to balance the dataset (fix 95% real / 5% fake)
    - Sigmoid Output  : Converts model output to a 0-1 probability score
=============================================================
"""

# ---- STANDARD LIBRARY ----
import os
import json
import pickle

# ---- DATA SCIENCE LIBRARIES ----
import numpy as np
import pandas as pd

# ---- OUR OWN PREPROCESSOR ----
from preprocess import TextPreprocessor

# ---- SCIKIT-LEARN ----
# Tools for splitting data and evaluating the model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---- IMBALANCED-LEARN (SMOTE) ----
# SMOTE = Synthetic Minority Over-sampling Technique
# Fixes the problem that only ~5% of jobs are fake
from imblearn.over_sampling import SMOTE

# ---- TENSORFLOW / KERAS ----
# The deep learning framework we use to build the neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,        # Converts word IDs to vectors
    Bidirectional,    # Wraps LSTM to read forwards AND backwards
    LSTM,             # The core memory unit of our network
    Dense,            # Fully connected layer
    Dropout,          # Prevents overfitting (memorising rather than learning)
    GlobalMaxPooling1D  # Extracts the most important features
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ---- CONFIGURATION ----
# These settings control how the model is built and trained.
# Think of them as the "recipe" for training.

MAX_VOCAB_SIZE = 20000    # Maximum number of unique words to track
                          # Words beyond this are treated as "unknown"

MAX_SEQUENCE_LENGTH = 300 # Maximum number of words per job posting
                          # Shorter texts get padded; longer texts get cut

EMBEDDING_DIM = 100       # How many numbers represent each word
                          # 100 dimensions captures rich word meaning

LSTM_UNITS = 64           # How many "memory cells" the LSTM has
                          # More units = smarter but slower to train

DENSE_UNITS = 32          # Size of the fully-connected layer after LSTM

DROPOUT_RATE = 0.3        # 30% of neurons randomly switched off during training
                          # This forces the model to not memorise the training data

BATCH_SIZE = 32           # How many examples to process at once
                          # Larger = faster training, more RAM needed

EPOCHS = 20               # Maximum training rounds
                          # EarlyStopping will stop before this if needed

VALIDATION_SPLIT = 0.15   # 15% of training data used for validation
                          # Helps us monitor training without touching test data


class FakeJobModelTrainer:
    """
    This class handles the complete training pipeline:
    
    1. Load data from CSV
    2. Preprocess text
    3. Tokenize and pad sequences
    4. Handle class imbalance with SMOTE
    5. Build the Bi-LSTM model
    6. Train the model
    7. Evaluate performance
    8. Save the model and tokenizer
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.tokenizer = None
        self.model = None
        
        # Create directory for saving models
        os.makedirs('models', exist_ok=True)
    
    # =========================================================
    # STEP 1: LOAD AND PREPARE THE DATASET
    # =========================================================
    
    def load_data(self, csv_path='data/fake_job_postings.csv'):
        """
        Load the EMSCAD dataset from a CSV file.
        
        The EMSCAD (Employment Scam Aegean Dataset) is a real research dataset
        of 17,880 job postings, of which ~800 are fraudulent.
        
        Download it from:
        https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
        
        Place the downloaded CSV in the 'data/' folder as:
        data/fake_job_postings.csv
        
        Dataset columns we use:
            title            : Job title
            description      : Full job description
            company_profile  : About the company
            requirements     : Job requirements
            benefits         : What the company offers
            fraudulent       : 0 = Real, 1 = Fake (this is what we predict)
        """
        print(f"\n{'='*60}")
        print("STEP 1: LOADING DATASET")
        print(f"{'='*60}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"\n❌ Dataset not found at '{csv_path}'\n"
                f"\nHow to fix this:\n"
                f"1. Go to: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction\n"
                f"2. Click 'Download' (you need a free Kaggle account)\n"
                f"3. Extract the ZIP file\n"
                f"4. Copy 'fake_job_postings.csv' to the 'data/' folder\n"
                f"5. Run this script again\n"
            )
        
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} job postings from dataset")
        print(f"   Columns found: {list(df.columns)}")
        
        # Check that the required columns exist
        required_cols = ['fraudulent']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset!")
        
        return df
    
    # =========================================================
    # STEP 2: PREPROCESS ALL TEXT
    # =========================================================
    
    def preprocess_data(self, df):
        """
        Run the full text cleaning pipeline on every job posting.
        
        This calls our TextPreprocessor from preprocess.py to:
        - Combine all text fields (title + description + company + requirements + benefits)
        - Clean the combined text (lowercase, remove symbols, lemmatize, etc.)
        - Extract the labels (0 or 1)
        """
        print(f"\n{'='*60}")
        print("STEP 2: PREPROCESSING TEXT")
        print(f"{'='*60}")
        
        X_text, y = self.preprocessor.prepare_dataset(df)
        
        print(f"✅ Preprocessing complete")
        print(f"   Example cleaned text: '{X_text[0][:100]}...'")
        
        return X_text, y
    
    # =========================================================
    # STEP 3: TOKENIZE AND CONVERT TO NUMBER SEQUENCES
    # =========================================================
    
    def tokenize_and_pad(self, X_text):
        """
        Convert words to numbers so the neural network can process them.
        
        WHY WE NEED THIS:
        Neural networks work with numbers, not words.
        We need to convert "urgent apply now" → [342, 17, 55]
        
        HOW IT WORKS:
        1. Tokenizer builds a word-to-number dictionary (vocabulary)
           e.g., {'urgent': 1, 'apply': 2, 'now': 3, 'salary': 4, ...}
           
        2. Each job posting becomes a list of numbers
           "urgent apply now" → [1, 2, 3]
        
        3. Padding: All sequences must be the SAME length for the neural network.
           Short sequences get zeros added at the start:
           [1, 2, 3] → [0, 0, 0, 0, 0, ..., 1, 2, 3]  (length = MAX_SEQUENCE_LENGTH)
           
        4. Truncating: Long sequences are cut off at MAX_SEQUENCE_LENGTH.
        """
        print(f"\n{'='*60}")
        print("STEP 3: TOKENIZATION & SEQUENCE PADDING")
        print(f"{'='*60}")
        
        # Build the vocabulary from all training text
        # num_words = only keep the top MAX_VOCAB_SIZE most common words
        # oov_token = '<OOV>' is used for words not in vocabulary (Out Of Vocabulary)
        self.tokenizer = Tokenizer(
            num_words=MAX_VOCAB_SIZE,
            oov_token='<OOV>'    # Unknown words become this token
        )
        
        # Learn the vocabulary from all job postings
        self.tokenizer.fit_on_texts(X_text)
        
        vocab_size = min(len(self.tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
        print(f"✅ Vocabulary built: {vocab_size} unique words")
        
        # Convert text to sequences of numbers
        # "urgent apply now" → [342, 17, 55]
        sequences = self.tokenizer.texts_to_sequences(X_text)
        
        # Pad/truncate all sequences to MAX_SEQUENCE_LENGTH
        # padding='pre'  = Add zeros at the BEGINNING of short sequences
        # truncating='post' = Cut the END of long sequences
        X_padded = pad_sequences(
            sequences,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding='pre',
            truncating='post'
        )
        
        print(f"✅ Sequences padded to length {MAX_SEQUENCE_LENGTH}")
        print(f"   Input shape for model: {X_padded.shape}")
        
        return X_padded, vocab_size
    
    # =========================================================
    # STEP 4: HANDLE CLASS IMBALANCE WITH SMOTE
    # =========================================================
    
    def handle_class_imbalance(self, X, y):
        """
        Fix the class imbalance problem using SMOTE.
        
        THE PROBLEM:
        In the EMSCAD dataset:
            - ~95% of jobs are REAL
            - ~5% of jobs are FAKE
        
        Without fixing this, the AI would just predict "REAL" for everything
        and be "95% accurate" — but completely useless for detecting fraud!
        
        THE SOLUTION — SMOTE:
        SMOTE = Synthetic Minority Over-sampling Technique
        
        It creates NEW, SYNTHETIC fake job examples by:
        1. Finding similar existing fake job examples
        2. Creating new examples that are "between" them mathematically
        
        Result: A balanced dataset where 50% are real, 50% are fake.
        Now the model must actually learn BOTH classes properly.
        
        Note: SMOTE works in vector space, so it operates on the
        padded sequence numbers, not on the original text.
        """
        print(f"\n{'='*60}")
        print("STEP 4: HANDLING CLASS IMBALANCE WITH SMOTE")
        print(f"{'='*60}")
        
        print(f"Before SMOTE:")
        print(f"   Real jobs: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
        print(f"   Fake jobs: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
        
        # Apply SMOTE to create synthetic fake job examples
        # random_state=42 makes results reproducible
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"\nAfter SMOTE:")
        print(f"   Real jobs: {sum(y_balanced == 0)} ({sum(y_balanced == 0)/len(y_balanced)*100:.1f}%)")
        print(f"   Fake jobs: {sum(y_balanced == 1)} ({sum(y_balanced == 1)/len(y_balanced)*100:.1f}%)")
        print(f"✅ Dataset balanced: {len(X_balanced)} total examples")
        
        return X_balanced, y_balanced
    
    # =========================================================
    # STEP 5: BUILD THE BI-LSTM NEURAL NETWORK
    # =========================================================
    
    def build_model(self, vocab_size):
        """
        Build the Bi-LSTM neural network architecture.
        
        ARCHITECTURE EXPLANATION (Layer by Layer):
        
        ┌─────────────────────────────────────────────────┐
        │ INPUT: Sequence of word numbers                  │
        │ e.g., [0, 0, 342, 17, 55, 98, ...]             │
        │ Shape: (batch_size, MAX_SEQUENCE_LENGTH)         │
        └────────────────────┬────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │ LAYER 1: Embedding                               │
        │                                                  │
        │ Converts each word number to a 100-dim vector.   │
        │ Similar words get similar vectors.               │
        │                                                  │
        │ Input:  [342, 17, 55]                           │
        │ Output: [[0.2,-0.5,...], [0.8,0.1,...], ...]   │
        │                                                  │
        │ Shape: (batch, seq_len, EMBEDDING_DIM)           │
        └────────────────────┬────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │ LAYER 2: Bidirectional LSTM                      │
        │                                                  │
        │ NORMAL LSTM: reads text left-to-right            │
        │ "The salary is too good to be true"             │
        │  →  →  →  →  →  →  →  →  →                    │
        │                                                  │
        │ BIDIRECTIONAL: ALSO reads right-to-left          │
        │ "true be to good too is salary The"             │
        │  ←  ←  ←  ←  ←  ←  ←  ←  ←                    │
        │                                                  │
        │ Combining both directions gives better context   │
        │ understanding — the model sees the FULL picture. │
        └────────────────────┬────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │ LAYER 3: GlobalMaxPooling1D                      │
        │                                                  │
        │ Picks the MOST IMPORTANT feature from the LSTM   │
        │ output sequence. Reduces to a single vector.     │
        └────────────────────┬────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │ LAYER 4: Dense (32 units) + Dropout              │
        │                                                  │
        │ A fully connected layer that combines features.  │
        │ Dropout randomly turns off 30% of connections   │
        │ during training to prevent memorisation.         │
        └────────────────────┬────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │ LAYER 5: Dense (1 unit, Sigmoid activation)      │
        │                                                  │
        │ OUTPUT: A single number between 0 and 1          │
        │         0.0 = Definitely REAL job                │
        │         1.0 = Definitely FAKE job                │
        │         0.7 = 70% likely to be fake              │
        └─────────────────────────────────────────────────┘
        """
        print(f"\n{'='*60}")
        print("STEP 5: BUILDING BI-LSTM NEURAL NETWORK")
        print(f"{'='*60}")
        
        model = Sequential([
            
            # LAYER 1: Word Embedding
            # vocab_size = how many unique words we have
            # EMBEDDING_DIM = how many numbers represent each word (100)
            # input_length = max words per posting (300)
            # mask_zero=True = Tell the model that 0 (padding) is not a real word
            Embedding(
                input_dim=vocab_size,
                output_dim=EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
                mask_zero=True
            ),
            
            # LAYER 2: Bidirectional LSTM
            # return_sequences=True = return the full sequence for pooling
            # This allows GlobalMaxPooling1D to work on the full output
            Bidirectional(
                LSTM(
                    units=LSTM_UNITS,        # 64 memory cells
                    return_sequences=True,   # Return full output sequence
                    dropout=0.2,             # 20% input dropout
                    recurrent_dropout=0.2    # 20% recurrent dropout
                )
            ),
            
            # LAYER 3: Global Max Pooling
            # Picks the maximum value from each feature dimension
            # Extracts the "most important signals" from the LSTM output
            GlobalMaxPooling1D(),
            
            # LAYER 4: Dense Layer + Dropout
            # ReLU activation: makes the network non-linear (able to learn complex patterns)
            Dense(DENSE_UNITS, activation='relu'),
            Dropout(DROPOUT_RATE),  # Regularisation: prevents overfitting
            
            # LAYER 5: Output Layer
            # Sigmoid activation squashes output to [0, 1] probability
            # 1 neuron = binary classification (real vs fake)
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        # optimizer='adam'     = Adaptive learning rate optimiser (industry standard)
        # loss='binary_crossentropy' = Loss function for binary classification
        # metrics=['accuracy'] = Track accuracy during training
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Print the model summary (shows each layer and parameter count)
        model.summary()
        
        self.model = model
        print("✅ Model built successfully")
        return model
    
    # =========================================================
    # STEP 6: TRAIN THE MODEL
    # =========================================================
    
    def train_model(self, X_train, y_train):
        """
        Train the model on the prepared data.
        
        WHAT TRAINING MEANS:
        The model starts with random weights (like a blank brain).
        During each epoch (training round), it:
        1. Processes batches of BATCH_SIZE examples
        2. Makes a prediction for each example
        3. Compares its prediction to the real answer (loss)
        4. Adjusts its weights to reduce the loss (backpropagation)
        5. Repeats until validation performance stops improving
        
        CALLBACKS (Automatic Training Helpers):
        - EarlyStopping: Stops training if validation loss doesn't improve
          for 3 consecutive epochs. Prevents overfitting.
        - ModelCheckpoint: Saves the BEST version of the model automatically.
          If epoch 8 is the best, that's what gets saved — not epoch 20.
        """
        print(f"\n{'='*60}")
        print("STEP 6: TRAINING THE MODEL")
        print(f"{'='*60}")
        print(f"Training on {len(X_train)} examples")
        print(f"Batch size: {BATCH_SIZE}, Max epochs: {EPOCHS}")
        
        # EarlyStopping: stop training if val_loss doesn't improve for 3 epochs
        early_stopping = EarlyStopping(
            monitor='val_loss',    # Watch validation loss
            patience=3,            # Stop after 3 epochs without improvement
            restore_best_weights=True,  # Restore weights from best epoch
            verbose=1
        )
        
        # ModelCheckpoint: save only weights to avoid Keras native format issues
        checkpoint = ModelCheckpoint(
            filepath='models/fake_job_model.weights.h5',  # Save best weights
            monitor='val_accuracy',                        # Save based on validation accuracy
            save_best_only=True,                           # Only save if it's an improvement
            save_weights_only=True,                        # Save weights, not full model
            verbose=1
        )
        
        # Train the model
        # validation_split = use 15% of training data to monitor performance
        history = self.model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        print(f"\n✅ Training complete!")
        print(f"   Final training accuracy:   {history.history['accuracy'][-1]:.4f}")
        print(f"   Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return history
    
    # =========================================================
    # STEP 7: EVALUATE THE MODEL ON TEST DATA
    # =========================================================
    
    def evaluate_model(self, X_test, y_test):
        """
        Test the model on data it has NEVER seen before.
        
        This tells us how well the model will perform in the real world.
        
        METRICS EXPLAINED:
        - Accuracy  : Overall correct predictions / total predictions
        - Precision : Of all jobs flagged as fake, how many were actually fake?
                      High precision = low false alarms
        - Recall    : Of all actual fake jobs, how many did we catch?
                      High recall = low misses (important for safety!)
        - F1 Score  : Harmonic mean of precision and recall
                      Best single measure of model performance
        
        CONFUSION MATRIX:
                         Predicted Real  Predicted Fake
        Actually Real  │     TN         │      FP       │
        Actually Fake  │     FN         │      TP       │
        
        TN = True Negative  (correctly said "real")
        TP = True Positive  (correctly said "fake")
        FP = False Positive (wrongly flagged real job as fake)
        FN = False Negative (missed a fake job — most dangerous!)
        """
        print(f"\n{'='*60}")
        print("STEP 7: EVALUATING MODEL PERFORMANCE")
        print(f"{'='*60}")
        
        # Get raw probability predictions (0.0 to 1.0)
        y_pred_proba = self.model.predict(X_test, verbose=0)
        
        # Convert probabilities to class labels
        # Threshold = 0.5: anything above 0.5 is classified as FAKE
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 MODEL PERFORMANCE ON TEST SET:")
        print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nDetailed Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Real Job (0)', 'Fake Job (1)']
        ))
        
        print(f"Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"   True Negatives  (correct real): {cm[0][0]}")
        print(f"   False Positives (wrong alarm):  {cm[0][1]}")
        print(f"   False Negatives (missed fakes): {cm[1][0]}  ← Most important to minimise!")
        print(f"   True Positives  (caught fakes): {cm[1][1]}")
        
        # Save evaluation results to a JSON file for app.py to display
        results = {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'total_test_examples': len(y_test)
        }
        with open('models/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Evaluation complete. Results saved to models/evaluation_results.json")
        return results
    
    # =========================================================
    # STEP 8: SAVE THE TRAINED MODEL AND TOKENIZER
    # =========================================================
    
    def save_artifacts(self):
        """
        Save the trained model and tokenizer to disk.
        
        WHY WE SAVE:
        Training takes time (minutes). We don't want to retrain every time
        someone visits the website. Instead, we save the trained model
        and load it instantly when the website starts.
        
        What we save:
        - models/fake_job_model.keras  → The neural network with all its weights
        - models/tokenizer.pkl         → The word-to-number vocabulary mapping
        
        When app.py starts, it loads these two files.
        """
        print(f"\n{'='*60}")
        print("STEP 8: SAVING MODEL AND TOKENIZER")
        print(f"{'='*60}")
        
        # Save model (already saved as best during training via ModelCheckpoint)
        # Save again explicitly as backup
        self.model.save('models/fake_job_model.keras')
        print("✅ Model saved: models/fake_job_model.keras")
        
        # Save tokenizer (the word vocabulary)
        # We use pickle to save the Python object to a file
        with open('models/tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print("✅ Tokenizer saved: models/tokenizer.pkl")
        
        # Save model configuration (useful for documentation)
        config = {
            'max_vocab_size': MAX_VOCAB_SIZE,
            'max_sequence_length': MAX_SEQUENCE_LENGTH,
            'embedding_dim': EMBEDDING_DIM,
            'lstm_units': LSTM_UNITS,
        }
        with open('models/model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Config saved: models/model_config.json")
    
    # =========================================================
    # MAIN TRAINING PIPELINE — RUNS ALL STEPS IN ORDER
    # =========================================================
    
    def run_full_pipeline(self, csv_path='data/fake_job_postings.csv'):
        """
        Run the complete training pipeline from raw CSV to saved model.
        
        Call this function to train the model from scratch.
        """
        print("\n" + "="*60)
        print("  FAKE JOB DETECTION — FULL TRAINING PIPELINE")
        print("  Olagunju Basheer Olaniyi | 220303010022")
        print("="*60)
        
        # STEP 1: Load dataset
        df = self.load_data(csv_path)
        
        # STEP 2: Preprocess text
        X_text, y = self.preprocess_data(df)
        
        # STEP 3: Tokenize and pad sequences
        X_padded, vocab_size = self.tokenize_and_pad(X_text)
        
        # STEP 4: Split into train/test sets BEFORE applying SMOTE
        # We only apply SMOTE to training data — test data must stay real/original
        # test_size=0.2 = 20% of data reserved for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, y,
            test_size=0.2,
            random_state=42,
            stratify=y  # Ensure both splits have same class ratio
        )
        print(f"\nData split: {len(X_train)} training, {len(X_test)} testing")
        
        # STEP 4b: Apply SMOTE only to training data
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # STEP 5: Build the model
        self.build_model(vocab_size)
        
        # STEP 6: Train the model
        self.train_model(X_train_balanced, y_train_balanced)
        
        # STEP 7: Evaluate on untouched test data
        self.evaluate_model(X_test, y_test)
        
        # STEP 8: Save everything
        self.save_artifacts()
        
        print("\n" + "="*60)
        print("  ✅ TRAINING COMPLETE!")
        print("  You can now run: python app.py")
        print("="*60)


# ---- ENTRY POINT ----
# This block runs when you execute: python model_training.py
if __name__ == '__main__':
    trainer = FakeJobModelTrainer()
    trainer.run_full_pipeline()
