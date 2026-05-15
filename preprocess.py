"""
=============================================================
FILE: preprocess.py
STEP 1 OF THE AI PIPELINE — DATA CLEANING & PREPARATION

PURPOSE:
    Raw job posting text is messy. Before the AI can learn from it,
    we must clean and standardise it. This file does that job.

WHAT THIS FILE DOES (in simple English):
    Imagine you receive thousands of job adverts.
    Some use ALL CAPS. Some have weird symbols. Some have HTML tags.
    The AI cannot learn from messy text — just like a student cannot
    study from a book full of typos and scribbles.

    This file "washes" the text so it is clean and uniform.

NLKT LIBRARY CONCEPTS USED:
    - Tokenization  : Splitting a sentence into individual words
    - Stopwords     : Removing common words like "the", "is", "at" 
                      that carry no meaning
    - Lemmatization : Converting words to their base form
                      e.g. "running" → "run", "jobs" → "job"
=============================================================
"""

# ---- IMPORTS ----
# re = Regular Expressions. Used to find/remove patterns in text.
import re

# pandas = Our data table tool. Like Excel but for Python.
import pandas as pd

# numpy = Math operations on arrays of numbers.
import numpy as np

# nltk = Natural Language Toolkit. The core NLP library.
import nltk

# WordNetLemmatizer = Converts words to their base/dictionary form.
from nltk.stem import WordNetLemmatizer

# stopwords = Common words (the, is, at, a, an) that carry no meaning.
from nltk.corpus import stopwords

# word_tokenize = Splits text into individual words (tokens).
from nltk.tokenize import word_tokenize

# Download required NLTK resources (only needed once)
# These are like "dictionaries" the NLP library uses
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("NLTK resources ready.")


class TextPreprocessor:
    """
    A class that handles all text cleaning operations.
    
    Think of this as a "text washing machine" — dirty text goes in,
    clean, standardised text comes out.
    """
    
    def __init__(self):
        # Load the English stop words list
        # These are words like "the", "a", "is" that carry no meaning
        self.stop_words = set(stopwords.words('english'))
        
        # The lemmatizer converts words to their base form
        # "running" → "run", "companies" → "company"
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        STEP 1: BASIC TEXT CLEANING
        
        Removes noise from text so the AI only sees meaningful words.
        
        Example:
            Input:  "URGENT!!! Apply NOW -- Earn $5,000/week!! 🚀"
            Output: "urgent apply now earn week"
        
        What happens step by step:
            1. Convert to lowercase      → "urgent!!! apply now..."
            2. Remove HTML tags          → removes <b>, <p>, etc.
            3. Remove special chars      → removes !, $, /, #
            4. Remove numbers            → removes 5000, 50, etc.
            5. Remove extra whitespace   → "urgent  apply" → "urgent apply"
        """
        # If no text is provided, return empty string
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Step 1a: Convert everything to lowercase
        # "URGENT" and "urgent" are the same word — make them match
        text = text.lower()
        
        # Step 1b: Remove HTML tags (some job sites use HTML in descriptions)
        # e.g., "<b>urgent</b>" → "urgent"
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Step 1c: Remove special characters and punctuation
        # Keep only letters (a-z) and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Step 1d: Remove extra whitespace (multiple spaces → single space)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text):
        """
        STEP 2: ADVANCED NLP PREPROCESSING
        
        After basic cleaning, we do NLP-specific processing:
        - Remove stop words (meaningless common words)
        - Lemmatize (convert to base word form)
        
        Example:
            Input:  "the running companies are hiring workers immediately"
            Output: "run company hire worker immediately"
            
        Why this matters:
            "run", "running", "runner", "ran" all mean similar things.
            By converting to "run", the AI treats them all the same —
            making the model smarter with less data.
        """
        # First, do basic cleaning
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # Step 2a: Tokenize (split into individual words)
        # "apply now for jobs" → ["apply", "now", "for", "jobs"]
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        
        # Step 2b: Remove stop words and lemmatize
        processed_tokens = []
        for token in tokens:
            # Skip stop words (the, is, at, a, an, etc.)
            # They carry no fraud-detection meaning
            if token not in self.stop_words and len(token) > 2:
                # Convert to base form: "running" → "run"
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)
        
        # Step 2c: Join back into a single string
        return ' '.join(processed_tokens)
    
    def combine_fields(self, row):
        """
        STEP 3: COMBINE ALL JOB POSTING FIELDS INTO ONE TEXT
        
        A job posting has multiple fields:
        - Job Title
        - Job Description
        - Company Profile
        - Requirements
        - Benefits
        
        We combine them all into one big text block for the AI to analyse.
        This gives the model the full picture.
        
        Example:
            title:       "Data Entry Clerk"
            description: "Earn $3000/week from home..."
            company:     "" (empty)
            
            combined: "data entry clerk earn week home"
        """
        fields = [
            str(row.get('title', '') or ''),
            str(row.get('description', '') or ''),
            str(row.get('company_profile', '') or ''),
            str(row.get('requirements', '') or ''),
            str(row.get('benefits', '') or ''),
        ]
        
        # Join all fields with a space separator
        combined = ' '.join(fields)
        
        # Run through full preprocessing pipeline
        return self.preprocess_text(combined)
    
    def prepare_dataset(self, df):
        """
        STEP 4: PREPARE THE FULL DATASET FOR TRAINING
        
        Takes the raw CSV data and prepares it for the AI model:
        1. Combine all text fields
        2. Clean and preprocess the combined text
        3. Extract the labels (0 = real, 1 = fake)
        
        Returns:
            X = list of cleaned text strings (the features)
            y = list of labels (0 or 1)
        """
        print(f"Preparing dataset with {len(df)} rows...")
        
        # Apply text combination and cleaning to every row
        # This may take a minute for large datasets
        X = df.apply(self.combine_fields, axis=1).tolist()
        
        # Extract labels: 'fraudulent' column (0 = real, 1 = fake)
        y = df['fraudulent'].values
        
        print(f"Dataset prepared. Texts: {len(X)}, Labels: {len(y)}")
        print(f"Fake jobs: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"Real jobs: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
        
        return X, y


# ---- STANDALONE TEST ----
# Run this file directly to test the preprocessor
if __name__ == '__main__':
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "URGENT!!! Apply NOW -- Earn $5,000/week!! No experience needed!",
        "We are looking for a skilled Software Engineer with 3+ years of Python experience.",
        "CONGRATULATIONS! You have been SELECTED for this EXCLUSIVE opportunity! Pay $50 registration fee!"
    ]
    
    print("\n=== PREPROCESSOR TEST ===\n")
    for i, text in enumerate(test_texts, 1):
        result = preprocessor.preprocess_text(text)
        print(f"Original {i}: {text}")
        print(f"Cleaned  {i}: {result}")
        print()
