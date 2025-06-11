# Emotion Recognition System

A text-based emotion classification system that identifies emotions in text samples using machine learning techniques. This project classifies text into five distinct emotion categories: **Joy**, **Sadness**, **Anger**, **Fear**, and **Surprise** with **91.18% test accuracy**.

## Project Overview

This emotion recognition system leverages Logistic Regression with TF-IDF vectorisation and emotion-aware preprocessing to achieve strong performance in text-based emotion classification. The system demonstrates that classical machine learning approaches can deliver excellent results when enhanced with domain-specific insights.

### Key Improvements

- **Advanced Text Preprocessing**: Incorporated emotion-specific pattern recognition to better capture subtle signals in the text
- **Expanded TF-IDF Vectorisation**: Included bigrams and optimised TF-IDF parameters for richer feature extraction
- **Improved Model Configuration**: Used balanced class weights and tuned regularisation to handle class imbalance more effectively
- **Better Data Utilisation**: Split the data into 80/10/10 for training, validation, and testing to make the most of the available data

## Performance Results

### Overall Performance
- **Test Accuracy**: 91.18%
- **Validation Accuracy**: 91.02%
- **Training Strategy**: 80/10/10 stratified split (118,125 / 16,875 / 15,000 samples)

### Per-Class Performance (F1 Scores)

| Emotion | F1 Score | Precision | Recall | Support | Distribution |
|---------|----------|-----------|--------|---------|-------------|
| **Joy** | 0.96 | 0.97 | 0.94 | 4,100 | 27.33% |
| **Sadness** | 0.93 | 0.95 | 0.92 | 4,100 | 27.33% |
| **Anger** | 0.92 | 0.91 | 0.92 | 3,200 | 21.33% |
| **Fear** | 0.86 | 0.86 | 0.86 | 2,700 | 18.00% |
| **Surprise** | 0.77 | 0.70 | 0.86 | 900 | 6.00% |

**Weighted Average F1**: 0.91 | **Macro Average F1**: 0.89

The model performs well even on the minority class (Surprise) despite significant class imbalance.

## Dataset Characteristics

**Total Dataset**: 150,000 text samples from `Emotion-Dataset.csv`

| Emotion | Count | Percentage | Model Performance |
|---------|-------|------------|-------------------|
| Sadness | 41,000 | 27.33% | Very Good (F1: 0.93) |
| Joy | 41,000 | 27.33% | Excellent (F1: 0.96) |
| Anger | 32,000 | 21.33% | Very Good (F1: 0.92) |
| Fear | 27,000 | 18.00% | Good (F1: 0.86) |
| Surprise | 9,000 | 6.00% | Good (F1: 0.77) |

## Emotion Class Examples

The following examples illustrate how emotions can be expressed with subtlety and complexity in natural language. Each example is taken from the dataset and accurately reflects its emotional category. These are supplementary to the examples shown in the notebook.

### Anger
> *"i’m feeling are selfish because i know people have it worse but i can’t help being frustrated"*

### Fear
> *"i feel a bit hesitant to go back to my life in the city after everything that’s happened"*

### Joy
> *"i want to feel cool but being myself at the same time makes me genuinely happy"*

### Sadness
> *"i can feel it coming did you know that pain me more than i could say"*

### Surprise
> *"i think back over the last few months i feel pretty amazed by how quickly things changed"*

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Setup

The Jupyter notebook requires the following libraries:

```python
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
```

## Technical Architecture

### Text Preprocessing Implementation

The notebook implements an enhanced text preprocessing function that handles:

```python
def enhanced_preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Return empty string for non-string inputs
    
    # Lowercase conversion
    text = text.lower()
    
    # Handle contractions and negations with word boundaries
    text = re.sub(r"\bn't\b", " not", text)
    
    # Emotion-specific patterns with word boundaries
    negation_replacements = [
        (r"\bnot happy\b|\bunhappy\b", " sad"),
        (r"\bnot sad\b", " happy")
    ]
    
    intensity_replacements = [
        (r"\b(very|really|extremely|totally|so)\b", "intensifier")
    ]
    
    emotion_indicators = [
        (r"\b(lol|haha|hehe)\b", "joy_indicator"),
        (r"\b(omg|wow|whoa|what)\b", "surprise_indicator"),
        (r"\b(hate|angry|mad)\b", "anger_indicator"),
        (r"\b(worried|scared|afraid|nervous)\b", "fear_indicator"),
        (r"\b(sad|depressed|sorrow|grief)\b", "sadness_indicator")
    ]
    
    # Apply all replacements
    for pattern, replacement in negation_replacements + intensity_replacements + emotion_indicators:
        text = re.sub(pattern, replacement, text)
    
    # Handle punctuation
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    text = re.sub(r'!{2,}', ' strong_emotion ', text)
    text = re.sub(r'\?{2,}', ' strong_confusion ', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

### TF-IDF Vectorisation

The notebook configures TF-IDF vectorisation as follows:

```python
tfidf = TfidfVectorizer(
    max_features=15000,        # Keep the top 15k features
    ngram_range=(1, 2),        # Use unigrams and bigrams (single words and pairs)
    min_df=3,                  # Ignore words that appear in fewer than 3 documents
    max_df=0.95,               # Ignore words that appear in more than 95% of documents
    use_idf=True,              # Boost rare words, downplay common ones
    sublinear_tf=True,         # Scale word counts with log function
    strip_accents='unicode',   # Remove accents from characters
    analyzer='word'            # Split text into words
)
```

### Model Configuration

The Logistic Regression model is configured with:

```python
improved_model = LogisticRegression(
    C=2.0,                     # Regularisation strength
    penalty='l2',              # It essentially prevents overfitting
    solver='liblinear',        # Good for medium to large sized datasets
    max_iter=8000,             # More steps to converge
    class_weight='balanced',   # Fixes class imbalance
    random_state=42,           # For reproducibility
)
```

### Data Splitting Strategy

The notebook implements a stratified 80/10/10 split using scikit-learn:

```python
# First we split off 10% for the test set 
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_val_idx, test_idx = next(stratified_split.split(df['Text'], df['Emotion']))

# Now we split remaining 90% into 80% for training and 10% for validation
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=1/8, random_state=42)
train_idx, val_idx = next(stratified_split.split(df.iloc[train_val_idx]['Text'], df.iloc[train_val_idx]['Emotion']))

# This essentially gives us 80% train, 10% validation, and 10% test sets
df_train = df.iloc[train_val_idx].iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[train_val_idx].iloc[val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)
```

## Model Analysis

### Error Analysis

The notebook identifies the following most common misclassifications:

```
Most common misclassification types:
  Fear - Surprise: 217 examples
  Sadness - Anger: 136 examples
  Anger - Fear: 114 examples
  Surprise - Fear: 104 examples
  Sadness - Fear: 103 examples
```

**Overall error rate**: 8.82% (1,323 errors out of 15,000 test samples)

### Distribution Analysis

The model's predicted emotion distribution vs. the actual distribution on the test set:

| Emotion | True Count | Predicted Count | True % | Predicted % | Difference % |
|---------|------------|-----------------|--------|-------------|--------------|
| Joy | 4100 | 4005 | 27.33 | 26.70 | -0.63 |
| Sadness | 4100 | 3956 | 27.33 | 26.37 | -0.96 |
| Anger | 3200 | 3249 | 21.33 | 21.66 | 0.33 |
| Fear | 2700 | 2682 | 18.00 | 17.88 | -0.12 |
| Surprise | 900 | 1108 | 6.00 | 7.39 | 1.39 |

## Prediction Function

The notebook implements a simple prediction function that can be used for new text samples:

```python
def predict_emotion(text):
    cleaned_text = enhanced_preprocess_text(text)
    tfidf_features = tfidf.transform([cleaned_text])
    predicted_emotion = improved_model.predict(tfidf_features)[0]
    return predicted_emotion
```

### Example Predictions

The notebook tests the model on the following custom examples:

```
Emotion category: Joy
Example text: Today has been the happiest day of my life, everything feels perfect!
Predicted emotion: Joy

Emotion category: Sadness
Example text: I can't stop crying after hearing the news about my friend's illness.
Predicted emotion: Sadness

Emotion category: Anger
Example text: I'm absolutely furious about how they treated our team at the meeting.
Predicted emotion: Anger

Emotion category: Fear
Example text: I'm terrified of what might happen if I fail this important exam.
Predicted emotion: Fear

Emotion category: Surprise
Example text: Wow! I never expected to win the lottery, I'm completely shocked!
Predicted emotion: Surprise
```

## Licence

This project is licensed under the **MIT Licence** - see the [LICENCE](LICENCE) file for complete details.

---

[![MIT Licence](https://img.shields.io/badge/Licence-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-91.18%25-success.svg)]()
[![F1 Score](https://img.shields.io/badge/Weighted_F1-0.91-success.svg)]()