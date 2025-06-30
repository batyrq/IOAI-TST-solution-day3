# IOAI TST Kazakhstan - Day 3: Code Difficulty Classification

This repository contains the solution for the third day's problem of the IOAI TST (Team Selection Test) in Kazakhstan. The task was to classify programming code snippets into one of three difficulty levels: "easy", "medium", or "hard".

**Kaggle Competition Link:** [https://www.kaggle.com/competitions/kz-tst-day-3](https://www.kaggle.com/competitions/kz-tst-day-3)

## Table of Contents

  - [Problem Description](https://www.google.com/search?q=%23problem-description)
  - [Solution Overview](https://www.google.com/search?q=%23solution-overview)
  - [Code Description](https://www.google.com/search?q=%23code-description)
  - [Dependencies](https://www.google.com/search?q=%23dependencies)
  - [Usage](https://www.google.com/search?q=%23usage)

## Problem Description

The challenge involved analyzing code snippets and assigning them a difficulty label (easy, medium, or hard). This is a multi-class text classification problem where the "text" is source code.

## Solution Overview

The solution employs a combination of handcrafted features extracted from the code and TF-IDF representations of the code text, followed by a LightGBM classifier. Key steps include:

1.  **Data Preprocessing:** Cleaning and preparing the code snippets and mapping difficulty labels to numerical values.
2.  **Class Balancing (Oversampling):** Addressing potential class imbalance in the training data by oversampling minority classes.
3.  **Feature Engineering:**
      * **Statistical Features:** Extracting basic textual statistics (e.g., character length, word count, sentence count, number of digits).
      * **Code Structure Features:** Identifying programming constructs (e.g., number of loops, if statements, functions, comments).
      * **Algorithmic Keywords:** Detecting the presence of common algorithmic concepts (e.g., "tree", "dynamic programming", "hash", "stack", "recursive").
      * **TF-IDF Vectorization:** Converting code text into numerical TF-IDF vectors to capture important terms and their frequencies.
4.  **Feature Concatenation:** Combining the handcrafted features with the TF-IDF vectors.
5.  **Model Training:** Training a LightGBM Classifier on the combined feature set.
6.  **Evaluation:** Assessing model performance on both oversampled validation data and the original, unbalanced training data.
7.  **Prediction and Submission:** Generating predictions for the test set and formatting them for submission.

## Code Description

The provided Python script `batyr-yerdenov-3.ipynb` details the implementation:

  * **Import Libraries:** `pandas` for data handling, `numpy` for numerical operations, `re` for regular expressions, `scipy.sparse` for efficient matrix operations, `sklearn.feature_extraction.text` for TF-IDF, `sklearn.model_selection` for splitting data, `lightgbm` for the classifier, and `sklearn.metrics` for evaluation.

  * **`extract_code_features(df)` function:**

      * Calculates various lexical and structural features: `char_len`, `word_count`, `sentence_count`, `avg_token_len`, `num_digits`.
      * Counts occurrences of common programming constructs: `num_loops`, `num_if`, `num_functions`, `num_comments`.
      * Detects the presence of specific keywords or patterns indicative of algorithmic concepts: `has_O_complexity`, `contains_tree`, `contains_dp`, `contains_hash`, `contains_stack`, `contains_recursive`, `has_algo_steps`.

  * **Data Loading and Preprocessing:**

      * Loads `train.csv`, `test.csv`, and `sample_submission.csv`.
      * Fills missing 'code' values with empty strings, strips whitespace, and removes rows with empty code.
      * Maps `difficulty` labels ('easy', 'medium', 'hard') to numerical values (0, 1, 2).
      * Creates a copy of the original training data (`train_orig`) for unbiased evaluation.

  * **Class Balancing (Oversampling):**

      * Identifies the count of the majority class.
      * Randomly samples (with replacement) from the minority classes to match the majority class count, creating a balanced `train_bal` DataFrame.

  * **Feature Extraction:**

      * Applies `extract_code_features` to the balanced training data.
      * Defines `feature_cols` for the numerical features extracted.
      * Initializes `TfidfVectorizer` to extract TF-IDF features from the 'code' column. `max_features` is set to 50000 and `ngram_range` to (1, 2) to capture common single words and two-word phrases.
      * Combines TF-IDF features with extracted numerical features using `hstack` (Horizontal Stack) from `scipy.sparse`.

  * **Train/Validation Split and Model Training:**

      * Splits the balanced `X_full` and `y` into training and validation sets using `train_test_split`, ensuring stratification to maintain class proportions.
      * Initializes and trains an `LGBMClassifier` with `random_state=42`.

  * **Evaluation:**

      * Prints a classification report for the validation set (oversampled data).
      * Evaluates the model's accuracy and generates a classification report on the *original* (unbalanced) training data to provide a more realistic assessment of performance before oversampling.

  * **Prediction on Test Set:**

      * Applies the same preprocessing and feature extraction steps to the `test` DataFrame.
      * Uses the trained LightGBM model to predict `difficulty` for the test set.

  * **Submission Generation:**

      * Maps the numerical predictions back to 'easy', 'medium', 'hard' labels.
      * Saves the results to `submission.csv` in the required format.

  * **Output Preview:** Prints the distribution of predicted difficulties and a preview of the submission file.

## Dependencies

  * `pandas`
  * `numpy`
  * `scipy`
  * `scikit-learn`
  * `lightgbm`

You can install these dependencies using pip:

```bash
pip install pandas numpy scipy scikit-learn lightgbm
```

## Usage

1.  **Download the data:** Obtain `train.csv`, `test.csv`, and `sample_submission.csv` from the competition page ([https://www.kaggle.com/competitions/kz-tst-day-3](https://www.kaggle.com/competitions/kz-tst-day-3)) and place them in the specified Kaggle input directory (`/kaggle/input/kz-tst-day-3/`). If running locally, adjust the paths in the script accordingly.
2.  **Run the Jupyter Notebook:** Open and run the `batyr-yerdenov-3.ipynb` notebook.
3.  **Generate Submission:** The script will automatically generate a `submission.csv` file in the same directory where the notebook is executed. This file will contain player IDs and their predicted cluster assignments.

-----
