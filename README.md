# Efficient Data Reduction for Sentiment Analysis

## Overview
This repository contains code to reproduce the experiments from the paper **"Efficient Data Reduction Through Maximum Separation Vector Selection and Centroid Embedding Representation"**. The study introduces two novel data reduction techniques—High-Distance Sentiment Vectors (HDSV) and Centroid Sentiment Embedding Vectors (CSEV)—that significantly reduce training data while maintaining classification accuracy for sentiment analysis.

## Repository Structure

### 1. **Dimensionality Reduction & Visualization**
   - **File:** `dimensionality_reduction.py`
   - **Purpose:** Performs PCA and t-SNE to visualize sentiment embeddings.
   - **Key Functions:**
     - `reduce_dimensions()`: Applies PCA and t-SNE.
     - `plot_embeddings()`: Generates visualizations.

### 2. **High-Distance Sentiment Vector Selection (HDSV)**
   - **File:** `HDSV.py`
   - **Purpose:** Selects maximally informative samples based on Euclidean distance.
   - **Key Functions:**
     - `select_distinctive_samples()`: Finds high-distance pairs.
     - `visualize_samples()`: Displays selected samples in 2D space.

### 3. **Centroid Sentiment Embedding Vectors (CSEV) Generation**
   - **File:** `CSEV_Generation.py`
   - **Purpose:** Computes representative centroids for each sentiment class.
   - **Key Functions:**
     - `average_embeddings()`: Computes class-wise centroids.
     - `create_averaged_dataset()`: Builds a dataset with centroid representations.

### 4. **Fine-Tuning Sentiment Analysis Models**
   - **File:** `fine_tuning.py`
   - **Purpose:** Trains and evaluates a DistilBERT-based model on reduced datasets.
   - **Key Functions:**
     - `compute_metrics()`: Evaluates model performance.
     - `train_and_evaluate()`: Fine-tunes and tests sentiment classifiers.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or prepare datasets under `Datasets/`.
4. Run individual scripts as needed, e.g.:
   ```bash
   python HDSV.py
   ```

## Results & Reproducibility
- The models trained on reduced datasets achieve comparable accuracy to full dataset training while requiring significantly fewer samples.
- Evaluations include cross-dataset generalization to validate robustness.
- Results are saved in `evaluation_results/`.


