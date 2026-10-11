# From Waveforms to Intelligence: Speech Grammar Scoring

An end-to-end multimodal machine learning pipeline that predicts a
numerical grammar score from spoken audio by combining **acoustic signal
processing, Whisper-based speech recognition, linguistic feature
engineering, sentence embeddings, and ensemble regression**.

> **Goal:** Convert a speech recording into a predicted grammar score on
> a 0--5 scale.


## Overview

Grammar is expressed primarily through language, so audio-only features
may miss important information about sentence structure and word choice.
This project takes a multimodal approach: it analyzes both **how the
audio sounds** and **what the speaker says**.

The pipeline uses pretrained models to extract useful representations
and trains regression models on the labelled dataset. This is designed
to make effective use of a relatively small training set while limiting
unnecessary model complexity.

## Pipeline Architecture

``` text
                  Raw Speech Audio
                         |
             Audio Inspection & Cleanup
        (16 kHz mono, silence trimming, checks)
                         |
              +----------+-----------+
              |                      |
        Acoustic Branch         Language Branch
              |                      |
     MFCCs + Delta Features      Whisper ASR
     Energy & Spectral Stats         |
     Chroma & Pause Stats       Transcript Text
              |                 +----+----------------+
              |                 |                     |
              |          Linguistic Features   Sentence Embeddings
              |                 |             (pretrained, frozen)
              +-----------------+---------------------+
                                |
                       Feature Fusion
             (including words-per-second features)
                                |
                    Preprocessing & Validation
                                |
             +------------------+------------------+
             |                  |                  |
           Ridge             ExtraTrees       Random Forest
             |                  |                  |
             +------------------+------------------+
                                |
                       HistGradientBoosting
                                |
                     Ensemble / Model Blending
                                |
                  Predicted Grammar Score [0, 5]
```

## How It Works

### 1. Audio preprocessing and inspection

Before feature extraction, the audio is inspected for duration, sample
rate, channel count, waveform shape, and Mel-spectrogram patterns.

-   Resample audio to **16 kHz, mono**.
-   Trim leading and trailing silence using a 30 dB threshold.
-   Cache speech transcripts so repeated experiments do not need to
    transcribe the same clip again.
-   Handle empty transcripts safely by assigning zero-valued text
    features.
-   Impute missing or infinite feature values using **training-set
    medians only**.
-   Drop zero-variance columns.
-   Clip final predictions to the valid score range of **\[0, 5\]**.

### 2. Acoustic feature extraction

The acoustic branch captures characteristics of the speech signal,
including:

-   MFCCs and delta MFCCs
-   Root mean square (RMS) energy
-   Zero-crossing rate (ZCR)
-   Spectral centroid, bandwidth, roll-off, and contrast
-   Chroma features
-   Pause-related statistics

For time-varying features, summary statistics such as the **mean,
standard deviation, minimum, and maximum** are calculated to create
fixed-length representations for regression models.

### 3. Speech recognition with Whisper

The audio is transcribed using **Whisper automatic speech recognition
(ASR)**. The resulting text provides a linguistic view of the spoken
response, complementing the information captured by acoustic features.

Transcripts are cached for efficiency. Empty or unavailable transcripts
are handled explicitly so they do not break downstream feature
generation.

### 4. Linguistic features and semantic embeddings

Two types of text representations are extracted:

**Handcrafted linguistic features** - Transcript and word-level
statistics - Language-related indicators derived from the recognized
text - Cross-modal fluency features, including words per second

**Sentence embeddings** - A pretrained sentence-transformer generates
dense semantic representations. - The embedding model is kept frozen;
only downstream regression models are trained on the competition labels.

Together, these features represent both interpretable text statistics
and richer semantic information.

### 5. Multimodal feature fusion

The acoustic, handcrafted linguistic, and sentence-embedding feature
blocks are concatenated into a single feature matrix.

This fusion allows the model to learn from complementary signals instead
of relying only on acoustic patterns or transcript features.

### 6. Regression models

The task is treated as **supervised regression**, because each recording
has a numerical grammar score.

The candidate models are:

  Model                            Role
  -------------------------------- ----------------------------------------
  Ridge Regression                 Regularized linear baseline
  ExtraTrees Regressor             Nonlinear ensemble of randomized trees
  Random Forest Regressor          Bagged tree-based regression
  HistGradientBoosting Regressor   Gradient-boosted tree model

### 7. Validation and ensemble selection

Model selection uses **5-fold stratified out-of-fold (OOF) validation**,
with score bins used for stratification where appropriate.

The workflow compares: - Individual regression models - An equal-weight
prediction blend - A non-negative least-squares (NNLS) stacked blend

The stacking approach is evaluated with its own nested cross-validation
procedure to reduce optimistic estimates from reusing the same
validation predictions.

After model selection, the chosen regressors are refit on the full
labelled training set. Their predictions are blended and clipped to the
valid range.

## Dataset

This project was developed for a grammar-scoring-from-speech machine
learning competition.

The working dataset contains a relatively small labelled training set
(reported as **769 audio clips**) and a separate test set. The notebook
expects the competition's CSV files and audio assets to be available
locally.

Typical competition files may include:

``` text
project/
├── README.md
├── notebook.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
└── audio/
    ├── ...
```

**Important:** Adjust the directory layout and column names in the
notebook to match the files supplied by the competition. Do not commit
restricted competition data or private audio recordings to a public
repository.

## Tech Stack

-   **Python**
-   **NumPy / pandas** --- data handling
-   **librosa** --- audio loading and signal features
-   **Whisper** --- automatic speech recognition
-   **sentence-transformers** --- text embeddings
-   **scikit-learn** --- preprocessing, regression, validation, and
    model blending
-   **Matplotlib / Seaborn** --- exploratory visualization, if used in
    the notebook

Exact package versions should be pinned in a `requirements.txt` after
confirming the versions used in your working environment.

## Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment

**Windows**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt`:

``` bash
pip install -r requirements.txt
```

Otherwise, install the packages used by your notebook and pin the
versions that work in your environment. Whisper and audio-processing
libraries may require additional system dependencies depending on the
operating system.

### 4. Add the competition data

Place the competition CSV files and audio files in the paths expected by
the notebook. Update the configuration or path cells if your local
folder structure differs.

### 5. Run the notebook

Open `notebook.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and
run the cells in order.

The pipeline extracts or loads cached features, validates candidate
regressors, selects a blend, fits the final models, and generates
test-set predictions in the required submission format.

## Results and Output

The final output is a numeric grammar-score prediction for each test
audio clip. The example submission format is:

``` csv
filename,score
_1138.wav,3.313394
```

Use the **exact column names and row ordering required by the
competition's `sample_submission.csv`**. The row above is an
illustrative prediction example, not a claim of overall model
performance.

A prediction distribution summary from one 204-row output was:

  Statistic              Value
  -------------------- -------
  Count                    204
  Mean                   3.263
  Standard deviation     0.307
  Minimum                1.347
  25th percentile        3.313
  Median                 3.313
  75th percentile        3.313
  Maximum                4.930

These are descriptive statistics of a prediction output, **not
validation metrics**. To report model quality, include the competition's
evaluation metric and a verified cross-validation score from the final
notebook.

## Engineering Considerations

-   **Small labelled dataset:** use pretrained representations and
    regularized or tree-based regressors rather than training a large
    speech model from scratch.
-   **Expensive transcription:** cache Whisper outputs to avoid
    repeatedly processing the same audio.
-   **Leakage prevention:** calculate imputation statistics from
    training data only, and keep validation folds isolated during model
    selection.
-   **Multimodal complementarity:** combine acoustic information with
    linguistic and semantic signals.
-   **Robust predictions:** handle empty transcripts, invalid feature
    values, and output-range constraints.
-   **Honest evaluation:** distinguish prediction-distribution
    statistics from held-out validation performance.

## Limitations

-   Whisper transcription errors can affect downstream linguistic
    features.
-   A small labelled dataset can make validation estimates noisy.
-   Grammar scores may be subjective or influenced by factors beyond
    grammatical correctness, including fluency, accent, and recording
    quality.
-   The prediction range constraint does not guarantee that the
    predictions are well calibrated.
-   Performance should be judged using the competition's official metric
    and a carefully designed validation strategy.

## Future Improvements

-   Compare Whisper model sizes and transcription settings.
-   Add grammar-specific features such as clause structure,
    part-of-speech patterns, and syntactic error indicators.
-   Evaluate feature groups through ablation studies.
-   Compare the current ensemble against a simple, well-tuned baseline.
-   Track fold-level metrics and prediction errors to understand where
    the model fails.
-   Package feature extraction and inference into a reproducible
    command-line pipeline.

## What This Project Demonstrates

-   End-to-end ML pipeline design
-   Audio signal processing and feature engineering
-   Automatic speech recognition integration
-   NLP feature extraction and sentence embeddings
-   Multimodal feature fusion
-   Cross-validation and ensemble regression
-   Leakage-aware preprocessing and reproducible inference

## Author

**Trisha Singh**\
AI/ML Engineering \| Machine Learning \| NLP \| Data Science

------------------------------------------------------------------------

*Built as a practical exploration of how acoustic signals and language
representations can work together to estimate spoken-language grammar
scores.*
