# ReviewLens: Fake Review Intelligence

ReviewLens is a Streamlit app for exploring a trained fake-review text classifier. It predicts whether a review resembles the dataset's **CG** (computer-generated) or **OR** (original) label, and includes a CSV dataset explorer.

## Requirements

- Python 3.10
- The saved model file `fake_review_nlp_model.pkl` in the project folder
- Packages listed in `requirements.txt`

## Setup

In PowerShell, from the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the virtual environment has not been created yet:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Features

- **Classify a review:** paste review text to get a predicted label and the classifier's raw decision score.
- **Explore dataset:** upload a CSV with `label` and `text_` columns, or download the public Kaggle dataset from the sidebar.
- **About:** view a brief model description and limitations.

The app applies the text cleaning, English stopword removal, and lemmatization used in the training notebook. It downloads the NLTK `stopwords` and `wordnet` resources on first use if they are not already available.

## Understanding predictions

- `CG` means computer-generated in the training dataset.
- `OR` means original in the training dataset.
- The displayed LinearSVC decision score is **not a probability** and should not be read as a confidence percentage.

Predictions are statistical estimates based on text patterns, not proof of who wrote a review. Results can be inaccurate, particularly for short, translated, or out-of-domain reviews.

## Dataset source

The optional dataset download uses [`mexwell/fake-reviews-dataset`](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) via KaggleHub. Kaggle access or network connectivity may be required. Alternatively, upload a local CSV containing the required columns.
