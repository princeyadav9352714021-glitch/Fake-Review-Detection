Fake Review Detection using NLP and Machine Learning
Overview
Fake Review Detection is an NLP and machine learning project that classifies product reviews using text preprocessing, TF-IDF vectorization, and a Linear Support Vector Classifier (Linear SVM). The notebook explores the dataset, visualizes label distribution, evaluates the model, and predicts labels for new reviews.

Dataset
The project downloads the dataset through KaggleHub: mexwell/fake-reviews-dataset.

The notebook reads fake reviews dataset.csv and uses:

text_: review text
label: target label
Check the actual values in df["label"].unique() before interpreting a prediction as “Fake” or “Real”.

Workflow
Download and load the dataset with KaggleHub.
Inspect dataset information, missing values, and duplicates.
Select the review text and label columns; remove missing, empty, and duplicate rows.
Visualize label distribution with Matplotlib and Seaborn.
Preprocess text using lowercase conversion, URL/HTML removal, alphabetic filtering, English stop-word removal, and NLTK lemmatization.
Split data into training and testing sets using an 80/20 stratified split.
Train a Scikit-learn pipeline with TF-IDF and LinearSVC.
Evaluate using accuracy, a classification report, and a confusion matrix.
Predict the label of a new review.
Save the trained pipeline as fake_review_nlp_model.pkl using Joblib.
Technologies
Python
Pandas
NLTK
Scikit-learn
Matplotlib and Seaborn
KaggleHub
Joblib
Run the Notebook
Open Fake_Review_Detection.ipynb in Google Colab or Jupyter and run the cells from top to bottom. Internet access may be required to download the dataset and NLTK resources.

Install Dependencies
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn nltk joblib
Load the Saved Model
Keep fake_review_nlp_model.pkl in your project folder. The saved pipeline includes TF-IDF vectorization and the Linear SVM classifier.

import joblib

model = joblib.load("fake_review_nlp_model.pkl")

# Use the same preprocess_text() function from the notebook.
review = "The product is excellent and works exactly as described."
clean_review = preprocess_text(review)
prediction = model.predict([clean_review])[0]

print("Predicted label:", prediction)
Use the same preprocess_text() function that was used during training. The meaning of each prediction depends on the dataset's label encoding.

Limitations
Results depend on the quality and coverage of the training dataset.
A prediction is not definitive proof that a review is fake or genuine.
Test the model on representative, unseen reviews before using it in a real application.
Model File
The notebook saves the trained pipeline as fake_review_nlp_model.pkl.
