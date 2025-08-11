# Spam-SMS-Detection
This project focuses on detecting and classifying SMS messages as Spam or Ham (non-spam) using Natural Language Processing (NLP) and Machine Learning techniques.
It applies text preprocessing, feature extraction, and supervised learning algorithms to build a robust spam classifier.

🚀 Features
🧹 Data Preprocessing & Cleaning – Removed noise, punctuation, stopwords, and unnecessary characters

✨ Feature Extraction – Used TF-IDF Vectorizer to convert text into numerical features

🤖 Model Training – Implemented Naive Bayes, Logistic Regression, and Support Vector Machine (SVM)

📊 Data Visualization – Spam vs Ham distribution using Matplotlib & Seaborn

📈 Model Evaluation – Accuracy Score & Confusion Matrix for performance measurement

🔮 Custom Prediction Function – Predict spam status for new/unseen messages

🛠 Tech Stack
Python

Pandas

scikit-learn

NLTK

Matplotlib

Seaborn

📂 Dataset
The dataset contains a collection of SMS messages labeled as ham (non-spam) or spam.
Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

📊 Model Workflow
Load and explore dataset

Clean and preprocess text data

Convert text to numerical features using TF-IDF

Train ML models (Naive Bayes, Logistic Regression, SVM)

Evaluate models using accuracy and confusion matrix

Predict new SMS messages

📌 Output Example
makefile
Copy
Edit
Example SMS: "Congratulations! You've won a free iPhone. Click here to claim now!"
Prediction: Spam 📩
📈 Results
Achieved high accuracy across all tested models with accuracy of 97%

