# Spam Email Classifier

This project is a machine learning-based spam detection system that classifies emails as **Spam** or **Not Spam** using natural language processing (NLP) and a **Naive Bayes classifier**. A simple **Streamlit** interface allows users to interactively test email texts.

---

# Features

- Text preprocessing using NLTK
- TF-IDF feature extraction
- Multinomial Naive Bayes classification
- Streamlit-powered web interface
- Model persistence with `joblib`

---

# Project Structure

spam-email-classifier/
<br/>
├── data/
<br/>
│ └── spam.csv
<br/>
├── models/
<br/>
│ ├── spam_classifier.pkl
<br/>
│ └── vectorizer.pkl
<br/>
├── spam_email_classifier.py
<br/>
├── README.md
<br/>
└── requirements.txt

---

# Installation

1. **Clone the repository**:

- git clone https://github.com/yourusername/spam-email-classifier.git
</br>
- cd spam-email-classifier

2. **Create and activate a virtual environment (optional but recommended):**

- python -m venv venv
<br/>
- On Mac: source venv/bin/activate
<br/>
- On Windows: venv\\Scripts\\activate

3. **Install dependencies:**
<br/>
- pip install -r requirements.txt

---

# Running the App

- Make sure the dataset (spam.csv) is in the 'data/' folder.
<br/>
Then, run:
<br/>
streamlit run spam_email_classifier.py

---

# Dataset

- SMS Spam Collection Dataset
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
<br/>
- Format
- label: spam or ham
- message: email or SMS content

---

# Tech used
- Python
- Scikit-learn
- NLTK
- Streamlit
- Pandas, Joblib, Regex
