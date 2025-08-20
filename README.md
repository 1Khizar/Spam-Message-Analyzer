## 📩 Spam Message Analyzer

A Streamlit-based web application for detecting spam messages using machine learning and NLTK preprocessing.

## Features:

 **Text Preprocessing:** Tokenization, stopword removal, and cleaning using NLTK.

 **Spam Detection:**: Predicts whether a message is Spam or Not Spam using a trained ML model.

##  Data Visualization:

Word frequency bar charts

Spam vs Ham distribution pie chart

Confidence gauge meter

**📁File Upload:** Upload multiple text files (.txt) for batch spam analysis.


## Tech Stack

**Python** 

**Streamlit**
 – UI framework

**NLP**
 – Natural Language preprocessing

**Scikit-learn**
 – ML model training & inference

**Plotly**
 – Interactive visualizations

**NumPy**
 – Numerical operations

## 📂 Project Structure
📁 spam-analyzer
│── app.py                # Main Streamlit app
│── model.pkl             # Trained ML model (pickle file)
│── vectorizer.pkl        # TF-IDF/CountVectorizer (pickle file)
│── requirements.txt      # Required Python packages
│── README.md             # Project documentation

## ⚙️ Installation

Clone the repository

git clone https://github.com/your-username/spam-analyzer.git
cd spam-analyzer


## Install dependencies

pip install -r requirements.txt


## Run the Streamlit app

streamlit run app.py


## 🖥️ Usage

Enter a message in the text box and click Analyze.

Upload .txt files to analyze batch messages.

View spam probability, confidence score, and data visualizations in real time.

## 📸 Screenshots
**Main Interface**
<img width="956" height="442" alt="Spam Message Analyzer " src="https://github.com/user-attachments/assets/fefb5d92-5e79-4b4b-9d6b-f19924681f29" />

**Anaytics**

<img width="941" height="437" alt="image" src="https://github.com/user-attachments/assets/8aa35118-6937-49cf-8e18-b5e6491ecedf" />


## Author

Developed by **Khizar Ishtiaq** ✨
