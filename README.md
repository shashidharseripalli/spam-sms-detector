# spam-sms-detector
# 📧 Spam Email Detection with Machine Learning

This project builds a machine learning model to classify SMS messages as **Spam** or **Ham** (not spam), using natural language processing and supervised learning techniques.

## 🔍 Overview

- **Goal**: Detect spam messages using text classification.
- **Algorithm**: Multinomial Naive Bayes
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Dataset**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

## 📁 Files Included

- `Spam_Email_Detection.ipynb`: Jupyter Notebook containing the full implementation in Python using scikit-learn.
- `README.md`: Project overview and instructions.

## 🛠️ Technologies Used

- Python 🐍
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- TF-IDF Vectorizer
- Naive Bayes Classifier

## 📊 Workflow

1. **Data loading** from UCI repository
2. **Text preprocessing** (cleaning, normalization)
3. **Feature extraction** using TF-IDF
4. **Model training** with Naive Bayes
5. **Model evaluation** (accuracy, precision, recall, F1-score)
6. **Confusion matrix** for performance visualization

## 📌 Example Output

- Model Accuracy: ~97%
- Confusion Matrix:
  
  |        | Predicted Ham | Predicted Spam |
  |--------|----------------|----------------|
  | Actual Ham  | ✅  |
  | Actual Spam | ✅  |

*(Replace ✅ with real numbers from your confusion matrix if you'd like)*

## 🚀 How to Run

You can run the notebook directly in [Google Colab](https://colab.research.google.com) or Jupyter Notebook:

```bash
pip install -r requirements.txt
jupyter notebook Spam_Email_Detection.ipynb

