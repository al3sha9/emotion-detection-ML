# Emotion Detection with Logistic Regression & Streamlit UI

This project demonstrates an **Emotion Detection** system that:
- Trains a text classification model using **TF-IDF** and **Logistic Regression**
- Provides a **Streamlit-based web app** for interactive emotion prediction

## 📌 Features
- Loads and preprocesses a text dataset (`train.txt`)
- Vectorizes text
- Trains a **Logistic Regression** model
- Evaluates performance using accuracy, classification report, and confusion matrix
- Visualizes results using **Seaborn** and **Matplotlib**

## Structure
```
├── main.ipynb # Notebook for training and evaluating the model
├── app.py # Streamlit web app for real-time predictions
├── train.txt # Dataset (text;emotion format)
├── run_app.sh # Bash script to launch the Streamlit app
└── env/ # Python virtual environment (optional)
```

## 📂 Dataset
The notebook expects a dataset file named `train.txt` in the same directory.  
Format:
```
text: emotion
I am happy: joy
I feel sad: sadness

```

## 🚀 Requirements
Install the required Python libraries before running the notebook:
```bash
pip install pandas seaborn matplotlib scikit-learn
```

## ▶️ Usage
```bash
python3 -m venv env
source env/bin/activate
```
## 🔹Run the Streamlit Web App:
Using the provided bash script
```bash
bash run_app.sh
```
