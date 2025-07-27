import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv("train.txt", sep=';', header=None, names=["text", "emotion"])
        return df
    except FileNotFoundError:
        st.error("Training data file 'train.txt' not found!")
        return None

@st.cache_resource
def train_model():
    """Train the emotion classification model"""
    df = load_data()
    if df is None:
        return None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['emotion'], test_size=0.2, random_state=42
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

def predict_emotion(text, model, vectorizer):
    """Predict emotion for given text"""
    if model is None or vectorizer is None:
        return None
    
    # Transform text
    text_tfidf = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf).max()
    
    return prediction, confidence

# Main app
def main():
    st.title("😊 Text Emotion Classifier")
    st.markdown("Enter some text and I'll predict the emotion!")
    st.markdown("🐈 meow meow!")

    
    # Load model
    with st.spinner("Loading model..."):
        model, vectorizer = train_model()
    
    if model is None:
        st.stop()
    
    # Input section
    st.markdown("### Enter your text:")
    user_input = st.text_area(
        "",
        placeholder="Type your text here... (e.g., 'I am feeling very happy today!')",
        height=100
    )
    
    # Prediction section
    if st.button("Predict Emotion", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                prediction, confidence = predict_emotion(user_input, model, vectorizer)
            
            if prediction:
                # Display result
                st.markdown("### Result:")
                
                # Emotion mapping for better display
                emotion_emojis = {
                    'joy': '😊',
                    'sadness': '😢',
                    'anger': '😠',
                    'fear': '😨',
                    'love': '💕',
                    'surprise': '😲'
                }
                
                emoji = emotion_emojis.get(prediction, '🤔')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Emotion", f"{emoji} {prediction.title()}")
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Show confidence bar
                st.progress(confidence)
        else:
            st.warning("Please enter some text to analyze!")
    
    # Dataset info (collapsible)
    with st.expander("ℹ️ About this model"):
        df = load_data()
        if df is not None:
            st.write(f"**Dataset size:** {len(df):,} samples")
            st.write("**Emotions detected:**")
            emotion_counts = df['emotion'].value_counts()
            for emotion, count in emotion_counts.items():
                emoji = emotion_emojis.get(emotion, '🤔')
                st.write(f"- {emoji} {emotion.title()}: {count:,} samples")
            
            st.write("**Model:** Logistic Regression with TF-IDF features")

if __name__ == "__main__":
    main()
