# 💬 EmotionSense — Smart Sentiment Classifier

EmotionSense is a simple yet intelligent **AI-powered web app** that classifies user-entered text as **Positive 😊 or Negative 😞**.  
The project focuses on **clarity, logic, and creativity**, showing how Natural Language Processing (NLP) can be implemented from scratch — **without using any pretrained AI models**.

---

## 🧠 Project Overview

This project demonstrates how text data can be analyzed and classified using a **machine learning model** built entirely from scratch.  
EmotionSense accepts user input through a **beautiful interactive web interface**, processes it using a custom-trained model, and instantly predicts the emotional tone.

The core idea is to show:
- Problem-solving ability in AI.
- Logical thinking in model creation.
- Creative front-end presentation and back-end integration.

---

## 🚀 Features

✅ Simple and clean Flask-based architecture
✅ Custom sentiment classifier (no pre-trained models)
✅ Elegant, glassmorphic UI built with HTML
✅ Emoji-based real-time feedback (Positive / Negative)
✅ Easy to understand and extend  
---

## 🏗️ Project Structure

EmotionSense/
│

   ├── model/

    └── sentiment_model.pkl # Trained sentiment model

│
├── templates/

    └── index.html # Webpage layout
│

├── app.py # Flask backend server

├── train_model.py # Model training script

├── requirements.txt # Python dependencies

└── README.md # Project description


## Install dependencies

pip install -r requirements.txt


## Train the model

python train_model.py


This script creates a small text dataset, trains a logistic regression model, and saves it as model/sentiment_model.pkl.

## Run the app

python app.py


Then open your browser at 👉 http://127.0.0.1:5000

## Live app
Open your browser 👉 https://sentiment-classifier-001.streamlit.app/
