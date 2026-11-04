# GridWise

## Project Overview

GridWise is an AI-powered smart grid management and customer support platform designed to help municipal electricity utilities modernize their operations by leveraging advanced data analytics, machine learning, deep learning, and natural language processing. It transforms raw transactional electricity demand data and unstructured customer feedback into actionable business intelligence, enabling data-driven decision-making, peak load forecasting, capacity warning alerts, and automated fault triage. The solution addresses key energy sector challenges such as inventory and load management, incident urgency classification, customer query resolution, and emergency safety routing, making sophisticated analytics accessible to local municipalities and utility service providers.

## Features

* **Data Loading and Cleaning:** Efficiently loads and preprocesses hourly grid consumption data, resolving timestamp duplicates, handling missing hours via time-based interpolation, and ensuring uniform hourly time-series frequency.
* **Time Series Analysis:** Conducts additive classical decomposition, autocorrelation profiling, and multi-temporal cycle analyses (hourly, day-of-week, and monthly) to identify grid consumption drivers.
* **Demand Forecasting Engine:** Predicts day-ahead hourly demand using models such as Seasonal Naïve benchmarks, HistGradientBoosting, Multi-Layer Perceptrons (MLP), optional SARIMA, and Deep Learning LSTMs to forecast peak loads and issue automated capacity warnings.
* **Automated Fault Classification & Triage:** Employs TF-IDF feature extraction (combining word and character n-grams) and Logistic Regression to categorize incident reports (e.g., outage, cable damage, billing) and assign urgency levels (high, medium, low).
* **Interactive Dashboard:** Features a multi-tab Streamlit dashboard displaying live peak forecasts, capacity alerts, interactive model comparison metrics, visual analytics charts, and a fault classification playground.
* **Chatbot/Voice Assistant:** Features an intent-matching conversational AI with full voice capabilities (speech recognition via SpeechRecognition and speech synthesis via pyttsx3/gTTS) for 24/7 text and voice query resolution, ticket logging, and live forecast retrieval.
* **Comprehensive Analytics Pipeline:** Integrates all preprocessing, training, evaluation, visualization generation, poster building, and document generation into a single executable pipeline.

## Installation

1. **Prerequisites:** Ensure you have Python 3.8+ installed on your system.

2. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/GridWise.git
   cd GridWise
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   The required packages include:
   * pandas
   * numpy
   * scikit-learn
   * matplotlib
   * joblib
   * python-docx
   * reportlab
   * pillow
   * streamlit
   * torch
   * statsmodels
   * holidays
   * speechrecognition
   * pyaudio
   * pyttsx3
   * gTTS

4. **Run the Application:**

   To generate synthetic baseline data (run once):
   ```bash
   python generate_data.py
   ```

   To execute the complete pipeline (train models, generate charts, and save metrics):
   ```bash
   python run_pipeline.py
   ```

   *(Optional)* To run the pipeline using a real hourly demand dataset (e.g., PJM data):
   ```bash
   python run_pipeline.py --demand path/to/AEP_hourly.csv
   ```

   To launch the interactive dashboard:
   ```bash
   streamlit run app.py
   ```

   To chat with the GridWise assistant via terminal (Text/Voice mode):
   ```bash
   python cli_chatbot.py
   ```