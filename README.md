# GridWise
GridWise is an AI-driven tool designed to help municipal electricity utilities plan supply and serve customers more effectively by leveraging time series analysis, machine learning, deep learning, and natural language processing. It transforms hourly demand data into 24-hour-ahead forecasts with capacity warnings, and turns customer fault reports and enquiries into classified, prioritised tickets. The solution addresses key utility challenges such as unplanned outages, over- or under-purchasing of power, and complaint queues on phone lines and at service counters, making sophisticated analytics accessible to municipalities that still rely on averages and manual estimates.

## Features

- **Data Loading and Cleaning**: Loads hourly demand data (such as the PJM dataset), removes duplicate hours, fills missing values and flags public holidays.
- **Time Series Analysis**: Applies decomposition, autocorrelation and hourly, weekly and monthly demand profiles to reveal daily and weekly seasonality.
- **Demand Forecasting**: Uses gradient boosting, a neural network (MLP), SARIMA and an LSTM to predict demand 24 hours ahead, benchmarked against a seasonal-naive model.
- **Capacity Warnings**: Flags forecast hours that exceed the 95th percentile of historical demand so that load-reduction measures can be planned.
- **NLP Fault Classification**: Uses TF-IDF and logistic regression to classify customer fault reports by category (outage, billing, meter fault, streetlight, cable damage) and urgency.
- **Chatbot**: Features an intent-based conversational assistant that logs tickets, gives safety advice for damaged cables and reports the live demand forecast.
- **Voice Interaction**: Supports speech recognition (via SpeechRecognition) and speech synthesis (via pyttsx3 or gTTS), falling back to text when voice is unavailable.
- **Interactive Dashboard**: Presents forecasts, the fault classifier, the chatbot and model results in a Streamlit app.
- **Comprehensive Analytics Pipeline**: Integrates all components into a single executable pipeline that also generates the project poster and documentation.

## Installation

1. **Prerequisites**: Ensure you have Python 3.9+ installed on your system.

2. **Clone the Repository**:
   ```
   git clone https://github.com/BA-Practical-Assignment/GridWise.git
   cd GridWise
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

   The required packages include:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - joblib
   - python-docx
   - reportlab
   - pillow

   Optional packages:
   - torch (LSTM forecaster)
   - statsmodels (SARIMA)
   - holidays (public-holiday calendar)
   - streamlit (dashboard)
   - SpeechRecognition, pyttsx3, gTTS, pyaudio (voice chat)

4. **Run the Pipeline**:
   ```
   python run_pipeline.py --demand data/AEP_hourly.csv
   ```

5. **Run the Application**:
   ```
   python cli_chatbot.py      
   streamlit run app.py       
   ```


