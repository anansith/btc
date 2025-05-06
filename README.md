# btc_prediction_final.ipynb

## Overview

This research aims to study the efficiency of Machine Learning models in predicting Bitcoin price trends by transforming the problem into a classification based on next-day closing price changes. Specifically, price increases exceeding the mean by 0.3 standard deviations are labeled as "Buy," decreases beyond 0.3 standard deviations below the mean as "Sell," and changes within this range as "Hold." The dataset comprises Bitcoin prices and Funding Rate from Binance API, gold prices and NASDAQ index from YFinance, along with generated technical indicators (MACD, RSI, ATR). The study compares four model types: Decision Tree, Random Forest, XGBoost, and LSTM, using Grid Search and K-Fold Cross-Validation to determine optimal parameters, except for LSTM which was configured by the researcher. Results demonstrate that LSTM achieves the highest performance across all metrics, followed by XGBoost, Random Forest, and Decision Tree, respectively. As LSTM can better learn time sequences and deep data relationships, it is particularly suitable for application in decision support systems for cryptocurrency investment markets.


## Contents

The notebook includes the following sections:

1.  **Import Libraries:** Imports necessary Python libraries such as pandas, scikit-learn, matplotlib, seaborn, `binance.client`, `yfinance`, `dotenv`, and etc.
2.  **Load Data:**
    * **Binance API:** Retrieves historical Klines (candlestick data) and funding rates for Bitcoin (BTCUSDT). It loads API keys from a `.env` file (you need to create this file with your Binance API key and secret). See the [Binance API documentation](https://www.binance.com/en/binance-api) for details.
    * **Yahoo Finance API:** Fetches additional historical data for BTC-USD using the `yfinance` library.
3.  **Data Preprocessing:** Steps to clean and prepare the combined data for modeling. This involves:
    * Feature engineering:
        * Extracting features from the timestamp.
        * Calculating various technical indicators (e.g., RSI, MACD, ATR).
        * Classification Labeling: The target variable, which defines the "Buy," "Hold," and "Sell" classes, is generated based on the percentage change of the next day's closing price. In essence, the process calculates the mean and standard deviation of the daily percentage price changes. If the next day's price change is greater than the mean plus 0.3 standard deviations, it's labeled as a "Buy" signal. If it's less than the mean minus 0.3 standard deviations, it's a "Sell" signal. Otherwise, it's a "Hold" signal.
    * Splitting   the data into training and testing sets. **The data is split into an 80% training set and a 20% testing set.**
4.  **Model Selection:** The notebook implements and evaluates several machine learning models for classification problem:
    * **Decision Tree Classifier**
    * **Random Forest Classifier**
    * **XGBoost Classifier**
    * **Long Short-Term Memory (LSTM) Neural Network**
        The performance of each model is compared based on evaluation metrics.
5.  **Model Training:** Code to train each of the selected models using the training data. The LSTM model involves specific data reshaping for sequential input.
6.  **Model Evaluation:** Evaluation of the trained models on the testing data using *classification* metrics. The notebook calculates and prints metrics such as accuracy, precision, recall, F1-score, and also includes a confusion matrix to analyze the classification performance.
7.  **Results and Visualization:**
    * Display   of the *classification* evaluation metrics for all tested models
    * Confusion   matrix to visualize the classification accuracy.
8.  **Conclusion:** A summary of the model performance.

## How to Use

To run this notebook, you will need:

* **Python 3** installed on your system.
* The following Python libraries (you can install them using pip):

    ```bash
    pip install pandas scikit-learn matplotlib seaborn binance-python yfinance python-dotenv tensorflow
    ```
* A **Binance account** and your **API key and secret**.

    * If you don't have a Binance account, you can sign up here: [Binance Registration](https://accounts.binance.com/en/register).
    * After signing up, you can find instructions on how to generate API keys in the Binance API documentation: [Binance API](https://www.binance.com/en/binance-api). Look for the "How to Get Started" or "API Keys" section.
* Create a `.env` file in the same directory as the notebook and store your Binance API credentials:

    ```
    BINANCE_API_KEY=YOUR_BINANCE_API_KEY
    BINANCE_API_SECRET=YOUR_BINANCE_API_SECRET
    ```

**Steps:**

1.  Clone or download this repository (if applicable).
2.  Ensure the `.env` file with your Binance API credentials is correctly set up.
3.  Open the `btc_prediction_final.ipynb` file using Jupyter Notebook.
4.  Run the cells sequentially to execute the code and see the results.

## Potential Improvements

This section could outline potential areas for future work, such as:

* Further hyperparameter tuning for the models.
* Exploring more advanced time series models or deep learning architectures.
* Incorporating sentiment analysis or other external data sources.
* Developing a trading strategy based on the predictions.

## Author

[Your Name/Organization - if applicable]
