# btc_prediction_final.ipynb

## Overview

This Jupyter Notebook contains the final implementation of a Bitcoin price prediction model. It includes steps for data loading from Binance and Yahoo Finance APIs, preprocessing, model selection, training, evaluation, and visualization of the results.

## Contents

The notebook includes the following sections:

1.  **Import Libraries:** Imports necessary Python libraries such as pandas, scikit-learn, matplotlib, seaborn, `binance.client`, `yfinance`, `dotenv`, and `ta` (for technical indicators).
2.  **Load Data:**
    * **Binance API:** Retrieves historical Klines (candlestick data) and funding rates for Bitcoin (BTCUSDT) using the `binance-python` library. It loads API keys from a `.env` file (you need to create this file with your Binance API key and secret). The notebook defines functions `get_historical_klines` and `get_historical_funding_rate` to fetch this data, handling pagination and including a delay to respect rate limits. See the [Binance API documentation](https://www.binance.com/en/binance-api) for details.
    * **Yahoo Finance API:** Fetches additional historical data for BTC-USD using the `yfinance` library. The `get_yfinance_data` function is used for this purpose.
3.  **Data Preprocessing:** Steps to clean and prepare the combined data for modeling. This involves:
    * Handling missing values (using forward fill).
    * Feature engineering:
        * Extracting features from the timestamp.
        * Calculating various technical indicators using the `ta` library (e.g., Simple Moving Average, Exponential Moving Average, RSI, MACD, Bollinger Bands, Ichimoku Cloud components, Average True Range, Volume indicators).
    * Scaling numerical features using `MinMaxScaler`.
    * Splitting the data into training and testing sets based on a specified split ratio.
4.  **Model Selection:** The notebook implements and evaluates several machine learning models for time series forecasting:
    * **Linear Regression**
    * **Ridge Regression**
    * **Lasso Regression**
    * **Support Vector Regression (SVR)**
    * **Random Forest Regressor**
    * **Gradient Boosting Regressor**
    * **Long Short-Term Memory (LSTM) Neural Network** (using TensorFlow/Keras)
        The performance of each model is compared based on evaluation metrics.
5.  **Model Training:** Code to train each of the selected models using the training data. The LSTM model involves specific data reshaping for sequential input.
6.  **Model Evaluation:** Evaluation of the trained models on the testing data using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared. The notebook calculates and prints these metrics for each model.
7.  **Results and Visualization:**
    * Visualizations comparing the actual Bitcoin prices with the predicted prices for the best-performing model (likely LSTM).
    * Plots of individual technical indicators.
    * Display of the evaluation metrics for all tested models in a pandas DataFrame.
8.  **Conclusion:** A summary of the model performance and potential future directions.

## How to Use

To run this notebook, you will need:

* **Python 3** installed on your system.
* The following Python libraries (you can install them using pip):

    ```bash
    pip install pandas scikit-learn matplotlib seaborn binance-python yfinance python-dotenv ta tensorflow
    ```
* A **Binance account** and your **API key and secret**.

    * If you don't have a Binance account, you can sign up here: [Binance Registration](https://accounts.binance.com/en/register).
    * After signing up, you can find instructions on how to generate API keys in the Binance API documentation: [Binance API](https://www.binance.com/en/binance-api). Look for the "How to Get Started" or "API Keys" section.
* Create a `.env` file in the same directory as the notebook (or adjust the path in the "Load Data" section) and store your Binance API credentials:

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
* Exploring more advanced time series models or deep learning architectures (e.g., Transformer networks).
* Incorporating sentiment analysis or other external data sources.
* Implementing cross-validation for more robust evaluation.
* Developing a trading strategy based on the predictions.

## Author

[Your Name/Organization - if applicable]
