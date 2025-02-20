import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from newsapi import NewsApiClient  # Install: pip install newsapi-python
import warnings
import os
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import pytz

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# NewsAPI Key
NEWS_API_KEY = 'abf7838a31954f7bac34118a7daf7daa'  # Replace with your NewsAPI key

# SECTORS = ["Technology", "Healthcare", "Finance", "Energy", "Consumer Discretionary", "Industrials"]
def fetch_sp500_data():
    """Fetch S&P 500 symbols, names, and sectors from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})
    rows = table.find_all('tr')
    sp500_data = []

    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) > 0:
            symbol = cols[0].text.strip()
            name = cols[1].text.strip()
            sector = cols[2].text.strip()
            sp500_data.append({'Symbol': symbol, 'Name': name, 'Sector': sector})

    return pd.DataFrame(sp500_data)

def fetch_nikkei225_data():
    """Fetch Nikkei 225 symbols, names, and sectors."""
    url = "https://indexes.nikkei.co.jp/en/nkave/index/component"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch Nikkei 225 data. Status code: {response.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, 'html.parser')
    company_codes, company_names, sectors = [], [], []
    current_sector = None

    for element in soup.find_all(['h3', 'table']):
        if element.name == 'h3':
            current_sector = element.text.strip()
        elif element.name == 'table':
            rows = element.find_all('tr')
            for row in rows[1:]:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    company_code = cols[0].text.strip() + '.T'
                    company_name = cols[1].text.strip()
                    company_codes.append(company_code)
                    company_names.append(company_name)
                    sectors.append(current_sector)

    return pd.DataFrame({'Symbol': company_codes, 'Name': company_names, 'Sector': sectors})

class MarketPredictor:
    def _init_(self, prediction_days=90):
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler()
        self.model = None

    def fetch_data(self, symbols, start_date):
        """Fetch stock data from Yahoo Finance"""
        try:
            data = {}
            for symbol in symbols:
                symbol_data = yf.download(symbol, start=start_date, end=datetime.now())
                if symbol_data.empty:
                    st.warning(f"No data found for {symbol}. Skipping.")
                    continue
                data[symbol] = symbol_data
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return {}

    def prepare_data(self, data):
        """Prepare data for LSTM model"""
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        X, y = [], []
        for x in range(self.prediction_days, len(scaled_data)):
            X.append(scaled_data[x - self.prediction_days:x, 0])
            y.append(scaled_data[x, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        return X_train, y_train, X_test, y_test, scaled_data

    def build_and_train_model(self, X_train, y_train):
        """Build and train LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        self.model = model
        return model

    def predict_with_arima(self, data, days):
        """Predict using ARIMA"""
        arima_model = ARIMA(data['Close'], order=(5, 1, 0))
        arima_result = arima_model.fit()
        forecast = arima_result.forecast(steps=days)
        return forecast

    def predict_with_lstm(self, data, days):
        """Predict using LSTM"""
        last_data = data['Close'].values[-self.prediction_days:]
        scaled_last_data = self.scaler.transform(last_data.reshape(-1, 1))

        future_predictions = []
        current_batch = scaled_last_data.reshape((1, self.prediction_days, 1))

        for _ in range(days):
            future_price = self.model.predict(current_batch, verbose=0)[0]
            future_predictions.append(future_price)
            current_batch = np.append(current_batch[:, 1:, :], [[future_price]], axis=1)

        return self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

class ImprovedMarketPredictor:
    def _init_(self, prediction_days=90):
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler()
        self.model = None

# ... Remaining methods (fetch_news, analyze_sentiment, etc.) remain unchanged from your provided code
def fetch_news(symbol):
    """Fetch recent news for a stock symbol."""
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    try:
        query = f"{symbol} stock"
        articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=5)
        return articles['articles']
    except Exception as e:
        st.warning(f"Could not fetch news for {symbol}: {str(e)}")
        return []



def analyze_sentiment(text):
    """Analyze sentiment of a text using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']  # Extract the compound score
    return compound_score

def display_news_and_sentiment(symbol, articles):
    """Display news articles and sentiment analysis."""
    st.markdown(f"### 📰 Recent News and Sentiment Analysis for {symbol}")

    sentiments = []
    for article in articles:
        title = article.get('title', 'No title available')
        description = article.get('description', 'No description available')
        url = article.get('url', '#')

        try:
            sentiment = analyze_sentiment(f"{title} {description}")
            sentiments.append(sentiment)
            sentiment_label = (
                "Positive" if sentiment > 0 else 
                "Negative" if sentiment < 0 else 
                "Neutral"
            )
        except Exception as e:
            st.error(f"Error in sentiment prediction for {symbol}: {str(e)}")
            sentiment_label = "Error"

        st.markdown(f"[{title}]({url})")
        st.write(f"Description: {description}")
        st.write(f"Sentiment: {sentiment_label} ({sentiment:.2f})")
        st.write("---")

    if sentiments:
        avg_sentiment = np.mean(sentiments)
        sentiment_label = (
            "Overwhelmingly Positive" if avg_sentiment > 0.5 else
            "Positive" if avg_sentiment > 0.1 else
            "Neutral" if -0.1 <= avg_sentiment <= 0.1 else
            "Negative" if avg_sentiment > -0.5 else
            "Overwhelmingly Negative"
        )
        st.markdown(f"<h4>Overall Sentiment: <b>{sentiment_label}</b> ({avg_sentiment:.2f})</h4>", unsafe_allow_html=True)
        
        recommendation = "Buy" if avg_sentiment > 0.1 else "Sell" if avg_sentiment < -0.1 else "Hold"
        st.markdown(f"<h4>Recommendation for {symbol}: <b>{recommendation}</b></h4>", unsafe_allow_html=True)

    return avg_sentiment, recommendation


def calculate_var(data, confidence_level=0.95):
    """Calculate Value at Risk (VaR) at a specified confidence level."""
    return np.percentile(data['Close'].pct_change().dropna(), 1 - confidence_level) * data['Close'].iloc[-1]

def stress_test(data, drop_percentage=0.1):
    """Perform a stress test assuming a price drop."""
    return data['Close'].iloc[-1] * (1 - drop_percentage)

def profit_chance(data):
    """Calculate the profit chance based on the last 30 days."""
    price_change = data['Close'].pct_change(periods=30).iloc[-1]
    return price_change * 100





def plot_predictions_combined(data, predictions, symbols, model_used, prediction_day, years):
    """Plot predictions of multiple stocks in a single graph for comparison with enhanced features"""
    fig = go.Figure()

    # Plot historical data for each stock
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']  # Add more if needed
    for idx, (symbol, stock_data) in enumerate(data.items()):
        # Ensure the historical start date is a timestamp (with or without timezone)
        historical_start_date = datetime.now() - timedelta(days=365 * years)
        historical_start_date = pd.Timestamp(historical_start_date)

        # If stock_data.index is timezone-naive, make it aware by localizing it to UTC
        if stock_data.index.tz is None:
            stock_data.index = stock_data.index.tz_localize('UTC')

        # If historical_start_date is timezone-aware, make sure it is in the same timezone
        if historical_start_date.tz is None and stock_data.index.tz is not None:
            historical_start_date = historical_start_date.tz_localize('UTC')

        # Filter stock data based on historical start date
        filtered_data = stock_data[stock_data.index >= historical_start_date]

        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data['Close'].values,
            mode='lines',
            name=f'{symbol} Historical Data',
            line=dict(color=colors[idx % len(colors)], width=2)
        ))

    # Generate future dates for predictions
    future_dates = pd.date_range(start=data[symbols[0]].index[-1], periods=prediction_day + 1)[1:]

    # Plot predictions for each stock
    for idx, (symbol, prediction) in enumerate(zip(symbols, predictions)):
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=np.ravel(prediction),
            mode='lines',
            name=f'{symbol} {model_used} Predictions',
            line=dict(color=colors[idx % len(colors)], dash='dash', width=2)
        ))

    # Enhance the layout with zooming and range selection
    fig.update_layout(
        title=f'Comparison of Stock Predictions using {model_used}',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),  # Enable range slider
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis=dict(
            fixedrange=False  # Allow zooming on the y-axis
        )
    )

    return fig


def display_predictions_table(predictions, symbols, prediction_days):
    """Display a table of predicted stock prices with dates in rows and companies in columns."""
    try:
        # Create prediction dates
        prediction_dates = pd.date_range(start=datetime.now(), periods=prediction_days)

        # Ensure predictions is a list of arrays
        predictions = [np.ravel(pred) for pred in predictions]

        # Pad or truncate predictions to match prediction_days
        predictions = [
            pred[:prediction_days] if len(pred) >= prediction_days 
            else np.pad(pred, (0, prediction_days - len(pred)), mode='constant', constant_values=np.nan)
            for pred in predictions
        ]

        # Create DataFrame
        prediction_table = pd.DataFrame(dict(zip(symbols, predictions)))
        prediction_table.insert(0, "Date", prediction_dates)

        # Display the prediction table
        st.subheader("Predicted Stock Prices")
        st.dataframe(prediction_table)

        # Option to Download the Prediction Table
        csv = prediction_table.to_csv(index=False)
        st.download_button(
            label="Download Prediction Table as CSV",
            data=csv,
            file_name="predicted_values.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"An error occurred while displaying predictions: {str(e)}")
        st.error(f"Predictions: {predictions}")
        st.error(f"Symbols: {symbols}")
        st.error(f"Prediction Days: {prediction_days}")


def predict_next_day_based_on_sentiment(current_price, sentiment_score):
    """Predict the next day's price based on sentiment analysis."""
    # Adjust multiplier as necessary to fine-tune impact
    sentiment_multiplier = 0.005  
    predicted_change = current_price * sentiment_score * sentiment_multiplier
    next_day_price = current_price + predicted_change
    return next_day_price


def main():
   
    
    st.set_page_config(page_title="Stock Market Prediction (ARIMA & LSTM)", layout="wide")

    st.title("📊 Stock Market Prediction App")

    st.sidebar.header("Parameters")
    
    # Market Selection
    st.sidebar.markdown("### Market Selection")
    markets = st.sidebar.multiselect("Select Markets", ["S&P 500", "Nikkei 225"])

    # Fetch data based on selected markets
    market_data = {}
    if "S&P 500" in markets:
        sp500_data = fetch_sp500_data()
        market_data["S&P 500"] = sp500_data
    if "Nikkei 225" in markets:
        nikkei_data = fetch_nikkei225_data()
        market_data["Nikkei 225"] = nikkei_data

    # Dynamic filtering per market
    selected_stocks = {}
    for market_name, data in market_data.items():
        st.sidebar.markdown(f"#### Select Sector for {market_name}")
        sectors = data['Sector'].unique()
        selected_sector = st.sidebar.selectbox(f"Sector for {market_name}", sectors, key=market_name)

        # Filter stocks by sector
        filtered_stocks = data[data['Sector'] == selected_sector]
        stock_symbols = filtered_stocks['Symbol'].tolist()

        # Select stocks
        selected_stocks[market_name] = st.sidebar.multiselect(
             f"Select Stocks for {market_name}", 
            [f"{stock['Symbol']} - {stock['Name']}" for _, stock in filtered_stocks.iterrows()],
            key=f"stocks_{market_name}"
        )

    # Combine all selected stocks across markets
    symbols = [stock.split(' - ')[0] for stocks in selected_stocks.values() for stock in stocks]
    
    # Validation
    if not symbols:
        st.error("Please select at least one stock.")
        return

    years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)
    prediction_day = st.sidebar.number_input(
        "Enter the day for prediction (1-90):",
        min_value=1, max_value=90, value=7, step=1
    )

    st.sidebar.markdown("Model Selection: Automatically chooses ARIMA for days 1-30 and LSTM for days 30-90.")

    if st.button("Run Prediction"):
        with st.spinner("Fetching data and generating predictions..."):
            try:
                predictor = MarketPredictor()
                start_date = datetime.now() - timedelta(days=365 * years)
                data = predictor.fetch_data(symbols, start_date)

                if not data:
                    st.error("No data found for the given symbols. Please check the symbols and try again.")
                    return

                st.session_state.data = data

                predictions = []
                prediction_dates = pd.date_range(start=datetime.now(), periods=prediction_day)

                col1, col2 = st.columns([0.7, 0.3])

                with col1:
                    st.subheader("Stock Prediction Analysis")
                    for symbol, stock_data in data.items():
                        try:
                            # Prediction for ARIMA if days are 1-30, else LSTM for days >30
                            if prediction_day <= 30:
                                prediction = predictor.predict_with_arima(stock_data, prediction_day)
                                model_used = "ARIMA"
                            else:
                                X_train, y_train, X_test, y_test, scaled_data = predictor.prepare_data(stock_data)
                                predictor.build_and_train_model(X_train, y_train)
                                prediction = predictor.predict_with_lstm(stock_data, prediction_day)
                                model_used = "LSTM"
                            predictions.append(prediction)
                        except Exception as e:
                            st.warning(f"Prediction for {symbol} failed: {str(e)}")
                            predictions.append(np.full(prediction_day, np.nan))  # Placeholder for failed predictions

                    # Ensure predictions are compatible
                    processed_predictions = [
                        np.ravel(pred)[:prediction_day] if len(np.ravel(pred)) > 0 
                        else np.full(prediction_day, np.nan)
                        for pred in predictions
                    ]

                    # Plotting predictions
                    st.plotly_chart(
                        plot_predictions_combined(data, processed_predictions, symbols, model_used, prediction_day, years),
                        use_container_width=True
                    )

                    # Prediction Table
                    st.subheader("Prediction Table")
                    
                    # Create prediction table
                    prediction_data = {
                        "Date": prediction_dates,
                        **dict(zip(symbols, processed_predictions))
                    }
                    prediction_table = pd.DataFrame(prediction_data)
                    
                    # Display prediction table
                    st.dataframe(prediction_table)

                    # Download option for prediction table
                    csv = prediction_table.to_csv(index=False)
                    st.download_button(
                        label="Download Prediction Table as CSV",
                        data=csv,
                        file_name="predicted_values.csv",
                        mime="text/csv"
                    )

                    # Sentiment-Based Next Day Prediction
                    st.subheader("Sentiment-Based  Prediction")
                    for symbol, stock_data in data.items():
                        try:
                            current_price = float(stock_data['Close'].iloc[-1])
                            articles = fetch_news(symbol)
                            avg_sentiment, _ = display_news_and_sentiment(symbol, articles)

                            next_day_price = predict_next_day_based_on_sentiment(current_price, avg_sentiment)

                            st.metric(
                                label=f"Next Day Prediction ({symbol})",
                                value=f"${next_day_price:.2f}",
                                delta=f"Sentiment: {avg_sentiment:.2f}"
                            )
                            st.markdown("<hr>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error in sentiment prediction for {symbol}: {e}")

                with col2:
                    st.subheader("Market Statistics")

                    for symbol, stock_data in data.items():
                        try:
                            # Safely extract prices
                            current_price = float(stock_data['Close'].iloc[-1])
                            previous_price = float(stock_data['Close'].iloc[-2])

                            # Calculate price change and percentage
                            price_change = current_price - previous_price
                            price_change_pct = (price_change / previous_price) * 100

                            # Display current price metric
                            st.metric(
                                label=f"Current Price ({symbol})",
                                value=f"${current_price:.2f}",
                                delta=f"{price_change_pct:.2f}%"
                            )

                            # Performance metrics
                            st.markdown(f"### Recent Performance for {symbol}")
                            st.write(f"30-Day High: ${float(stock_data['High'].tail(30).max()):.2f}")
                            st.write(f"30-Day Low: ${float(stock_data['Low'].tail(30).min()):.2f}")
                            
                            # Convert and format volume
                            avg_volume = int(stock_data['Volume'].tail(30).mean())
                            st.write(f"30-Day Avg Volume: {avg_volume:,}")

                            # Risk and profit analysis
                            var = calculate_var(stock_data)
                            st.write(f"Value at Risk (VaR) 95% Confidence: ${var:.2f}")

                            stress_price = stress_test(stock_data)
                            st.write(f"Stress Test Price (10% drop): ${stress_price:.2f}")

                            profit_percentage = profit_chance(stock_data)
                            st.write(f"Profit Chance (30 days): {profit_percentage:.2f}%")

                            st.write("---")

                        except Exception as e:
                            st.error(f"Error processing statistics for {symbol}: {e}")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                # Optional: Log the full traceback for debugging
                import traceback
                st.error(traceback.format_exc())


if __name__ == "_main_":
    main()