import pandas as pd
import numpy as np
import datetime

from django.contrib.admin import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from .models import finnish_stock_daily, signals


def train_machine_learning_model():
    # Load data from the finnish_stock_daily and signals models into a pandas DataFrame
    df_daily = pd.DataFrame.from_records(finnish_stock_daily.objects.all().values())
    df_signals = pd.DataFrame.from_records(signals.objects.all().values())
    df = pd.merge(df_daily, df_signals, left_on=['symbol', 'id'], right_on=['symbol', 'stock_id'])
    df = df.reset_index()

    # Preprocess the data
    scaler = MinMaxScaler()
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
        df[['open', 'high', 'low', 'close', 'volume']])

    # Engineer features from the preprocessed data
    df['return'] = df['close'].pct_change()
    df['volume_return'] = df['volume'].pct_change()
    df = df.dropna()

    # Split the data into training and test sets
    split_date = datetime.date(2017, 1, 1)
    train_data = df[df['date'] < split_date]
    test_data = df[df['date'] >= split_date]

    # Train a Random Forest Regressor on the training set
    features = ['open', 'high', 'low', 'close', 'volume', 'adl', 'obv', 'bbp8', 'adx',
                'rsi7', 'rsi14', 'rsi50', 'aroon_up', 'aroon_down', 'macd', 'macd_signal', 'std_dev', 'ema20', 'ema50',
                'ema100', 'ema200']
    target = 'close'

    X_train = train_data[features]
    y_train = train_data[target]

    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)

    X_test = test_data[features]
    y_test = test_data[target]
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


def train_machine_learning_model_future_values():
    # Load data from the finnish_stock_daily and signals models into a pandas DataFrame
    df_daily = pd.DataFrame.from_records(finnish_stock_daily.objects.all().values())
    df_signals = pd.DataFrame.from_records(signals.objects.all().values())
    df = pd.merge(df_daily, df_signals, left_on=['symbol', 'id'], right_on=['symbol', 'stock_id'])

    # Preprocess the data
    # scaler = MinMaxScaler()
    # df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
    #     df[['open', 'high', 'low', 'close', 'volume']])

    # Engineer features from the preprocessed data
    df['return'] = df['close'].pct_change()
    df['volume_return'] = df['volume'].pct_change()
    df = df.dropna()

    n_future_days = 0
    df['future_close'] = df['close'].shift(-n_future_days)
    df = df.dropna()
    split_date = datetime.date(2023, 1, 1)
    train_data = df[df['date'] < split_date]
    test_data = df[df['date'] >= split_date]

    # Train a Random Forest Regressor on the training set
    features = ['open', 'high', 'low', 'close', 'volume', 'adl', 'obv', 'bbp8', 'adx',
                'rsi7', 'rsi14', 'rsi50', 'aroon_up', 'aroon_down', 'macd', 'macd_signal', 'std_dev', 'ema20', 'ema50',
                'ema100', 'ema200']
    target = 'future_close'

    X_train = train_data[features]
    y_train = train_data[target]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    # Evaluate the model on the test set
    X_test = test_data[features]
    y_test = test_data[target]
    y_pred = model.predict(X_test)

    # Calculate the mean absolute error
    mae = np.mean(np.abs(y_pred - y_test))
    symbols = test_data['symbol']
    dates = test_data['date']
    results = pd.DataFrame({'symbol': symbols, 'predicted_close': y_pred, 'date': dates, 'actual_close': y_test})
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(results)

    r2 = r2_score(y_test, y_pred)
    print("R^2 value:", r2)
    print("First 5 predicted values:", y_pred[:5])
    print("Mean Absolute Error train future values:", mae)


# def use_trained_machine_learning_model():
#     # Load data from the finnish_stock_model and signals into a pandas DataFrame
#     df_daily = pd.DataFrame.from_records(finnish_stock_daily.objects.all().values())
#     df_signals = pd.DataFrame.from_records(signals.objects.all().values())
#     df = pd.merge(df_daily, df_signals, on=['stock_id', 'date'])
#
#     # Preprocess the data
#     scaler = MinMaxScaler()
#     df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
#         df[['open', 'high', 'low', 'close', 'volume']])
#
#     # Engineer features from the preprocessed data
#     df['return'] = df['close'].pct_change()
#     df['volume_return'] = df['volume'].pct_change()
#     df = df.dropna()
#
#     # Train a machine learning model on the preprocessed data
#     # Code for training the model should be added here
#
#     # Prepare the input data for prediction
#     # The input data should have the same number of columns and same scaling as the training data
#     future_data = ...  # input data for prediction
#     scaled_input_data = scaler.transform(future_data)
#
#     # Use the trained model to predict the values
#     predicted_values = model.predict(scaled_input_data)
#
#     # Post-process the output if necessary
#     # In this case, if the input features were scaled, the predicted values need to be inverse-transformed to get the original scale
#     original_scale_predictions = scaler.inverse_transform(predicted_values)
