import math
from datetime import timedelta
from typing import List
import numpy as np
import pandas as pd
import django
from django.db import connection
from django.core.exceptions import ObjectDoesNotExist
from django.utils import timezone
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.exponential_moving_average import exponential_moving_average as ema
from .models import finnish_stock_daily, signals, optimal_buy_sell_points, reverse_signals


def calculate_BBP8(stock_symbol, model, reverse, period, daterange):
    # BBP8 Calculation
    if reverse:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('-date')[:daterange]
    else:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('date')

    for i in range(len(stock_data)):
        if i < period:
            if reverse:
                reverse_signal, _ = reverse_signals.objects.get_or_create(stock=stock_data[i], symbol=stock_symbol)
                reverse_signal.bbp8 = 0
                reverse_signal.save()
                continue
            else:
                signal, _ = signals.objects.get_or_create(stock=stock_data[i], symbol=stock_symbol)
                signal.bbp8 = 0
                signal.save()
                continue

        close_prices = [stock.close for stock in stock_data[i - period:i]]
        upper_band = max(close_prices)
        lower_band = min(close_prices)

        if upper_band != lower_band and stock_data[i].close != lower_band:
            BBP8 = (stock_data[i].close - lower_band) / (upper_band - lower_band)
        else:
            BBP8 = 0
        # Check if the entry already exists before updating or creating it
        if reverse:
            reverse_signal, _ = reverse_signals.objects.get_or_create(stock=stock_data[i], symbol=stock_symbol)
            reverse_signal.bbp8 = BBP8
            reverse_signal.save()
        else:
            signal, _ = signals.objects.get_or_create(stock=stock_data[i], symbol=stock_symbol)
            signal.bbp8 = BBP8
            signal.save()


def calculate_obv(stock_symbol, model, reverse):
    if reverse:
        stock_data = list(model.objects.filter(symbol=stock_symbol).values('date', 'close', 'volume').order_by('-date'))
    else:
        stock_data = list(model.objects.filter(symbol=stock_symbol).values('date', 'close', 'volume').order_by('date'))
    df = pd.DataFrame(stock_data)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(int)
    df['change'] = df['close'] - df['close'].shift(1)
    df['obv'] = np.where(df['change'] > 0, df['volume'], -df['volume'])
    df['obv'] = df['obv'].cumsum()
    for index, row in df.iterrows():
        stock = model.objects.get(date=row['date'], symbol=stock_symbol)
        if reverse:
            if not reverse_signals.objects.filter(stock__id=index, obv__isnull=False).exists():
                reverse_signals_obj = reverse_signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                if reverse_signals_obj:
                    reverse_signals_obj.obv = row['obv']
                    reverse_signals_obj.save()
        else:
            if not signals.objects.filter(stock__id=index, obv__isnull=False).exists():
                signals_obj = signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                if signals_obj:
                    signals_obj.obv = row['obv']
                    signals_obj.save()


def calculate_adl(stock_symbol, model, reverse):
    if reverse:
        stock_data = list(
            model.objects.filter(symbol=stock_symbol).values('date', 'close', 'open', 'volume').order_by('-date'))
    else:
        stock_data = list(
            model.objects.filter(symbol=stock_symbol).values('date', 'close', 'open', 'volume').order_by('date'))
    df = pd.DataFrame(stock_data)
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['adl'] = (df['close'] - df['open']) * df['volume']
    df['adl'] = df['adl'].cumsum()
    for index, row in df.iterrows():
        stock = model.objects.get(date=row['date'], symbol=stock_symbol)
        if reverse:
            if not reverse_signals.objects.filter(stock__id=index, adl__isnull=False).exists():
                reverse_signals_obj = reverse_signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                if reverse_signals_obj:
                    reverse_signals_obj.adl = row['adl']
                    reverse_signals_obj.save()
        else:
            if not signals.objects.filter(stock__id=index, adl__isnull=False).exists():
                signals_obj = signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                if signals_obj:
                    signals_obj.adl = row['adl']
                    signals_obj.save()


def calculate_adx(stock_symbol, model, reverse, period, daterange):
    if reverse:
        additional = 2
        reversed_stock_data = list(
            model.objects.filter(symbol=stock_symbol)
            .order_by('date')[:daterange + 1])[::-1]
        stock_data = reversed_stock_data[:period + additional]
    else:
        reversed_stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('-date').reverse()[
                              :daterange + 1]
        stock_data = reversed_stock_data[0 if daterange < 27 else daterange - 27 + 1:]

    if reverse and len(stock_data) >= period + 2 or not reverse and len(stock_data) >= 27:
        if reverse:
            last_stock = stock_data[period - 1]
        else:
            last_stock = stock_data[27 - 1]

        df = pd.DataFrame({
            'date': [entry.date for entry in stock_data],
            'open': [entry.open for entry in stock_data],
            'high': [entry.high for entry in stock_data],
            'low': [entry.low for entry in stock_data],
            'close': [entry.close for entry in stock_data],
            'volume': [entry.volume for entry in stock_data],
        })
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        # Calculating true range
        df['tr'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
        # Calculating +DI and -DI
        df['+DI'] = 100 * (df['close'].diff() > 0) * df['volume'] / df['tr']
        df['-DI'] = 100 * (df['close'].diff() < 0) * df['volume'] / df['tr']
        df['+DI'] = df['+DI'].rolling(period).mean()
        df['-DI'] = df['-DI'].rolling(period).mean()
        # Calculating DX
        df['DX'] = 100 * (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'])
        # Calculating ADX
        df['ADX'] = df['DX'].rolling(period).mean()

        # Get the last ADX value
        last_row = df.iloc[-1]

        # Save ADX value to the appropriate model
        if reverse:
            reverse_signals_obj = reverse_signals.objects.filter(stock=last_stock, symbol=stock_symbol).first()
            if reverse_signals_obj.adx is None or reverse_signals_obj.adx == 0:
                reverse_signals_obj.adx = last_row['ADX']
                reverse_signals_obj.save()
        else:
            signals_obj = signals.objects.filter(stock=last_stock, symbol=stock_symbol).first()
            if signals_obj.adx is None or signals_obj.adx == 0:
                signals_obj.adx = last_row['ADX']
                signals_obj.save()


def calculate_aroon(stock_symbol, daterange, period=25):
    reversed_stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('date')[:daterange + 1]
    stock_data = reversed_stock_data[0 if daterange < period else daterange - period + 1:]
    if len(stock_data) >= period:
        # Get the high and low prices for the entire period
        high_prices = [stock.high for stock in stock_data]
        low_prices = [stock.low for stock in stock_data]

        # Find the index of the maximum and minimum prices within the period
        max_index = high_prices.index(max(high_prices))
        min_index = low_prices.index(min(low_prices))

        # Calculate aroon_up and aroon_down for the entire period
        aroon_up = ((period - max_index) / period) * 100
        aroon_down = ((period - min_index) / period) * 100

        # Save the calculated Aroon values
        signal, created = signals.objects.get_or_create(stock=stock_data[period - 1], symbol=stock_symbol)
        if created or signal.aroon_up is None or signal.aroon_up == 0:
            signal.aroon_up = aroon_up
            signal.aroon_down = aroon_down
            signal.save()


def calculate_ema(stock_symbol, model, reverse, period, daterange):
    if reverse:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('-date')
    else:
        reversed_stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('date')[:daterange + 1]
        stock_data = reversed_stock_data[0 if daterange < period else daterange - period + 1:]

    k = 2 / (period + 1)
    if len(stock_data) >= period:
        # Initialize EMA with the first closing price
        exp_mov_av = stock_data[0].close

        # Calculate EMA for the last date based on the first and last closing prices
        last_stock = stock_data[period - 1]
        exp_mov_av = (last_stock.close * k) + (exp_mov_av * (1 - k))

        # Update the appropriate EMA field based on the period
        if period == 5:
            reverse_signals_obj = reverse_signals.objects.filter(stock=last_stock, symbol=stock_symbol).first()
            if reverse_signals_obj:
                reverse_signals_obj.ema5 = exp_mov_av
                reverse_signals_obj.save()
        elif period == 20:
            signals_obj = signals.objects.filter(stock=last_stock, symbol=stock_symbol).first()
            if signals_obj.ema20 is None:
                signals_obj.ema20 = exp_mov_av
                signals_obj.save()
        elif period == 50:
            signals_obj, created = signals.objects.get_or_create(stock=last_stock, symbol=stock_symbol)
            if signals_obj.ema50 is None:
                signals_obj.ema50 = exp_mov_av
                signals_obj.save()
        elif period == 100:
            signals_obj = signals.objects.filter(stock=last_stock, symbol=stock_symbol).first()
            if signals_obj.ema100 is None:
                signals_obj.ema100 = exp_mov_av
                signals_obj.save()
        elif period == 200:
            signals_obj, created = signals.objects.get_or_create(stock=last_stock, symbol=stock_symbol)
            if signals_obj.ema200 is None:
                signals_obj.ema200 = exp_mov_av
                signals_obj.save()


def macd(prices: List[float], fast_period: int, slow_period: int) -> List[float]:
    fast_ema = ema(prices, fast_period)
    slow_ema = ema(prices, slow_period)
    macd_values = [fast_ema[i] - slow_ema[i] for i in range(len(fast_ema))]
    return macd_values


def ema(prices: List[float], period: int) -> List[float]:
    ema_values = [prices[0]]
    k = 2 / (period + 1)
    for i in range(1, len(prices)):
        ema_values.append(prices[i] * k + ema_values[i - 1] * (1 - k))
    return ema_values


def calculate_macd(stock_symbol, model, reverse, period, daterange):
    if reverse:
        reversed_stock_data = list(model.objects.filter(symbol=stock_symbol).order_by('date')[:daterange])[::-1]
        stock_data = reversed_stock_data[:27]
    else:
        reversed_stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('date')[:daterange + 1]
        stock_data = reversed_stock_data[0 if daterange < period else daterange - period + 1:]

    if len(stock_data) >= period:
        close_prices = [stock.close for stock in stock_data]
        macd_values = macd(close_prices, 13, 26)
        signal_line = ema(macd_values, period)

        if reverse:
            signal, _ = reverse_signals.objects.get_or_create(stock=stock_data[period - 1], symbol=stock_symbol)
        else:
            signal, _ = signals.objects.get_or_create(stock=stock_data[period - 1], symbol=stock_symbol)

        # Update MACD and signal line values in the database
        if signal.macd is None or signal.macd == 0:
            signal.macd = macd_values[period - 1]
            signal.macd_signal = signal_line[period - 1]
            signal.save()


def calculate_rsi(stock_symbol, model, reverse, period, daterange):
    if reverse:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('-date')
    else:
        reversed_stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('date')[:daterange + 1]
        stock_data = reversed_stock_data[0 if daterange < period else daterange - period + 1:]

    if len(stock_data) >= period:
        signals_obj, _ = signals.objects.get_or_create(stock=stock_data[period - 1])
        data = [data.close for data in stock_data[::-1]]
        close_prices = {'close': data}
        delta = np.diff(close_prices['close'])
        gain = np.where(delta >= 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])

        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi = 100 - (100 / (1 + rs))
        if signals_obj.rsi14 is None:
            signals_obj.rsi14 = rsi
            signals_obj.save()


def calculate_sd(stock_symbol, model, reverse, period, daterange):
    # Retrieve stock data from the database and convert it to a pandas DataFrame
    if reverse:
        reversed_stock_data = list(model.objects.filter(symbol=stock_symbol).order_by('date')[:daterange + 1])[::-1]
        stock_data = reversed_stock_data[:period]
    else:
        reversed_stock_data = list(model.objects.filter(symbol=stock_symbol).order_by('date')[:daterange + 1])
        stock_data = reversed_stock_data[0 if daterange < period else daterange - period + 1:]
    if len(stock_data) >= period:
        df = pd.DataFrame({
            'date': [entry.date for entry in stock_data],
            'open': [entry.open for entry in stock_data],
            'high': [entry.high for entry in stock_data],
            'low': [entry.low for entry in stock_data],
            'close': [entry.close for entry in stock_data],
            'volume': [entry.volume for entry in stock_data],
            'bid': [entry.bid for entry in stock_data],
            'ask': [entry.ask for entry in stock_data],
            'turnover': [entry.turnover for entry in stock_data],
            'trades': [entry.trades for entry in stock_data],
            'average': [entry.average for entry in stock_data]
        })
        # Calculate the rolling standard deviation of the 'close' column
        df['std_dev'] = df['close'].rolling(period).std()
        last_row = df.iloc[-1]
        if reverse:
            last_stock = stock_data[0]
        else:
            last_stock = stock_data[period - 1]

        std_dev = last_row['std_dev']
        if reverse:
            reverse_signal, created = reverse_signals.objects.get_or_create(stock=last_stock,
                                                                            symbol=stock_symbol)
            if created or reverse_signal.std_dev is None or reverse_signal.std_dev == 0:
                reverse_signal.std_dev = std_dev
                reverse_signal.save()
        else:
            signals_obj, created = signals.objects.get_or_create(stock=last_stock,
                                                                 symbol=stock_symbol)
            if signals_obj.std_dev is None or signals_obj.std_dev == 0:
                signals_obj.std_dev = std_dev
                signals_obj.save()


def find_optimum_buy_sell_points(stock_symbol, daterange, alldata):
    django.setup()
    connection.close()
    if alldata:
        stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('date')
        for i in range(len(stock_data)):
            if i >= daterange:
                stock_daily = stock_data[i]
                reverse_indicator = reverse_signals.objects.get(stock=stock_daily)
                indicator = signals.objects.get(stock=stock_daily)
                if optimal_buy_sell_points.objects.filter(stock=stock_daily).exists():
                    print("gaga")
                else:
                    close_val = stock_daily.close
                    prev_stock = finnish_stock_daily.objects.filter(symbol=stock_symbol,
                                                                    date__lt=stock_daily.date).order_by(
                        '-date').first()
                    prev_indicator = signals.objects.filter(symbol=stock_symbol,
                                                            stock__date__lt=stock_daily.date).order_by(
                        '-stock__date').first()
                    prev_close_val = prev_stock.close
                    if indicator.adx > 20 and indicator.adx > prev_indicator.adx \
                            and close_val > indicator.ema20 and close_val > indicator.ema50 \
                            and close_val > indicator.ema100 and close_val > indicator.ema200:
                        optimal_buy_sell_points.objects.create(stock=stock_daily, symbol=stock_symbol,
                                                               command="BUY", value=close_val)
                    elif indicator.adx < 20 and indicator.adx < prev_indicator.adx \
                            and close_val < indicator.ema20 and close_val < indicator.ema50 \
                            and close_val < indicator.ema100 and close_val < indicator.ema200:
                        optimal_buy_sell_points.objects.create(stock=stock_daily, symbol=stock_symbol,
                                                               command="SELL", value=close_val)
    else:
        if daterange > 203:
            reverse_indicators = reverse_signals.objects.filter(symbol=stock_symbol).order_by('-stock__date')[:5][::-1]
            reverse_indicator = reverse_indicators[0]
            stock_daily = reverse_indicator.stock
            indicator = signals.objects.get(stock=stock_daily)
            if optimal_buy_sell_points.objects.filter(stock=stock_daily).exists():
                print("gaga")
            else:
                close_val = stock_daily.close
                prev_stock = finnish_stock_daily.objects.filter(symbol=stock_symbol,
                                                                date__lt=stock_daily.date).order_by(
                    '-date').first()
                prev_close_val = prev_stock.close
                if (indicator.aroon_up > indicator.aroon_down * 1.1
                        # and indicator.macd * 5 < indicator.macd_signal
                        and close_val > prev_close_val - indicator.std_dev
                        and indicator.ema50 > indicator.ema200):
                    optimal_buy_sell_points.objects.create(stock=stock_daily, symbol=stock_symbol,
                                                           command="BUY", value=close_val)
                elif (indicator.aroon_up < indicator.aroon_down * 1.4
                      and indicator.macd > indicator.macd_signal * 3
                      and close_val < prev_close_val + indicator.std_dev
                      and indicator.ema50 < indicator.ema200):
                    optimal_buy_sell_points.objects.create(stock=stock_daily, symbol=stock_symbol,
                                                           command="SELL", value=close_val)


# stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('-date')[:daterange][::-1]
# reversed_stock_data = list(finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('date')[:daterange])
# stock_data = reversed_stock_data[-14:]
# for i in range(period, len(stock_data) - 5):
#     stock_daily = stock_data[i]
#     if optimal_buy_sell_points.objects.filter(stock=stock_daily).exists():
#         print(f"LÖYTYY: {stock_daily.date}")
#         continue
#     else:
#         try:
#             indicators = reverse_signals.objects.get(stock=stock_daily)
#         except ObjectDoesNotExist:
#             print(f"EI LÖYDY: {stock_daily.date}")
#             continue
#         close_val = stock_daily.close
#         prev_close_val = stock_data[i - period].close
#         print(f"UUSI OPTIMAL: {stock_daily.date}")
#         if (indicators.adx > 22 and indicators.macd * 5 < indicators.macd_signal
#                 and close_val > prev_close_val - indicators.std_dev):
#             optimal_buy_sell_points.objects.create(stock=stock_daily, symbol=stock_symbol,
#                                                    command="BUY", value=close_val)
#         elif (indicators.adx < 22 and indicators.macd * 1.1 > indicators.macd_signal
#               and close_val < prev_close_val + indicators.std_dev):
#             optimal_buy_sell_points.objects.create(stock=stock_daily, symbol=stock_symbol,
#                                                    command="SELL", value=close_val)


def calculate_profit_loss(stock_symbol, initial_investment):
    buy_sell_points = optimal_buy_sell_points.objects.filter(symbol=stock_symbol).order_by('stock__date')

    investment = initial_investment
    stocks = 0
    investment_added = 0
    last_command = 'SELL'

    for point in buy_sell_points:
        if point.command == 'BUY' and last_command != 'BUY':
            queryset = finnish_stock_daily.objects.filter(symbol=stock_symbol, date__gt=point.stock.date).order_by(
                'date')
            if queryset.count() >= 5:
                five_rows_later_stock = queryset[4]
            else:
                five_rows_later_stock = None
            if investment > 0:
                stocks = (investment - 3) / five_rows_later_stock.close if five_rows_later_stock is not None \
                    else (investment - 3) / point.value
                investment = 0
            else:
                stocks = (initial_investment - 3) / five_rows_later_stock.close if five_rows_later_stock is not None \
                    else (investment - 3) / point.value
                investment_added += 1
            last_command = 'BUY'
        elif point.command == 'SELL' and last_command != 'SELL':
            queryset = finnish_stock_daily.objects.filter(symbol=stock_symbol, date__gt=point.stock.date).order_by(
                'date')
            if queryset.count() >= 5:
                five_rows_later_stock = queryset[4]
            else:
                five_rows_later_stock = None
            investment = (stocks * five_rows_later_stock.close) - 3 if five_rows_later_stock is not None \
                else (stocks * point.value) - 3
            stocks = 0
            last_command = 'SELL'

    # Calculate profit/loss based on remaining investment and current stock price
    current_stock_price = finnish_stock_daily.objects.filter(symbol=stock_symbol).latest('date').close
    total_value = investment if investment != 0 else stocks * current_stock_price
    profit_loss = total_value
    winning = True if profit_loss - initial_investment > 0 else False
    return profit_loss, investment_added, winning
