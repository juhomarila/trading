import math
from datetime import timedelta
from typing import List
import numpy as np
import pandas as pd
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
            # Skip calculation for the initial period
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
        print("OBV " + stock_symbol, row['obv'], row['date'])
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
        print("ADL " + stock_symbol, row['adl'], row['date'])
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
        stock_data = list(
            model.objects.filter(symbol=stock_symbol).values('date', 'close', 'open', 'high', 'low', 'volume')
            .order_by('-date')[:daterange])
    else:
        stock_data = list(
            model.objects.filter(symbol=stock_symbol).values('date', 'close', 'open', 'high', 'low', 'volume')
            .order_by('date'))
    df = pd.DataFrame(stock_data)
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
    for index, row in df.iterrows():
        stock = model.objects.get(date=row['date'], symbol=stock_symbol)
        print("ADX " + stock_symbol, row['ADX'], period, row['date'])
        if reverse:
            reverse_signals_obj = reverse_signals.objects.filter(stock=stock, symbol=stock_symbol).first()
            if reverse_signals_obj:
                reverse_signals_obj.adx = row['ADX']
                reverse_signals_obj.save()
        else:
            if not signals.objects.filter(stock__id=index, adx__isnull=False).exists():
                signals_obj = signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                if signals_obj:
                    signals_obj.adx = row['ADX']
                    signals_obj.save()


def calculate_aroon(stock_symbol, model, period=14):
    stock_data = model.objects.filter(symbol=stock_symbol).order_by('date')
    for i, stock in enumerate(stock_data):
        if i < period:
            if not signals.objects.filter(stock__id=i, aroon_up__isnull=False, aroon_down__isnull=False).exists():
                signals_obj = signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                if signals_obj:
                    signals_obj.aroon_up = 0
                    signals_obj.aroon_down = 0
                    signals_obj.save()
                continue
        high_prices = [stock.high for stock in stock_data[i - period:i]]
        low_prices = [stock.low for stock in stock_data[i - period:i]]
        hh = max(high_prices)
        ll = min(low_prices)
        aroon_up = ((period - (high_prices.index(hh))) / period) * 100
        aroon_down = ((period - (low_prices.index(ll))) / period) * 100
        print("AROON " + stock.symbol, aroon_up, aroon_down, stock.date)
        if not signals.objects.filter(stock__id=i, aroon_up__isnull=False, aroon_down__isnull=False).exists():
            signals_obj = signals.objects.filter(stock=stock, symbol=stock_symbol).first()
            if signals_obj:
                signals_obj.aroon_up = aroon_up
                signals_obj.aroon_down = aroon_down
                signals_obj.save()


def calculate_ema(stock_symbol, model, reverse, period):
    if reverse:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('-date')
    else:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('date')
    k = 2 / (period + 1)
    exp_mov_av = 0
    for i in range(len(stock_data)):
        if i == 0:
            exp_mov_av = stock_data[i].close
        else:
            exp_mov_av = (stock_data[i].close * k) + (exp_mov_av * (1 - k))
        print('EMA: ', period, exp_mov_av, stock_data[i].date)
        if period == 5:
            reverse_signals_obj = reverse_signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
            if reverse_signals_obj:
                reverse_signals_obj.ema5 = exp_mov_av
                reverse_signals_obj.save()
        if period == 20:
            signals_obj = signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
            if signals_obj:
                signals_obj.ema20 = exp_mov_av
                signals_obj.save()
            continue
        if period == 50:
            signals_obj = signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
            if signals_obj:
                signals_obj.ema50 = exp_mov_av
                signals_obj.save()
            continue
        if period == 100:
            signals_obj = signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
            if signals_obj:
                signals_obj.ema100 = exp_mov_av
                signals_obj.save()
            continue
        else:
            signals_obj = signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
            if signals_obj:
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
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('-date')[:daterange]
    else:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('date')
    close_prices = [stock.close for stock in stock_data]
    macd_values = macd(close_prices, 12, 26)
    signal_line = ema(macd_values, period)
    for i, stock in enumerate(stock_data):
        if i < period:
            if reverse:
                reverse_signals_obj = reverse_signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                if reverse_signals_obj:
                    reverse_signals_obj.macd = 0
                    reverse_signals_obj.macd_signal = 0
                    reverse_signals_obj.save()
                continue
            else:
                if not signals.objects.filter(stock__id=i, macd__isnull=False, macd_signal__isnull=False).exists():
                    signals_obj = signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                    if signals_obj:
                        signals_obj.macd = 0
                        signals_obj.macd_signal = 0
                        signals_obj.save()
                    continue
        print("MACD " + stock_symbol, macd_values[i], signal_line[i], stock.date)
        if reverse:
            if not reverse_signals.objects.filter(stock__id=i, macd__isnull=False, macd_signal__isnull=False).exists():
                reverse_signals_obj = reverse_signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                if reverse_signals_obj:
                    reverse_signals_obj.macd = macd_values[i]
                    reverse_signals_obj.macd_signal = signal_line[i]
                    reverse_signals_obj.save()
        else:
            if not signals.objects.filter(stock__id=i, macd__isnull=False, macd_signal__isnull=False).exists():
                signals_obj = signals.objects.filter(stock=stock, symbol=stock_symbol).first()
                if signals_obj:
                    signals_obj.macd = macd_values[i]
                    signals_obj.macd_signal = signal_line[i]
                    signals_obj.save()


def calculate_rsi(stock_symbol, model, reverse, period):
    if reverse:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('-date')
    else:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('date')
    close_prices = [data.close for data in stock_data]
    changes = [close_prices[i] - close_prices[i - 1] for i in range(1, len(close_prices))]
    avg_gain = 0
    avg_loss = 0
    rsi_values = []
    for i in range(len(stock_data)):
        if i < period:
            rsi_val = None
            continue
        else:
            gain = max(changes[i - 1], 0)
            loss = abs(min(changes[i - 1], 0))
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period
            if avg_loss == 0:
                rs = float("inf")
            else:
                rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi
            rsi_values.append(rsi_val)
        print("RSI ", period, stock_symbol, rsi_val, stock_data[i].date)
        if period == 5:
            reverse_signals_obj = reverse_signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
            if reverse_signals_obj:
                reverse_signals_obj.rsi5 = rsi_val
                reverse_signals_obj.save()
            continue
        if period == 7:
            if reverse:
                reverse_signals_obj = reverse_signals.objects.filter(stock=stock_data[i],
                                                                     symbol=stock_symbol).first()
                if reverse_signals_obj:
                    reverse_signals_obj.rsi7 = rsi_val
                    reverse_signals_obj.save()
                continue
            else:
                signals_obj = signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
                if signals_obj:
                    signals_obj.rsi7 = rsi_val
                    signals_obj.save()
                continue
        if period == 14:
            signals_obj = signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
            if signals_obj:
                signals_obj.rsi14 = rsi_val
                signals_obj.save()
            continue
        if period == 50:
            signals_obj = signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
            if signals_obj:
                signals_obj.rsi50 = rsi_val
                signals_obj.save()


def calculate_sd(stock_symbol, model, reverse, period, daterange):
    # Retrieve stock data from the database and convert it to a pandas DataFrame
    if reverse:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('-date')[:daterange]
    else:
        stock_data = model.objects.filter(symbol=stock_symbol).order_by('date')
    df = pd.DataFrame.from_records(stock_data.values())

    # Calculate the rolling standard deviation of the 'close' column
    df['std_dev'] = df['close'].rolling(period).std()

    # Iterate over the rows of the DataFrame and create a corresponding SD object
    for i, row in df.iterrows():
        stock = model.objects.get(date=row['date'], symbol=stock_symbol)
        # If the row does not have a valid 'std_dev' value, use a default value
        if pd.isna(row['std_dev']):
            std_dev = 0.0
        else:
            std_dev = row['std_dev']
        if reverse:
            print('SD: ', std_dev, stock.date)
            reverse_signals_obj = reverse_signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
            if reverse_signals_obj:
                reverse_signals_obj.std_dev = std_dev
                reverse_signals_obj.save()
        else:
            if not signals.objects.filter(stock__id=i, std_dev__isnull=False).exists():
                print('SD: ', std_dev, stock.date)
                signals_obj = signals.objects.filter(stock=stock_data[i], symbol=stock_symbol).first()
                if signals_obj:
                    signals_obj.std_dev = std_dev
                    signals_obj.save()


def find_optimum_buy_sell_points(stock_symbol, period=14):
    stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('date')
    investment = 100
    compare_investment = 100
    compare_stocks = 0
    first_buy = False
    stocks = 0
    sell_count = 0
    buy_count = 0
    last_command = 'SELL'
    for i in range(period, len(stock_data)):
        stock_daily = stock_data[i]
        try:
            indicators = reverse_signals.objects.get(stock=stock_daily)
        except ObjectDoesNotExist:
            continue
        close_val = stock_data[i].close
        # open_val = stock_data[i].open
        # date = stock_data[i].date
        prev_close_val = stock_data[i - period].close
        if indicators.adx > 22 \
                and indicators.macd * 5 < indicators.macd_signal and close_val > prev_close_val - indicators.std_dev:
            if not optimal_buy_sell_points.objects.filter(stock=stock_daily).exists():
                optimal_buy_sell_points.objects.create(stock=stock_daily, symbol=stock_symbol,
                                                       command="BUY", value=close_val)
            if last_command != "BUY":
                buy_count += 1
                stocks = investment / close_val
                investment = 0
                last_command = "BUY"
                if not first_buy:
                    first_buy = True
                    compare_stocks = compare_investment / close_val
        elif indicators.adx < 22 \
                and indicators.macd * 1.1 > indicators.macd_signal and close_val < prev_close_val + indicators.std_dev:
            if not optimal_buy_sell_points.objects.filter(stock=stock_daily).exists():
                optimal_buy_sell_points.objects.create(stock=stock_daily, symbol=stock_symbol,
                                                       command="SELL", value=close_val)
            if last_command != "SELL":
                sell_count += 1
                investment = stocks * close_val
                stocks = 0
                last_command = "SELL"
    if investment != 0:
        return investment - 100, (compare_stocks * stock_data[len(stock_data) - 1].close) - 100
    else:
        return (stocks * stock_data[len(stock_data) - 1].close) - 100, \
               (compare_stocks * stock_data[len(stock_data) - 1].close) - 100


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
                # print("ARVO KYMMENEN PÄIVÄÄ MYÖHEMMIN: ", str(ten_rows_later_stock.close) + " PÄIVÄ: ",
                #      ten_rows_later_stock.date)
            else:
                five_rows_later_stock = None
                # print("ARVO KYMMENEN PÄIVÄÄ MYÖHEMMIN: ", str(point.value) + " PÄIVÄ: ",
                #      point.stock.date)
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
            # print("INVESTOINTI: ", str(int(investment)) + " INVESTOITI VIIKKOA MYÖHEMMIN: ", str(int(investment_10_days_later)))
            last_command = 'SELL'

    # Calculate profit/loss based on remaining investment and current stock price
    current_stock_price = finnish_stock_daily.objects.filter(symbol=stock_symbol).latest('date').close
    total_value = investment if investment != 0 else stocks * current_stock_price
    profit_loss = total_value
    print("OSAKE: " + stock_symbol + " TULOS: " + str(profit_loss) + " INVESTOINTEJA LISÄTTY " + str(investment_added))
    # print("TULOS 10 PÄIVÄÄ MYHÖEMMIN: " + str(profit_loss_10_days_later) + " INVESTOINTEJA LISÄTTY " + str(investment_10_days_later_added))
    winning = True if profit_loss - initial_investment > 0 else False
    return profit_loss, investment_added, winning
