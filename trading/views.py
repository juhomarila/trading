import requests
import datetime
import time
import os
import csv
import matplotlib.pyplot as plt
import io
import urllib
from django.http import HttpResponse
# from alpha_vantage.timeseries import TimeSeries
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from yahoofinancials import YahooFinancials

from .machinelearning import train_machine_learning_model_future_values, train_machine_learning_model
from .models import signals, finnish_stock_daily, optimal_buy_sell_points
from .serializers import finnish_stock_daily_serializer, signals_serializer
from .indicators import calculate_adl, calculate_adx, calculate_obv, calculate_rsi, \
    calculate_aroon, calculate_macd, calculate_BBP8, calculate_sd, find_optimum_buy_sell_points, calculate_ema, \
    calculate_profit_loss
from .visualization import visualize_stock_and_investment


def process_csv_data(request):
    print(os.getcwd())
    folder_path = '/home/juhomarila/Downloads/tradingData/20192024'
    symbol_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            symbol = filename.split("-")[0]
            symbol_list.append(symbol)
            with open(os.path.join(folder_path, filename), newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                counter = 0
                for row in reader:
                    if counter < 2:
                        counter += 1
                        continue
                    if (len(row[0]) == 0 or len(row[1]) == 0 or len(row[2]) == 0 or len(row[3]) == 0
                            or len(row[4]) == 0 or len(row[5]) == 0 or len(row[6]) == 0 or len(row[7]) == 0
                            or len(row[8]) == 0 or len(row[9]) == 0 or len(row[10]) == 0):
                        continue
                    else:
                        date = datetime.datetime.strptime(row[0], '%Y-%m-%d').date()
                        bid = float(row[1].replace(",", "."))
                        ask = float(row[2].replace(",", "."))
                        open_price = float(row[3].replace(",", "."))
                        high_price = float(row[4].replace(",", "."))
                        low_price = float(row[5].replace(",", "."))
                        closing_price = float(row[6].replace(",", "."))
                        average_price = float(row[7].replace(",", "."))
                        total_volume = int(float(row[8].replace(",", ".")))
                        turnover = float(row[9].replace(",", "."))
                        trades = int(float(row[10].replace(",", ".")))

                        print(symbol, date, bid, ask, open_price, high_price, low_price,
                              closing_price, average_price, total_volume, turnover, trades)
                        if not finnish_stock_daily.objects.filter(symbol=symbol, date=date).exists():
                            finnish_stock_daily.objects.create(symbol=symbol, date=date, bid=bid, ask=ask,
                                                               open=open_price, high=high_price,
                                                               low=low_price, close=closing_price,
                                                               average=average_price, volume=total_volume,
                                                               turnover=turnover, trades=trades)
    # train_machine_learning_model()
    # train_machine_learning_model_future_values()
    for i in range(len(symbol_list)):
        calculate_BBP8(symbol_list[i], finnish_stock_daily, True, 3)
        calculate_sd(symbol_list[i], finnish_stock_daily, True, 3)
        # calculate_rsi(symbol_list[i], finnish_stock_daily, True, 3)
        # calculate_rsi(symbol_list[i], finnish_stock_daily, True, 5)
        calculate_macd(symbol_list[i], finnish_stock_daily, True, 3)
        # calculate_obv(symbol_list[i], finnish_stock_daily, True)
        # calculate_adl(symbol_list[i], finnish_stock_daily, True)
        calculate_adx(symbol_list[i], finnish_stock_daily, True, 3)
        # calculate_ema(symbol_list[i], finnish_stock_daily, True, 3)
    # Oikeinpäin
    #     calculate_BBP8(symbol_list[i], finnish_stock_daily, False, 8)
    #     calculate_aroon(symbol_list[i], finnish_stock_daily, 14)
    # calculate_ema(symbol_list[i], finnish_stock_daily, False, 20)
    # calculate_ema(symbol_list[i], finnish_stock_daily, False, 50)
    # calculate_ema(symbol_list[i], finnish_stock_daily, False, 100)
    # calculate_ema(symbol_list[i], finnish_stock_daily, False, 200)
    # calculate_macd(symbol_list[i], finnish_stock_daily, False, 9)
    # calculate_rsi(symbol_list[i], finnish_stock_daily, False, 7)
    # calculate_rsi(symbol_list[i], finnish_stock_daily, False, 14)
    # calculate_rsi(symbol_list[i], finnish_stock_daily, False, 50)
    # calculate_obv(symbol_list[i], finnish_stock_daily,False)
    # calculate_adl(symbol_list[i], finnish_stock_daily, False)
    # calculate_adx(symbol_list[i], finnish_stock_daily, False, 14)
    # calculate_sd(symbol_list[i], finnish_stock_daily, False, 14)
    return HttpResponse(status=201)


def find_buy_sell_points(request):
    total_initial_investment_list = []
    total_profit_list = []
    total_investments_added_list = []
    winning_count_list = []
    losing_count_list = []
    total_procentual_added_list = []
    total_procentual_revenue_list = []

    for initial_investment in range(100, 101, 10):
        symbol_list = finnish_stock_daily.objects.values('symbol').distinct()
        total_profit = 0
        total_investments_added = 0
        winning_count = 0
        losing_count = 0
        tickers = 0
        for symbol_data in symbol_list:
            symbol = symbol_data['symbol']
            tickers += 1
            find_optimum_buy_sell_points(symbol, True)
            profit_loss, investment_added, winning = calculate_profit_loss(symbol, initial_investment)
            total_profit += profit_loss
            total_investments_added += investment_added
            if winning:
                winning_count += 1
            else:
                losing_count += 1
        total_initial_investment_list.append(initial_investment)
        total_profit_list.append(total_profit)
        total_investments_added_list.append(total_investments_added * initial_investment)
        winning_count_list.append(winning_count)
        losing_count_list.append(losing_count)
        total_procentual_added_list.append(
            (total_investments_added * initial_investment) / (tickers * initial_investment))
        total_procentual_revenue_list.append(
            total_profit / (tickers * initial_investment + total_investments_added * initial_investment))
        print("KOKONAISTULOS 10 PÄIVÄÄ MYÖHEMMMIN: " + str(total_profit) + " INVESTOINTEJA LISÄTTY: " + str(
            int(total_investments_added * initial_investment)))
        print("VOITTAVIA OSAKKEITA: " + str(winning_count) + " HÄVIÄVIÄ OSAKKEITA " + str(losing_count))
        print("TUOTTO: + " + str(
            total_profit / (tickers * initial_investment + total_investments_added * initial_investment)))
    for i in range(len(total_initial_investment_list)):
        print("SIJOITETTU SUMMA: " + str(total_initial_investment_list[i]))
        print("TULOS: " + str(total_profit_list[i]))
        print("LISÄTTY INVESTOINTEJA: " + str(total_investments_added_list[i]))
        print("VOITTAVIA OSAKKEITA: " + str(winning_count_list[i]))
        print("HÄVIÄVIÄ OSAKKEITA: " + str(losing_count_list[i]))
        print("PROSENTUAALINEN RAHAN LISÄYS: " + str(total_procentual_added_list[i] * 100) + "%")
        print("PROSENTUAALINEN TUOTTO: " + str(total_procentual_revenue_list[i] * 100) + "%")
    return HttpResponse(status=201)


def process_daily_data(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct()
    for i in range(len(symbol_list)):
        calculate_adx(symbol_list[i]['symbol'], finnish_stock_daily, True)
        calculate_aroon(symbol_list[i]['symbol'], finnish_stock_daily, True)
        calculate_macd(symbol_list[i]['symbol'], finnish_stock_daily, True)
        calculate_rsi(symbol_list[i]['symbol'], finnish_stock_daily, True)
    return HttpResponse(status=201)


def get_daily_data(request):
    names = finnish_stock_daily.objects.values('symbol').distinct()
    today = datetime.date.today()
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    for i in range(len(names)):
        if names[i]['symbol'] == 'NDA':
            data = YahooFinancials(names[i]['symbol'] + '-FI.HE').get_historical_price_data(start_date=str(today),
                                                                                            end_date=str(tomorrow),
                                                                                            time_interval='daily')
            stock_data = data[names[i]['symbol'] + '-FI.HE']['prices'][0]
            if not finnish_stock_daily.objects.filter(symbol=names[i]['symbol'],
                                                      date=stock_data['formatted_date']).exists():
                finnish_stock_daily.objects.create(symbol=names[i]['symbol'], date=stock_data['formatted_date'],
                                                   open=stock_data['open'], high=stock_data['high'],
                                                   low=stock_data['low'], close=stock_data['close'],
                                                   volume=stock_data['volume'])
        else:
            data = YahooFinancials(names[i]['symbol'] + '.HE').get_historical_price_data(start_date=str(today),
                                                                                         end_date=str(tomorrow),
                                                                                         time_interval='daily')
            stock_data = data[names[i]['symbol'] + '.HE']['prices'][0]
            if not finnish_stock_daily.objects.filter(symbol=names[i]['symbol'],
                                                      date=stock_data['formatted_date']).exists():
                finnish_stock_daily.objects.create(symbol=names[i]['symbol'], date=stock_data['formatted_date'],
                                                   open=stock_data['open'], high=stock_data['high'],
                                                   low=stock_data['low'], close=stock_data['close'],
                                                   volume=stock_data['volume'])
    return HttpResponse(status=201)


def index(request):
    q = request.GET.get('q', None)
    if q:
        return HttpResponse('Haista vittu')
    else:
        return HttpResponse("Hello, world. You're at the polls index.")


def visualize(request):
    ticker = request.GET.get('t')
    investment = int(request.GET.get('i'))
    if ticker and investment:
        stock_symbol = ticker
        buy_sell_points = optimal_buy_sell_points.objects.filter(symbol=stock_symbol).order_by('stock__date')
        visualize_stock_and_investment(stock_symbol, buy_sell_points, investment)
    else:
        # Assuming you have already retrieved the buy_sell_points for a specific stock symbol
        stock_symbol = 'HARVIA'
        buy_sell_points = optimal_buy_sell_points.objects.filter(symbol=stock_symbol).order_by('stock__date')
        visualize_stock_and_investment(stock_symbol, buy_sell_points, 100)
    # Convert the plot to a PNG image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Embed the image in the HTTP response
    response = HttpResponse(buffer, content_type='image/png')
    return response


def index(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct()
    return render(request, 'index.html', {'symbol_list': symbol_list})


def settings(request, symbol):
    return render(request, 'settings.html', {'stock_symbol': symbol})