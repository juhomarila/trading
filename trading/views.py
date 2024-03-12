import base64
import datetime
import os
import csv
import matplotlib.pyplot as plt
import io

from django.db.models import Min, Max, OuterRef, Subquery
from django.http import HttpResponse
from django.shortcuts import render, redirect
from yahoofinancials import YahooFinancials

# from .machinelearning import train_machine_learning_model_future_values, train_machine_learning_model
from .models import signals, finnish_stock_daily, optimal_buy_sell_points
from .indicators import calculate_adx, calculate_rsi, calculate_aroon, calculate_macd, \
    calculate_BBP8, calculate_sd, find_optimum_buy_sell_points, calculate_profit_loss
from .visualization import visualize_stock_and_investment, create_strategy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


def process_csv_data(request):
    print(os.getcwd())
    folder_path = '/home/juhomarila/Downloads/tradingData/uudet2'
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
        daterange = 3650
        calculate_BBP8(symbol_list[i], finnish_stock_daily, True, 3, daterange)
        calculate_sd(symbol_list[i], finnish_stock_daily, True, 3, daterange)
        calculate_macd(symbol_list[i], finnish_stock_daily, True, 3, daterange)
        calculate_adx(symbol_list[i], finnish_stock_daily, True, 3, daterange)
    return HttpResponse(status=201)


def process_sp500_data(request):
    print(os.getcwd())
    folder_path = '/home/juhomarila/Downloads/tradingData/SP500'
    symbol_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            symbol = filename.split("-")[0]
            symbol_list.append(symbol)
            with open(os.path.join(folder_path, filename), newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                counter = 0
                for row in reader:
                    if counter < 1:
                        counter += 1
                        continue
                    if (len(row[0]) == 0 or len(row[1]) == 0 or len(row[2]) == 0 or len(row[3]) == 0
                            or len(row[4]) == 0):
                        continue
                    else:
                        date = datetime.datetime.strptime(row[0], "%m/%d/%Y").date()
                        open_price = float(row[1].replace(",", ""))
                        high_price = float(row[2].replace(",", ""))
                        low_price = float(row[3].replace(",", ""))
                        closing_price = float(row[4].replace(",", ""))

                        print(symbol, date, open_price, high_price, low_price, closing_price)
                        if not finnish_stock_daily.objects.filter(symbol=symbol, date=date).exists():
                            finnish_stock_daily.objects.create(symbol=symbol, date=date, bid=0, ask=0,
                                                               open=open_price, high=high_price,
                                                               low=low_price, close=closing_price,
                                                               average=0, volume=0,
                                                               turnover=0, trades=0)
    return HttpResponse(status=201)


def find_buy_sell_points(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct()
    for symbol_data in symbol_list:
        symbol = symbol_data['symbol']
        find_optimum_buy_sell_points(symbol, True)
        print(symbol)
    return HttpResponse(status=201)


def process_daily_data(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct()
    for i in range(len(symbol_list)):
        print(symbol_list[i]['symbol'])
        daterange = 10
        calculate_BBP8(symbol_list[i]['symbol'], finnish_stock_daily, True, 3, daterange)
        calculate_sd(symbol_list[i]['symbol'], finnish_stock_daily, True, 3, daterange)
        calculate_macd(symbol_list[i]['symbol'], finnish_stock_daily, True, 3, daterange)
        calculate_adx(symbol_list[i]['symbol'], finnish_stock_daily, True, 3, daterange)
    return HttpResponse(status=201)


def get_daily_data(request):
    names = finnish_stock_daily.objects.values('symbol').distinct()
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5)  # Fetch data for the last 5 days
    for i in range(len(names)):
        print(names[i])
        print(end_date)
        print(start_date)
        if names[i]['symbol'] == 'NDA':
            data = YahooFinancials(names[i]['symbol'] + '-FI.HE').get_historical_price_data(
                start_date=str(start_date),
                end_date=str(end_date),
                time_interval='daily'
            )
            stock_data = data[names[i]['symbol'] + '-FI.HE']['prices']
            for day_data in stock_data:
                if not finnish_stock_daily.objects.filter(symbol=names[i]['symbol'],
                                                          date=day_data['formatted_date']).exists():
                    finnish_stock_daily.objects.create(symbol=names[i]['symbol'], date=day_data['formatted_date'],
                                                       open=day_data['open'], high=day_data['high'],
                                                       low=day_data['low'], close=day_data['close'],
                                                       volume=day_data['volume'])
        elif names[i]['symbol'] == 'S&P500':
            symbol = '^GSPC'
            data = YahooFinancials(symbol).get_historical_price_data(
                start_date=str(start_date),
                end_date=str(end_date),
                time_interval='daily'
            )
            stock_data = data[symbol]['prices']
            for day_data in stock_data:
                if not finnish_stock_daily.objects.filter(symbol=names[i]['symbol'],
                                                          date=day_data['formatted_date']).exists():
                    finnish_stock_daily.objects.create(symbol=names[i]['symbol'], date=day_data['formatted_date'],
                                                       open=day_data['open'], high=day_data['high'],
                                                       low=day_data['low'], close=day_data['close'],
                                                       volume=0)
        else:
            data = YahooFinancials(names[i]['symbol'] + '.HE').get_historical_price_data(
                start_date=str(start_date),
                end_date=str(end_date),
                time_interval='daily'
            )
            stock_data = data[names[i]['symbol'] + '.HE']['prices']
            for day_data in stock_data:
                if not finnish_stock_daily.objects.filter(symbol=names[i]['symbol'],
                                                          date=day_data['formatted_date']).exists():
                    finnish_stock_daily.objects.create(symbol=names[i]['symbol'], date=day_data['formatted_date'],
                                                       open=day_data['open'], high=day_data['high'],
                                                       low=day_data['low'], close=day_data['close'],
                                                       volume=day_data['volume'])
    return HttpResponse(status=201)


def visualize(request):
    ticker = request.GET.get('t')
    investment = int(request.GET.get('i'))
    expenses = int(request.GET.get('e'))
    startdate = request.GET.get('sd')
    enddate = request.GET.get('ed')
    if ticker and investment:
        stock_symbol = ticker
        buy_sell_points = optimal_buy_sell_points.objects.filter(symbol=stock_symbol).order_by('stock__date')
        buy_sell_dates = visualize_stock_and_investment(stock_symbol, buy_sell_points, investment, expenses, startdate,
                                                        enddate)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    context = {
        'img': image_base64,
        'buy_sell_dates': buy_sell_dates
    }

    return render(request, 'visualization.html', context)


def index(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct().order_by('symbol').exclude(symbol='S&P500')
    juujuu()
    return render(request, 'index.html', {'symbol_list': symbol_list})


def settings(request, symbol):
    start_date = finnish_stock_daily.objects.filter(symbol=symbol).aggregate(start_date=Min('date'))

    max_date = finnish_stock_daily.objects.filter(symbol=symbol).aggregate(max_date=Max('date'))
    context = {
        'stock_symbol': symbol,
        'start_date': start_date['start_date'],
        'max_date': max_date['max_date']
    }
    return render(request, 'settings.html', context)


def signals_page(request, symbol):
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    half_year_ago = datetime.datetime.now() - datetime.timedelta(days=183)
    if symbol == 'ALL':
        buy_sell_signals = optimal_buy_sell_points.objects.filter(stock__date__gte=thirty_days_ago).order_by(
            '-stock__date')
    else:
        # buy_sell_signals = optimal_buy_sell_points.objects.filter(symbol=symbol, stock__date__gte=half_year_ago).order_by('-stock__date')
        buy_sell_signals = optimal_buy_sell_points.objects.filter(symbol=symbol).order_by(
            '-stock__date')

    five_rows_ahead_date = finnish_stock_daily.objects.filter(
        symbol=OuterRef('symbol'),
        date__gt=OuterRef('stock__date')
    ).order_by('date').values('date')[4:5]

    buy_sell_signals = buy_sell_signals.annotate(
        date_five_rows_ahead=Subquery(five_rows_ahead_date)
    )

    return render(request, 'signals_page.html', {'signals': buy_sell_signals})


def strategy(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct().order_by('symbol').exclude(symbol='S&P500')
    start_date = finnish_stock_daily.objects.aggregate(start_date=Min('date'))
    max_date = finnish_stock_daily.objects.aggregate(max_date=Max('date'))
    context = {
        'symbol_list': symbol_list,
        'start_date': start_date['start_date'],
        'max_date': max_date['max_date']
    }
    return render(request, 'strategy.html', context)


def created_strategy(request):
    if request.method == 'POST':
        investment = request.POST.get('investment')
        start_date = request.POST.get('startdate')
        end_date = request.POST.get('enddate')
        chosen_stocks = request.POST.getlist('symbols')
        chosen_provider = request.POST.get('provider')

        buffer = io.BytesIO()
        transactions, initial_investment_total, final_investment_total = create_strategy(investment, start_date,
                                                                                         end_date, chosen_stocks,
                                                                                         chosen_provider)
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        context = {
            'investment': investment,
            'start_date': start_date,
            'end_date': end_date,
            'chosen_stocks': chosen_stocks,
            'chosen_provider': chosen_provider,
            'img': image_base64,
            'transactions': transactions,
            'initial_investment_total': initial_investment_total,
            'final_investment_total': final_investment_total,
        }
        return render(request, 'createdstrategy.html', context)
    else:
        return redirect('index')
