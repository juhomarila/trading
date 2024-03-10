import base64
import datetime
import os
import csv
import matplotlib.pyplot as plt
import io

from django.db.models import Min, Max, OuterRef, Subquery
from django.http import HttpResponse
from django.shortcuts import render
from yahoofinancials import YahooFinancials

# from .machinelearning import train_machine_learning_model_future_values, train_machine_learning_model
from .models import signals, finnish_stock_daily, optimal_buy_sell_points
from .indicators import calculate_adx, calculate_rsi, calculate_aroon, calculate_macd, \
    calculate_BBP8, calculate_sd, find_optimum_buy_sell_points, calculate_profit_loss
from .visualization import visualize_stock_and_investment


def process_csv_data(request):
    print(os.getcwd())
    folder_path = '/home/juhomarila/Downloads/tradingData/uudet'
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
        daterange = 14
        calculate_BBP8(symbol_list[i], finnish_stock_daily, True, 3, daterange)
        calculate_sd(symbol_list[i], finnish_stock_daily, True, 3, daterange)
        # calculate_rsi(symbol_list[i], finnish_stock_daily, True, 3)
        # calculate_rsi(symbol_list[i], finnish_stock_daily, True, 5)
        calculate_macd(symbol_list[i], finnish_stock_daily, True, 3, daterange)
        # calculate_obv(symbol_list[i], finnish_stock_daily, True)
        # calculate_adl(symbol_list[i], finnish_stock_daily, True)
        calculate_adx(symbol_list[i], finnish_stock_daily, True, 3, daterange)
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

def visualize(request):
    ticker = request.GET.get('t')
    investment = int(request.GET.get('i'))
    expenses = int(request.GET.get('e'))
    startdate = request.GET.get('sd')
    enddate = request.GET.get('ed')
    if ticker and investment:
        stock_symbol = ticker
        buy_sell_points = optimal_buy_sell_points.objects.filter(symbol=stock_symbol).order_by('stock__date')
        visualize_stock_and_investment(stock_symbol, buy_sell_points, investment, expenses, startdate, enddate)
    else:
        # Assuming you have already retrieved the buy_sell_points for a specific stock symbol
        stock_symbol = 'HARVIA'
        buy_sell_points = optimal_buy_sell_points.objects.filter(symbol=stock_symbol).order_by('stock__date')
        visualize_stock_and_investment(stock_symbol, buy_sell_points, 100, 3)
    # Convert the plot to a PNG image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert the image buffer to base64 format
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()  # Close the plot to free up memory

    return render(request, 'visualization.html', {'img': image_base64})


def index(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct().order_by('symbol')
    return render(request, 'index.html', {'symbol_list': symbol_list})


def settings(request, symbol):
    # Query the start date for the given symbol
    start_date = finnish_stock_daily.objects.filter(symbol=symbol).aggregate(start_date=Min('date'))

    # Query the max date for the given symbol
    max_date = finnish_stock_daily.objects.filter(symbol=symbol).aggregate(max_date=Max('date'))
    context = {
        'stock_symbol': symbol,
        'start_date': start_date['start_date'],  # Accessing the start_date value from the dictionary
        'max_date': max_date['max_date']  # Accessing the max_date value from the dictionary
    }
    return render(request, 'settings.html', context)


def signals_page(request, symbol):
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    half_year_ago = datetime.datetime.now() - datetime.timedelta(days=183)
    if symbol == 'ALL':
        buy_sell_signals = optimal_buy_sell_points.objects.filter(stock__date__gte=thirty_days_ago).order_by(
            '-stock__date')
    else:
        buy_sell_signals = optimal_buy_sell_points.objects.filter(symbol=symbol, stock__date__gte=half_year_ago).order_by('-stock__date')

    # Subquery to fetch the date five rows ahead for each signal
    five_rows_ahead_date = finnish_stock_daily.objects.filter(
        symbol=OuterRef('symbol'),
        date__gt=OuterRef('stock__date')
    ).order_by('date').values('date')[4:5]

    # Annotate the buy_sell_signals queryset with the five_rows_ahead_date subquery
    buy_sell_signals = buy_sell_signals.annotate(
        date_five_rows_ahead=Subquery(five_rows_ahead_date)
    )

    return render(request, 'signals_page.html', {'signals': buy_sell_signals})
