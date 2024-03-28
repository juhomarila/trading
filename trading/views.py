import base64
import datetime
import multiprocessing
from multiprocessing import Manager, Semaphore
import os
import timeit
import csv
import matplotlib.pyplot as plt
import io

from django.db.models import Min, Max, OuterRef, Subquery
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
import django
from django.db import connection
from yahoofinancials import YahooFinancials
from io import StringIO

# from .machinelearning import train_machine_learning_model_future_values, train_machine_learning_model
from .models import signals, finnish_stock_daily, optimal_buy_sell_points
from .indicators import calculate_adx, calculate_rsi, calculate_aroon, calculate_macd, \
    calculate_BBP8, calculate_sd, find_optimum_buy_sell_points, calculate_profit_loss, calculate_ema
from .simulation import optimize_parameters
from .visualization import visualize_stock_and_investment, create_strategy

MAX_PROCESSES = 8


def process_uploaded_csv(request):
    if request.method == 'POST' and request.FILES.getlist('csv_files'):
        # Get list of uploaded CSV files
        csv_files = request.FILES.getlist('csv_files')

        # Process each uploaded CSV file
        manager = Manager()
        semaphore = Semaphore(MAX_PROCESSES)
        processes = []
        for csv_file in csv_files:
            if csv_file.name.endswith(".csv"):
                symbol = csv_file.name.split("-")[0]
                # Read the CSV file content
                csv_content = csv_file.read().decode('utf-8')
                # Create a file-like object from the CSV content
                csv_content_file = StringIO(csv_content)
                # Process CSV data and save to database
                if symbol == 'S&P500':
                    delimiter = ','
                else:
                    delimiter = ';'
                p = multiprocessing.Process(target=csv_worker,
                                            args=(semaphore, symbol, csv_content_file, delimiter))
                processes.append(p)
                p.start()

        for process in processes:
            process.join()

        return redirect('process_data')

    return render(request, 'upload_csv.html')


def csv_worker(semaphore, symbol, csv_content_file, delimiter):
    semaphore.acquire()
    try:
        process_csv_data(symbol, csv_content_file, delimiter)
    finally:
        semaphore.release()


def process_csv_data(symbol, file, delimiter):
    counter = 0
    reader = csv.reader(file, delimiter=delimiter)
    for row in reader:
        if delimiter == ',':
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
        else:
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


def process_data(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').order_by('symbol').distinct()
    if request.method == 'POST':
        func_name = request.POST.get('func_name')
        if func_name == 'bulk':
            process_bulk_data()
            return JsonResponse({'message': 'Massadatan prosessointi valmis'})
        elif func_name == 'buy_sell':
            find_buy_sell_points()
            return JsonResponse({'message': 'Osto/myynti pisteiden prosessointi valmis'})
        elif func_name == 'daily':
            process_daily_data()
            return JsonResponse({'message': 'Päivittäisen uuden datan prosessointi valmis'})
        elif func_name == 'get_daily':
            get_daily_data()
            return JsonResponse({'message': 'Päivittäinen data haettu ja kirjoitettu kantaan'})
        elif func_name == 'get_daily_buy_sell':
            find_buy_sell_points_for_daily_data()
            return JsonResponse({'message': 'Päivittäinen osto-/myyntipisteet kirjoitettu kantaan'})
        elif func_name == 'simulate':
            simulation()
            return JsonResponse({'message': 'Simuloitu'})
        else:
            return JsonResponse({'message': 'Invalid request'}, status=400)
    elif request.method == 'GET':
        return render(request, 'process_data.html', {'symbols': symbol_list})


def simulation():
    list_half_step = [i / 2 for i in range(0, 201)]
    list_one_step = list(range(101))
    buy_param_ranges = {
        'adx_threshold': [55.5],
        'adx_high_threshold': [63.5],
        'aroon_up_thresholds': [69.5],
        'aroon_down_thresholds': [73.5],
        'rsi_threshold': [56],
    }
    sell_param_ranges = {
        'adx_threshold': [33.5],
        'adx_low_threshold': [22.5],
        'aroon_up_thresholds': [50.5],
        'aroon_down_thresholds': [89],
        'rsi_threshold': [64.5],
    }
    # buy_param_ranges = {
    #     'adx_threshold': [55.5],
    #     'adx_high_threshold': [63.5],
    #     'aroon_up_thresholds': [69.5],
    #     'aroon_down_thresholds': [73.5],
    #     'rsi_threshold': [56],
    # }
    # sell_param_ranges = {
    #     'adx_threshold': [33.5],
    #     'adx_low_threshold': [22.5],
    #     'aroon_up_thresholds': [50.5],
    #     'aroon_down_thresholds': [89],
    #     'rsi_threshold': [64.5],
    # }

    start = timeit.default_timer()
    best_buy_params, best_sell_params, best_profit = optimize_parameters(buy_param_ranges, sell_param_ranges)
    end = timeit.default_timer()
    print(f"Execution Time: {end - start}")
    print("Best Buy Parameters:", best_buy_params)
    print("Best Sell Parameters:", best_sell_params)
    print("Best Overall Profit:", best_profit)


def process_bulk_data():
    daterange = 14  # For history as long as it goes
    # calculate_BBP8(symbol, finnish_stock_daily, True, 3, daterange)
    manager = Manager()
    semaphore = Semaphore(100)
    processes = []
    symbols = finnish_stock_daily.objects.values('symbol').distinct()
    data = finnish_stock_daily.objects.all().order_by('date')
    for stock_symbol in symbols:
        if stock_symbol['symbol'] != 'S&P500':
            # multiprocess_data(stock_symbol['symbol'], data)
            p = multiprocessing.Process(target=bulk_worker, args=(semaphore, stock_symbol['symbol'], data,))
            processes.append(p)
            p.start()

    for process in processes:
        process.join()


def bulk_worker(semaphore, symbol, data):
    semaphore.acquire()
    try:
        multiprocess_data(symbol, data)
    finally:
        semaphore.release()


def multiprocess_data(symbol, data):
    django.setup()
    connection.close()
    stocks = data.filter(symbol=symbol)
    for i in range(len(stocks)):
        # if symbol == 'OUT1V':
        print(f"ITERAATIO: {i + 1}, PÄIVÄ: {stocks[i].date} OSAKE: {symbol}")
        calculate_aroon(symbol, i, stocks)
        # calculate_macd(symbol, finnish_stock_daily, False, 26, i, stocks)
        # calculate_sd(symbol, finnish_stock_daily, False, 14, i, stocks)
        # calculate_ema(symbol, finnish_stock_daily, False, 20, i, stocks)
        # calculate_ema(symbol, finnish_stock_daily, False, 50, i, stocks)
        # calculate_ema(symbol, finnish_stock_daily, False, 100, i, stocks)
        # calculate_ema(symbol, finnish_stock_daily, False, 200, i, stocks)
        calculate_rsi(symbol, finnish_stock_daily, False, 15, i, stocks)
        calculate_adx(symbol, finnish_stock_daily, False, 14, i, True, stocks)
        # find_optimum_buy_sell_points(symbol, i, False)


def find_buy_sell_points():
    manager = Manager()
    semaphore = Semaphore(MAX_PROCESSES)
    processes = []
    symbols = signals.objects.values('symbol').distinct()
    for stock_symbol in symbols:
        print(f"SYMBOL: {stock_symbol['symbol']}")
        p = multiprocessing.Process(target=buy_sell_points_worker, args=(semaphore, stock_symbol['symbol'], 26, True))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
    # daterange = 14  # For history as long as it goes
    # # calculate_BBP8(symbol, finnish_stock_daily, True, 3, daterange)

    # for i in range(len(symbols)):
    #     find_optimum_buy_sell_points(symbols[i]['symbol'], 203, True)
    #     print(symbols[i]['symbol'])


def buy_sell_points_worker(semaphore, stock_symbol, daterange, alldata):
    semaphore.acquire()
    try:
        find_optimum_buy_sell_points(stock_symbol, daterange, alldata)
    finally:
        semaphore.release()


def find_buy_sell_points_for_daily_data():
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct()
    for i in range(len(symbol_list)):
        if symbol_list[i]['symbol'] != "S&P500":
            find_optimum_buy_sell_points(symbol_list[i]['symbol'], 50,
                                         False)  # daterange here can be anything, doesn't affect calculations
            print(symbol_list[i]['symbol'])


def find_buy_sell_points_for_daily_data_cronjob():
    find_buy_sell_points_for_daily_data()
    return HttpResponse(status=201)


def process_daily_data():
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct()

    for i in range(len(symbol_list)):
        if symbol_list[i]['symbol'] != 'S&P500':
            data = finnish_stock_daily.objects.filter(symbol=symbol_list[i]['symbol']).order_by('-date')[
                   :27]  # IMPORTANT TO HAVE 27 since adx needs exactly 27 entries. If more results distort
            stocks = data[::-1]
            print(symbol_list[i]['symbol'])
            for y in range(len(stocks)):
                calculate_aroon(symbol_list[i]['symbol'], y, stocks)
                calculate_rsi(symbol_list[i]['symbol'], finnish_stock_daily, False, 15, y, stocks)
            for z in range(len(stocks)):
                calculate_adx(symbol_list[i]['symbol'], finnish_stock_daily, False, 14, z, False, stocks)


def process_daily_data_cronjob(request):
    process_daily_data()
    return HttpResponse(status=201)


def get_daily_data():
    names = finnish_stock_daily.objects.values('symbol').distinct()
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=3)  # Fetch data for the last 3 days
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
                                                       average=0, volume=day_data['volume'],
                                                       turnover=0, trades=0, bid=0, ask=0)
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
                                                       average=0, volume=0, bid=0, ask=0,
                                                       turnover=0, trades=0)
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
                                                       average=0, volume=day_data['volume'],
                                                       turnover=0, trades=0, bid=0, ask=0)


def get_daily_data_cronjob(request):
    get_daily_data()
    return HttpResponse(status=201)


def visualize(request):
    ticker = request.GET.get('t')
    investment = int(request.GET.get('i'))
    expenses = int(request.GET.get('e'))
    startdate = request.GET.get('sd')
    enddate = request.GET.get('ed')
    if ticker and investment and expenses and startdate and enddate:
        stock_symbol = ticker
        buy_sell_points = optimal_buy_sell_points.objects.filter(symbol=stock_symbol).order_by('stock__date')
        plot1, plot2, buy_sell_dates = visualize_stock_and_investment(stock_symbol, buy_sell_points, investment,
                                                                      expenses, startdate,
                                                                      enddate)
    else:
        return HttpResponse(status=404)

    context = {
        'img': plot1,
        'img2': plot2,
        'buy_sell_dates': buy_sell_dates
    }

    return render(request, 'visualization.html', context)


def index(request):
    symbol_list = finnish_stock_daily.objects.values('symbol').distinct().order_by('symbol').exclude(symbol='S&P500')
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
    if symbol == 'ALL':
        buy_sell_signals = optimal_buy_sell_points.objects.filter(stock__date__gte=thirty_days_ago).order_by(
            '-stock__date')
    else:
        buy_sell_signals = optimal_buy_sell_points.objects.filter(symbol=symbol).order_by(
            '-stock__date')

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
        (transactions, initial_investment_total, final_investment_total, hold_investment, investment_growth,
         hold_investment_growth, results) = create_strategy(
            investment, start_date,
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
            'hold_investment': hold_investment,
            'investment_growth': investment_growth,
            'hold_investment_growth': hold_investment_growth,
            'results': results
        }
        return render(request, 'createdstrategy.html', context)
    else:
        return redirect('index')
