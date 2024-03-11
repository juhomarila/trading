from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from trading.models import finnish_stock_daily, optimal_buy_sell_points


def visualize_stock_and_investment(stock_symbol, buy_sell_points, initial_investment, expenses, search_start_date,
                                   search_end_date):
    stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol,
                                                    date__range=(search_start_date, search_end_date)).order_by('date')
    sp_500_benchmark_data = finnish_stock_daily.objects.filter(symbol='S&P500', date__range=(
        search_start_date, search_end_date)).order_by('date')
    dates = [stock.date for stock in stock_data]
    stock_values = [stock.close for stock in stock_data]
    sp_500_dates = [sp_500.date for sp_500 in sp_500_benchmark_data]
    sp_500_values = [sp_500.close for sp_500 in sp_500_benchmark_data]

    initial_stock_value = stock_values[0]
    initial_sp_500_value = sp_500_values[0]

    stocks = 0
    investment = initial_investment
    investments = []
    last_command = 'SELL'
    search_start_date = datetime.strptime(search_start_date, '%Y-%m-%d').date()
    search_end_date = datetime.strptime(search_end_date, '%Y-%m-%d').date()
    buy_sell_dates = []

    stock_data_dict = {stock.date: stock for stock in stock_data}

    for point in buy_sell_points:
        stock_keys = sorted(stock_data_dict.keys())
        index = stock_keys.index(point.stock.date)
        if (point.command == 'BUY' and last_command != 'BUY'
                and search_start_date < point.stock.date < search_end_date):
            if index + 6 < len(stock_keys):
                last_buy_date = stock_keys[index + 5]
                last_buy_stock = stock_data_dict[last_buy_date]
            else:
                last_buy_stock = None
                last_buy_date = point.stock.date
            if investment > 0:
                stocks = (investment - expenses) / last_buy_stock.close if last_buy_stock is not None \
                    else (investment - expenses) / point.value
                investments.append((last_buy_date, stocks, 'BUY'))
                investment = 0
            else:
                stocks = (initial_investment - expenses) / last_buy_stock.close if last_buy_stock is not None \
                    else (investment - expenses) / point.value
                investments.append((last_buy_date, stocks, 'BUY'))
            buy_sell_dates.append(
                (last_buy_date, last_buy_stock.close if last_buy_stock is not None else point.stock.close, 'BUY'))
            last_command = 'BUY'
        elif point.command == 'SELL' and last_command != 'SELL' \
                and search_start_date < point.stock.date < search_end_date:
            if index + 6 < len(stock_keys):
                last_sell_date = stock_keys[index + 5]
                last_sell_stock = stock_data_dict[last_sell_date]
            else:
                last_sell_stock = None
                last_sell_date = point.stock.date
            investment = (stocks * last_sell_stock.close) - expenses if last_sell_stock is not None \
                else (stocks * point.value) - expenses
            stocks = 0
            investments.append((last_sell_date, investment, 'SELL'))
            last_command = 'SELL'
            buy_sell_dates.append(
                (last_sell_date, last_sell_stock.close if last_sell_stock is not None else point.stock.close, 'SELL'))

    stock_changes = [(100 * (stock_values[i] - initial_stock_value) / initial_stock_value) for i in
                     range(1, len(stock_values))]
    sp_500_changes = [(100 * (sp_500_values[i] - initial_sp_500_value) / initial_sp_500_value) for i in
                      range(1, len(sp_500_values))]

    investment_changes = []

    for i in range(len(investments)):
        date, investment_value_or_stocks, signal = investments[i]

        if signal == 'SELL':
            for x in range(i + 1, len(investments)):
                next_date, _, next_signal = investments[x]
                if next_signal == 'BUY':
                    buy_date = next_date
                    break
            else:
                buy_date = dates[-1]
            date_data = []
            for date_key in stock_data_dict.keys():
                if date <= date_key <= buy_date:
                    date_data.append(stock_data_dict[date_key])
            date_data.sort(key=lambda o: o.date)
            if len(date_data) > 0:
                for r in range(len(date_data)):
                    investment_changes.append((date_data[r].date, 100 * (
                            investment_value_or_stocks - initial_investment) / initial_investment))
        else:
            for j in range(i + 1, len(investments)):
                next_date, _, next_signal = investments[j]
                if next_signal == 'SELL':
                    sell_date = next_date
                    break
            else:
                sell_date = dates[-1]
            stock_change_data = []
            for date_key in stock_data_dict.keys():
                if date <= date_key <= sell_date:
                    stock_change_data.append(stock_data_dict[date_key])
            stock_change_data.sort(key=lambda o: o.date)
            if len(stock_change_data) > 0:
                for y in range(len(stock_change_data)):
                    investment_change_percentage = 100 * (investment_value_or_stocks * stock_change_data[y].close
                                                          - initial_investment) / initial_investment
                    investment_changes.append((stock_change_data[y].date, investment_change_percentage))

    plt.figure(figsize=(16, 8))
    plt.plot(dates[1:], stock_changes, label=stock_symbol + ' Value Change (%)', color='blue')
    plt.plot(sp_500_dates[1:], sp_500_changes, label='SP&500 Value Change (%)', color='yellow')
    plt.plot(*zip(*investment_changes), linestyle='-', color='red',
             label='Investment Change (%), initial: ' + str(initial_investment) + "€")
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(25))
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.title('Percentage Change in ' + stock_symbol + ' Stock Value and Investment between '
              + str(search_start_date) + ' and ' + str(search_end_date))
    plt.legend()
    plt.show()

    return sorted(buy_sell_dates, key=lambda o: o[0], reverse=True)


def create_strategy(investment, start_date, end_date, chosen_stocks, chosen_provider):
    if chosen_provider == 'Osuuspankki':  # case osuuspankki
        expenses = 5
    else:  # Case nordnet
        expenses = 5
    stock_data = finnish_stock_daily.objects.filter(symbol__in=chosen_stocks,
                                                    date__range=(start_date, end_date)).order_by('date')
    sp_500_benchmark_data = finnish_stock_daily.objects.filter(symbol='S&P500', date__range=(
        start_date, end_date)).order_by('date')

    sp_500_dates = [sp_500.date for sp_500 in sp_500_benchmark_data]
    sp_500_values = [sp_500.close for sp_500 in sp_500_benchmark_data]
    initial_sp_500_value = sp_500_values[0]
    sp_500_changes = [(100 * (sp_500_values[i] - initial_sp_500_value) / initial_sp_500_value) for i in
                      range(1, len(sp_500_values))]

    sell_counter = 0
    buy_counter = 0
    initial_investment = int(investment)
    investment_by_stock = {stock_symbol: initial_investment for stock_symbol in chosen_stocks}
    investments = {}
    stocks = {}
    for i in range(len(stock_data)):
        signal = optimal_buy_sell_points.objects.filter(stock=stock_data[i]).first()
        if signal:
            previous_signal = optimal_buy_sell_points.objects.filter(
                stock__symbol=signal.stock.symbol,
                stock__date__lt=signal.stock.date,
            ).order_by('-stock__date').first()
            if signal.command == 'BUY' and (
                    (previous_signal and previous_signal.command == 'SELL') or previous_signal is None):
                buy_counter += 1
                queryset = finnish_stock_daily.objects.filter(symbol=signal.symbol,
                                                              date__gt=signal.stock.date).order_by(
                    'date')
                if queryset.count() >= 5:
                    last_buy_stock = queryset[4]
                    last_buy_date = last_buy_stock.date
                else:
                    last_buy_stock = None
                    last_buy_date = signal.stock.date
                if int(investment_by_stock[signal.symbol]) > 0:
                    stocks[signal.symbol] = (int(investment_by_stock[
                                                     signal.symbol]) - expenses) / last_buy_stock.close if last_buy_stock is not None \
                        else (int(investment_by_stock[signal.symbol]) - expenses) / signal.value
                    if signal.symbol in investments:
                        investments[signal.symbol].append((last_buy_date, stocks[signal.symbol], 'BUY'))
                    else:
                        investments[signal.symbol] = [(last_buy_date, stocks[signal.symbol], 'BUY')]
                    investment_by_stock[signal.symbol] = 0
                else:
                    stocks[signal.symbol] = (int(investment_by_stock[
                                                     signal.symbol]) - expenses) / last_buy_stock.close if last_buy_stock is not None \
                        else (int(investment_by_stock[signal.symbol]) - expenses) / signal.value
                    if signal.symbol in investments:
                        investments[signal.symbol].append((last_buy_date, stocks[signal.symbol], 'BUY'))
                    else:
                        investments[signal.symbol] = [(last_buy_date, stocks[signal.symbol], 'BUY')]

            elif signal.command == 'SELL' and previous_signal and previous_signal.command == 'BUY':
                sell_counter += 1
                queryset = finnish_stock_daily.objects.filter(symbol=signal.symbol,
                                                              date__gt=signal.stock.date).order_by(
                    'date')
                if queryset.count() >= 5:
                    last_sell_stock = queryset[4]
                    last_sell_date = last_sell_stock.date
                else:
                    last_sell_stock = None
                    last_sell_date = signal.stock.date
                investment_by_stock[signal.symbol] = (stocks[
                                                          signal.symbol] * last_sell_stock.close) - expenses if last_sell_stock is not None \
                    else (stocks[signal.symbol] * signal.value) - expenses
                stocks[signal.symbol] = 0
                if signal.symbol in investments:
                    investments[signal.symbol].append((last_sell_date, investment_by_stock[signal.symbol], 'SELL'))
                else:
                    investments[signal.symbol] = [(last_sell_date, investment_by_stock[signal.symbol], 'SELL')]

            else:
                continue
        else:
            continue

    all_stock_dates = []

    for stock_symbol in chosen_stocks:
        stock_dates = finnish_stock_daily.objects.filter(
            symbol=stock_symbol,
            date__range=(start_date, end_date)
        ).values_list('date', flat=True)

        all_stock_dates.extend(stock_dates)

    all_stock_dates = sorted(list(set(all_stock_dates)))

    all_investment_changes = {}
    last_signals = {symbol: ('SELL', initial_investment) for symbol in chosen_stocks}

    symbols_with_investments = {date: set() for date in all_stock_dates}

    for symbol, investments_list in investments.items():
        for investment_data in investments_list:
            investment_date, _, _ = investment_data
            symbols_with_investments[investment_date].add(symbol)

    for date in all_stock_dates:
        total_investment_for_date = 0
        for symbol in chosen_stocks:
            if symbol in symbols_with_investments[date]:
                for investment_data in investments.get(symbol, []):
                    investment_date, investment_value_or_stocks, signal = investment_data
                    if investment_date == date:
                        if signal == 'SELL':
                            last_signals[symbol] = (signal, investment_value_or_stocks)
                            investment_value = investment_value_or_stocks
                        else:
                            investment_value = None
                            previous_date = date
                            days_to_subtract = 1
                            last_signals[symbol] = (signal, investment_value_or_stocks)
                            try:
                                stock_close_price = stock_data.filter(date=date, symbol=symbol).first().close
                            except AttributeError:
                                stock_close_price = 0
                                print(f"Warning: No stock data found for {symbol} on {date}")
                            if stock_close_price == 0:
                                while investment_value is None and days_to_subtract <= 5:
                                    previous_date -= timedelta(days=1)
                                    investment_value = all_investment_changes.get(previous_date)
                                    days_to_subtract += 1
                            else:
                                investment_value = investment_value_or_stocks * stock_close_price
                        total_investment_for_date += investment_value
            else:
                last_signal, stocks = last_signals[symbol]
                if last_signal == 'BUY':
                    investment_value = None
                    previous_date = date
                    days_to_subtract = 1
                    try:
                        stock_close_price = stock_data.filter(date=date, symbol=symbol).first().close
                    except AttributeError:
                        stock_close_price = 0
                        print(f"Warning: No stock data found for {symbol} on {date}")
                    if stock_close_price == 0:
                        while investment_value is None and days_to_subtract <= 5:
                            previous_date -= timedelta(days=1)
                            investment_value = all_investment_changes.get(previous_date)
                            days_to_subtract += 1
                    else:
                        investment_value = stocks * stock_close_price
                    total_investment_for_date += investment_value
                else:
                    investment_value = stocks
                    total_investment_for_date += investment_value

        all_investment_changes[date] = total_investment_for_date
    all_investment_dates = sorted(all_investment_changes.keys())
    all_investment_values = [all_investment_changes[date] for date in all_investment_dates]

    initial_investment_total = initial_investment * len(chosen_stocks)
    all_investment_changes_percentage = [
        (100 * (all_investment_values[i] - initial_investment_total) / initial_investment_total) for i in
        range(1, len(all_investment_values))]

    dates_for_investment = [mdates.date2num(date) for date in all_investment_dates[1:]]

    dates = [mdates.date2num(date) for date in sp_500_dates[1:]]

    plt.figure(figsize=(16, 8))
    plt.plot(dates_for_investment, all_investment_changes_percentage, label='Investment Value Change (%)', color='blue')
    plt.plot(dates, sp_500_changes, label='SP&500 Value Change (%)', color='yellow')
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(25))
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.title('Percentage Change in Stock Value and Investment')
    plt.legend()
    plt.show()

    return sell_counter + buy_counter, investments
