import models
import csv
from datetime import datetime
import os


def store_stock_data_from_csv(file_path, symbol):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            # convert the date string to a Python datetime object
            date = datetime.strptime(row['Date'], '%Y-%m-%d').date()
            try:
                # create a new Stock object and populate its fields
                stock = models.Stock_daily(
                    symbol=symbol,
                    date=date,
                    open=float(row['Opening price'].replace(',', '.')),
                    high=float(row['High price'].replace(',', '.')),
                    low=float(row['Low price'].replace(',', '.')),
                    close=float(row['Closing price'].replace(',', '.')),
                    volume=int(row['Total volume']),
                )
                stock.save()
            except ValueError:
                # handle any conversion errors that may occur
                pass


# def process_file(csv_file):
#     store_stock_data_from_csv(csv_file, 'SYMBOL')
#     pass

print(os.get_exec_path())
folder_path = '/path/to/folder'

#
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
#     if os.path.isfile(file_path):
#         with open(file_path, 'r') as file:
#             process_file(file)
