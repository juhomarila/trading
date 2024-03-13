from django.core.mail import send_mail
from django.http import HttpResponse
from datetime import datetime, timedelta
from . import settings
from .models import optimal_buy_sell_points, finnish_stock_daily


def send_email(request):
    current_date = datetime.now().date()
    subject = f"KOMENNOT {current_date}"
    message = calculate_from_buy_sell_points_when_to_send(current_date)
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [settings.EMAIL_HOST_USER, ]
    if message:
        send_mail(subject, "\n".join(message), email_from, recipient_list)
        return HttpResponse(status=201)
    else:
        return HttpResponse(status=200)


def calculate_from_buy_sell_points_when_to_send(current_date):
    symbols = finnish_stock_daily.objects.values('symbol').distinct()
    message = []
    for i in range(len(symbols)):
        stocks = finnish_stock_daily.objects.filter(symbol=symbols[i]['symbol']).order_by('-date')[:6]
        newest_stock = stocks.first()
        oldest_stock = stocks[5]
        if optimal_buy_sell_points.objects.filter(stock=oldest_stock).exists() and newest_stock.date + timedelta(
                days=1) == current_date:
            optimal = optimal_buy_sell_points.objects.filter(stock=oldest_stock).first()
            optimal_prev = optimal_buy_sell_points.objects.filter(symbol=oldest_stock.symbol,
                                                                  stock__date__lt=oldest_stock.date).order_by(
                '-stock__date').first()
            if optimal.command != optimal_prev.command:
                message.append(
                    f'{optimal.command}: {newest_stock.symbol}, @CLOSING {newest_stock.close}, @DATE: {newest_stock.date}')
            else:
                continue
        else:
            continue
    return message
