from django.db import models


class finnish_stock_daily(models.Model):
    symbol = models.CharField(max_length=10)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.IntegerField()
    bid = models.FloatField(null=True)
    ask = models.FloatField(null=True)
    turnover = models.FloatField(null=True)
    trades = models.IntegerField(null=True)
    average = models.FloatField(null=True)
    date = models.DateField()


class signals(models.Model):
    stock = models.ForeignKey(finnish_stock_daily, on_delete=models.CASCADE, default=None)
    symbol = models.CharField(max_length=10)
    adl = models.FloatField(null=True)
    obv = models.FloatField(null=True)
    bbp8 = models.FloatField(null=True)
    adx = models.FloatField(null=True)
    rsi7 = models.FloatField(null=True)
    rsi14 = models.FloatField(null=True)
    rsi50 = models.FloatField(null=True)
    aroon_up = models.FloatField(null=True)
    aroon_down = models.FloatField(null=True)
    macd = models.FloatField(null=True)
    macd_signal = models.FloatField(null=True)
    std_dev = models.FloatField(null=True)
    ema20 = models.FloatField(null=True)
    ema50 = models.FloatField(null=True)
    ema100 = models.FloatField(null=True)
    ema200 = models.FloatField(null=True)


class reverse_signals(models.Model):
    stock = models.ForeignKey(finnish_stock_daily, on_delete=models.CASCADE, default=None)
    symbol = models.CharField(max_length=10)
    adl = models.FloatField(null=True)
    obv = models.FloatField(null=True)
    bbp8 = models.FloatField(null=True)
    adx = models.FloatField(null=True)
    rsi5 = models.FloatField(null=True)
    rsi7 = models.FloatField(null=True)
    macd = models.FloatField(null=True)
    macd_signal = models.FloatField(null=True)
    std_dev = models.FloatField(null=True)
    ema5 = models.FloatField(null=True)


class optimal_buy_sell_points(models.Model):
    stock = models.ForeignKey(finnish_stock_daily, on_delete=models.CASCADE, default=None)
    symbol = models.CharField(max_length=10)
    command = models.CharField(max_length=10)
    value = models.FloatField()

