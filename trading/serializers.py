
from rest_framework import serializers
from .models import finnish_stock_daily, signals


class finnish_stock_daily_serializer(serializers.ModelSerializer):
    class Meta:
        model = finnish_stock_daily
        fields = ('symbol', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask', 'turnover', 'trades', 'date')


class signals_serializer(serializers.ModelSerializer):
    class Meta:
        model = signals
        fields = ('stock', 'symbol', 'adl', 'obv', 'bbp8', 'adx', 'rsi',
                  'aroon_up', 'aroon_down', 'macd', 'macd_signal', 'macd_signal',
                  'std_dev', 'ema20', 'ema50', 'ema100', 'ema200')
