# Generated by Django 4.1.5 on 2023-01-30 21:06

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('trading', '0016_alter_finnish_stock_daily_ask_and_more'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='optimal_buy_point',
            new_name='optimal_buy_sell_points',
        ),
        migrations.DeleteModel(
            name='optimal_sell_point',
        ),
    ]
