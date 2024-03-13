from datetime import datetime

from django.urls import path

from . import views

urlpatterns = [
    path('index', views.index, name='index'),
    # path('scrapeintradaydata', views.scrape_intraday_data, name='scrapeintradaydata'),
    # path('getintradaydata', views.get_intraday_data_by_ticker, name='getintradaydata'),
    path('processcsvdata', views.process_csv_data, name='processcsvdata'),
    path('processbulkdata', views.process_bulk_data, name='processbulkdata'),
    path('getdailydata', views.get_daily_data, name='getdailydata'),
    path('processdailydata', views.process_daily_data, name='processdailydata'),
    path('findbuysellpoints', views.find_buy_sell_points, name='findbuysellpoints'),
    path('', views.index, name='index'),
    path('settings/<str:symbol>/', views.settings, name='settings'),
    path('visualize/', views.visualize, name='visualize'),
    path('signals/<str:symbol>/', views.signals_page, name='signals_page'),
    path('strategy/', views.strategy, name='strategy'),
    path('createdstrategy/', views.created_strategy, name='created_strategy'),
    path('upload-csv/', views.process_uploaded_csv, name='upload_csv'),
    path('process-data/', views.process_data, name='process_data'),
]