from django.urls import path
from .views import *

urlpatterns = [
    path('register', RegisterUser.as_view(), name='register'),
    path('login', LoginUser.as_view(), name='login'),
    path('logout', LogoutUser.as_view(), name='logout'),
    path('get_info',GetDataForHomeConsumption.as_view(),name='get_info'),
    path('create_report', CreateReport.as_view(), name='create_report'),
    path('predict_global_active_power', PredictGlobalActivePower.as_view(), name='predict_global_active_power'),
    path('generate_plot_daily_global_active_power', GenerateCurrentDayPlotForGlobalActivePower.as_view(), name='generate_plot_daily_global_active_power'),
    path('generate_plot_weekly_global_active_power', GenerateWeeklyPlotForGlobalActivePower.as_view(), name='generate_plot+weekly_global_active_power'),
    path('generate_plot_monthly_global_active_power', GenerateMonthlyPlotForGlobalActivePower.as_view(), name='generate_plot_monthly_global_active_power'),
    path('generate_plot_daily_sub_metering_1', GenerateDailyPlotForSubMetering1.as_view(), name='generate_plot_daily_sub_metering_1'),
    path('generate_plot_weekly_sub_metering_1', GenerateWeeklyPlotForSubMetering1.as_view(), name='generate_plot_weekly_sub_metering_1'),
    path('generate_plot_monthly_sub_metering_1', GenerateMonthlyPlotForSubMetering1.as_view(), name='generate_plot_monthly_sub_metering1'),
    path('generate_plot_daily_sub_metering_2', GenerateDailyPlotForSubMetering2.as_view(), name='generate_plot_daily_sub_metering_2'),
    path('generate_plot_weekly_sub_metering_2', GenerateWeeklyPlotForSubMetering2.as_view(),name='generate_plot_weekly_sub_metering_2'),
    path('generate_plot_monthly_sub_metering_2', GenerateMonthlyPlotForSubMetering2.as_view(),name='generate_plot_monthly_sub_metering_2'),
    path('generate_plot_daily_sub_metering_3', GenerateDailyPlotForSubMetering3.as_view(), name='generate_plot_daily_sub_metering_3'),
    path('generate_plot_weekly_sub_metering_3', GenerateWeeklyPlotForSubMetering3.as_view(),name='generate_plot_weekly_sub_metering_3'),
    path('generate_plot_monthly_sub_metering_3', GenerateMonthlyPlotForSubMetering3.as_view(),name='generate_plot_monthly_sub_metering_3'),



    #path('plot', PlotView.as_view(), name='api-plot')
]