import base64
import io
from datetime import datetime, timedelta
from io import BytesIO
import random
from ultralytics import YOLOv10
import matplotlib
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.http import FileResponse
from django.utils.timezone import make_aware, is_naive
from rest_framework import status
from rest_framework.authentication import TokenAuthentication, SessionAuthentication
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action, authentication_classes, permission_classes
import matplotlib.pyplot as plt
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from smart_home.models import Home_electricity_consumption
from smart_home.serializers import CreateUserSerializer, HomeElectricityConsumptionSerializer
from smart_home.static_numbers import reports
import pandas as pd
import tensorflow as tf
import joblib
import plotly.express as px
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
matplotlib.use('agg')



class RegisterUser(APIView):
    def post(self, request):
        serializer = CreateUserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            user = User.objects.get(username=request.data['username'])
            token = Token.objects.create(user=user)
            return Response({"token": token.key, "user": serializer.data}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginUser(APIView):

    def post(self, request):
        user = get_object_or_404(User, username=request.data.get('username'))
        if not user.check_password(request.data.get('password')):
            return Response({"details": "Incorrect username or password"}, status=status.HTTP_404_NOT_FOUND)
        token, created = Token.objects.get_or_create(user=user)
        serializer = CreateUserSerializer(instance=user)
        print(token)
        return Response({"token": token.key, "user": serializer.data}, status=status.HTTP_200_OK)

class LogoutUser(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        request.user.auth_token.delete()
        logout(request)
        return Response({'message': "Successful logout"}, status=status.HTTP_200_OK)


class GetDataForHomeConsumption(APIView):
    def get(self, request):
        random_report_key = random.choice(list(reports.keys()))
        random_report = reports[random_report_key]

        data = {
            "Global_active_power": random_report["Global_active_power"],
            "Global_reactive_power": random_report["Global_reactive_power"],
            "Global_intensity": random_report["Global_intensity"],
            "Voltage": random_report["Voltage"],
            "Sub_metering_1": random_report["Sub_metering_1"],
            "Sub_metering_2": random_report["Sub_metering_2"],
            "Sub_metering_3": random_report["Sub_metering_3"]
        }
        return Response({"dataForHomeConsumption": data}, status=status.HTTP_200_OK)
class CreateReport(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = HomeElectricityConsumptionSerializer(data=request.data)
        if serializer.is_valid():
            Home_electricity_consumption.objects.create(
                user=request.user,
                global_active_power=serializer.validated_data['global_active_power'],
                global_reactive_power=serializer.validated_data['global_reactive_power'],
                voltage=serializer.validated_data['voltage'],
                global_intensity=serializer.validated_data['global_intensity'],
                sub_metering_1=serializer.validated_data['sub_metering_1'],
                sub_metering_2=serializer.validated_data['sub_metering_2'],
                sub_metering_3=serializer.validated_data['sub_metering_3'],
            )
            return Response({"message": "Report created successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class PredictGlobalActivePowerDaily(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_day = make_aware(datetime.combine(current_date, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(current_date, datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_day, end_of_day))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user on the current day"}, status=status.HTTP_400_BAD_REQUEST)


        aggregated_data = data.mean()
        global_intensity = aggregated_data['global_intensity']
        voltage = aggregated_data['voltage']

        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'Adam': tf.keras.optimizers.Adam
        }
        model = tf.keras.models.load_model('smart_home/Models/Global_active_power/my_model.keras', custom_objects=custom_objects)
        ct = joblib.load('smart_home/Models/Global_active_power/column_transformer.pkl')


        input_data = pd.DataFrame([[global_intensity, voltage]], columns=['Global_intensity', 'Voltage'])
        input_standardized = ct.transform(input_data)


        prediction = model.predict(input_standardized)

        result = {
            "predicted_global_active_power": prediction[0][0]
        }

        return Response(result, status=status.HTTP_200_OK)


class PredictGlobalActivePowerWeekly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_week = current_date - timedelta(days=current_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)


        start_of_week = make_aware(datetime.combine(start_of_week, datetime.min.time()))
        end_of_week = make_aware(datetime.combine(end_of_week, datetime.max.time()))

        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_week, end_of_week))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current week"}, status=status.HTTP_400_BAD_REQUEST)


        data['date'] = pd.to_datetime(data['date']).dt.date
        daily_aggregated_data = data.groupby('date').mean()


        weekly_averages = daily_aggregated_data.mean()
        print(weekly_averages)
        global_intensity = weekly_averages['global_intensity']
        voltage = weekly_averages['voltage']


        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'Adam': tf.keras.optimizers.Adam
        }
        model = tf.keras.models.load_model('smart_home/Models/Global_active_power/my_model.keras', custom_objects=custom_objects)
        ct = joblib.load('smart_home/Models/Global_active_power/column_transformer.pkl')


        input_data = pd.DataFrame([[global_intensity, voltage]], columns=['Global_intensity', 'Voltage'])
        input_standardized = ct.transform(input_data)


        prediction = model.predict(input_standardized)

        result = {
            "predicted_global_active_power": prediction[0][0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictGlobalActivePowerMonthly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_month = current_date.replace(day=1)
        next_month = (start_of_month.replace(day=28) + timedelta(days=4)).replace(day=1)
        end_of_month = next_month - timedelta(days=1)


        start_of_month = make_aware(datetime.combine(start_of_month, datetime.min.time()))
        end_of_month = make_aware(datetime.combine(end_of_month, datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_month, end_of_month))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current month"}, status=status.HTTP_400_BAD_REQUEST)


        data['date'] = pd.to_datetime(data['date']).dt.date
        daily_aggregated_data = data.groupby('date').mean()


        monthly_averages = daily_aggregated_data.mean()
        global_intensity = monthly_averages['global_intensity']
        voltage = monthly_averages['voltage']


        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'Adam': tf.keras.optimizers.Adam
        }
        model = tf.keras.models.load_model('smart_home/Models/Global_active_power/my_model.keras', custom_objects=custom_objects)
        ct = joblib.load('smart_home/Models/Global_active_power/column_transformer.pkl')

        input_data = pd.DataFrame([[global_intensity, voltage]], columns=['Global_intensity', 'Voltage'])
        input_standardized = ct.transform(input_data)

        prediction = model.predict(input_standardized)

        result = {
            "predicted_global_active_power": prediction[0][0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering1Daily(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        current_date = datetime.now().date()
        start_of_day = make_aware(datetime.combine(current_date, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(current_date, datetime.max.time()))

        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_day, end_of_day))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user on the current day"}, status=status.HTTP_400_BAD_REQUEST)
        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_2 = aggregated_data['sub_metering_2']
        sub_metering_3 = aggregated_data['sub_metering_3']


        loaded_model = joblib.load('smart_home/Models/Sub_metering_1/best_knn_model_sub_1.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_1/scaler_sub_1.pkl')


        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_2, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)

        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering1": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)
class PredictSubMetering1Weekly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_week = current_date - timedelta(days=current_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)


        start_of_week = make_aware(datetime.combine(start_of_week, datetime.min.time()))
        end_of_week = make_aware(datetime.combine(end_of_week, datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_week, end_of_week))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current week"}, status=status.HTTP_400_BAD_REQUEST)


        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_2 = aggregated_data['sub_metering_2']
        sub_metering_3 = aggregated_data['sub_metering_3']


        loaded_model = joblib.load('smart_home/Models/Sub_metering_1/best_knn_model_sub_1.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_1/scaler_sub_1.pkl')


        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_2, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)


        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering1": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)
class PredictSubMetering1Monthly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_month = current_date.replace(day=1)
        end_of_month = start_of_month.replace(day=1, month=start_of_month.month+1) - timedelta(days=1)


        start_of_month = make_aware(datetime.combine(start_of_month, datetime.min.time()))
        end_of_month = make_aware(datetime.combine(end_of_month, datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_month, end_of_month))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current month"}, status=status.HTTP_400_BAD_REQUEST)


        aggregated_data = data.mean()
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_2 = aggregated_data['sub_metering_2']
        sub_metering_3 = aggregated_data['sub_metering_3']


        loaded_model = joblib.load('smart_home/Models/Sub_metering_1/best_knn_model_sub_1.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_1/scaler_sub_1.pkl')


        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_2, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)


        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering1": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering2Daily(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_day = make_aware(datetime.combine(current_date, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(current_date, datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_day, end_of_day))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user on the current day"},
                            status=status.HTTP_400_BAD_REQUEST)

        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_1 = aggregated_data['sub_metering_1']
        sub_metering_3 = aggregated_data['sub_metering_3']


        loaded_model = joblib.load('smart_home/Models/Sub_metering_2/best_knn_model_sub_2.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_2/scaler_sub_2.pkl')


        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_1, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_1',
                                           'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)


        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering2": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering2Weekly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_week = current_date - timedelta(days=current_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)


        start_of_week = make_aware(datetime.combine(start_of_week, datetime.min.time()))
        end_of_week = make_aware(datetime.combine(end_of_week, datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_week, end_of_week))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current week"}, status=status.HTTP_400_BAD_REQUEST)


        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_1 = aggregated_data['sub_metering_1']
        sub_metering_3 = aggregated_data['sub_metering_3']


        loaded_model = joblib.load('smart_home/Models/Sub_metering_2/best_knn_model_sub_2.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_2/scaler_sub_2.pkl')


        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_1, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)


        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering2": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering2Monthly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_month = current_date.replace(day=1)
        end_of_month = start_of_month.replace(day=1, month=start_of_month.month+1) - timedelta(days=1)


        start_of_month = make_aware(datetime.combine(start_of_month, datetime.min.time()))
        end_of_month = make_aware(datetime.combine(end_of_month, datetime.max.time()))

        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_month, end_of_month))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current month"}, status=status.HTTP_400_BAD_REQUEST)


        aggregated_data = data.mean()
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_1 = aggregated_data['sub_metering_1']
        sub_metering_3 = aggregated_data['sub_metering_3']

        loaded_model = joblib.load('smart_home/Models/Sub_metering_2/best_knn_model_sub_2.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_2/scaler_sub_2.pkl')

        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_1, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)

        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering2": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering3Daily(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_day = make_aware(datetime.combine(current_date, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(current_date, datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_day, end_of_day))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user on the current day"}, status=status.HTTP_400_BAD_REQUEST)


        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']


        loaded_model = joblib.load('smart_home/Models/Sub_metering_3/best_knn_model.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_3/scaler.pkl')

        input_data = pd.DataFrame([[global_active_power, global_intensity]],
                                  columns=['Global_active_power', 'Global_intensity'])
        input_scaled = scaler.transform(input_data)

        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering3": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering3Weekly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_week = current_date - timedelta(days=current_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)


        start_of_week = make_aware(datetime.combine(start_of_week, datetime.min.time()))
        end_of_week = make_aware(datetime.combine(end_of_week, datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_week, end_of_week))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current week"}, status=status.HTTP_400_BAD_REQUEST)


        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']


        loaded_model = joblib.load('smart_home/Models/Sub_metering_3/best_knn_model.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_3/scaler.pkl')

        input_data = pd.DataFrame([[global_active_power, global_intensity]],
                                  columns=['Global_active_power', 'Global_intensity'])
        input_scaled = scaler.transform(input_data)

        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering3": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering3Monthly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        current_date = datetime.now().date()
        start_of_month = current_date.replace(day=1)
        end_of_month = start_of_month.replace(day=1, month=start_of_month.month+1) - timedelta(days=1)

        start_of_month = make_aware(datetime.combine(start_of_month, datetime.min.time()))
        end_of_month = make_aware(datetime.combine(end_of_month, datetime.max.time()))

        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_month, end_of_month))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current month"}, status=status.HTTP_400_BAD_REQUEST)

        aggregated_data = data.mean()
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']



        loaded_model = joblib.load('smart_home/Models/Sub_metering_3/best_knn_model.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_3/scaler.pkl')


        input_data = pd.DataFrame([[global_active_power, global_intensity]],
                                  columns=['Global_active_power', 'Global_intensity'])
        input_scaled = scaler.transform(input_data)


        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering3": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class BaseGenerateDailyPlot(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    metering_field = None
    plot_title = None
    plot_ylabel = None

    def get(self, request):
        user = request.user

        current_date = datetime.now().date()
        start_of_day = make_aware(datetime.combine(current_date, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(current_date, datetime.max.time()))

        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_day, end_of_day))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user on the current day"}, status=status.HTTP_400_BAD_REQUEST)

        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_day) & (data['date'] <= end_of_day)]

        hourly_data = data.groupby(data['date'].dt.hour)[self.metering_field].mean().reset_index()

        plt.figure(figsize=(10, 8))
        plt.plot(hourly_data['date'], hourly_data[self.metering_field], marker='o')
        plt.title(self.plot_title)
        plt.xlabel('Hour of Day')
        plt.ylabel(self.plot_ylabel)
        plt.grid(True)
        plt.xticks(hourly_data['date'], hourly_data['date'])

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        return FileResponse(buffer, content_type='image/png')

class GenerateDailyPlotForGlobalActivePower(BaseGenerateDailyPlot):
    metering_field = 'global_active_power'
    plot_title = 'Average Hourly Global Active Power for the Current Day'
    plot_ylabel = 'Global Active Power'


class GenerateDailyPlotForSubMetering1(BaseGenerateDailyPlot):
    metering_field = 'sub_metering_1'
    plot_title = 'Average Hourly Consumption for the Kitchen for the Current Day'
    plot_ylabel = 'Consumption by Kitchen'


class GenerateDailyPlotForSubMetering2(BaseGenerateDailyPlot):
    metering_field = 'sub_metering_2'
    plot_title = 'Average Hourly Consumption for the Laundry Room for the Current Day'
    plot_ylabel = 'Consumption by Laundry Room'


class GenerateDailyPlotForSubMetering3(BaseGenerateDailyPlot):
    metering_field = 'sub_metering_3'
    plot_title = 'Average Hourly Consumption for the Living Room for the Current Day'
    plot_ylabel = 'Consumption by Living Room'

class BaseGenerateWeeklyPlot(APIView):
    metering_field = None
    plot_title = None
    plot_ylabel = None
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        current_date = datetime.now().date()
        start_of_week = current_date - timedelta(days=current_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)

        start_of_week = make_aware(datetime.combine(start_of_week, datetime.min.time()))
        end_of_week = make_aware(datetime.combine(end_of_week, datetime.max.time()))

        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_week, end_of_week))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user in the current week"},
                            status=status.HTTP_400_BAD_REQUEST)

        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_week) & (data['date'] <= end_of_week)]

        daily_data = data.groupby(data['date'].dt.date)[self.metering_field].mean().reset_index()

        plt.figure(figsize=(10, 8))
        plt.plot(daily_data['date'], daily_data[self.metering_field], marker='o')
        plt.title(self.plot_title)
        plt.xlabel('Date')
        plt.ylabel(self.plot_ylabel)
        plt.grid(True)

        plt.xticks(daily_data['date'], daily_data['date'].apply(lambda x: x.strftime('%Y-%m-%d')))

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        return FileResponse(buffer, content_type='image/png')
class GenerateWeeklyPlotForGlobalActivePower(BaseGenerateWeeklyPlot):
    metering_field = 'global_active_power'
    plot_title = 'Average Daily Global Active Power for the Current Week'
    plot_ylabel = 'Global Active Power'

class GenerateWeeklyPlotForSubMetering1(BaseGenerateWeeklyPlot):
    metering_field = 'sub_metering_1'
    plot_title = 'Average Daily Consumption of the kitchen for the Current Week'
    plot_ylabel = 'Kitchen consumption'

class GenerateWeeklyPlotForSubMetering2(BaseGenerateWeeklyPlot):
    metering_field = 'sub_metering_3'
    plot_title = 'Average Daily Consumption of the laundry room for the Current Week'
    plot_ylabel = 'Living room consumption'

class GenerateWeeklyPlotForSubMetering3(BaseGenerateWeeklyPlot):
    metering_field = 'sub_metering_3'
    plot_title = 'Average Daily Consumption of the Living room for the Current Week'
    plot_ylabel = 'Laundry room consumption'

class BaseGenerateMonthlyPlot(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    metering_field = None
    plot_title = None
    plot_ylabel = None
    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_month = make_aware(datetime(current_date.year, current_date.month, 1))
        end_of_month = make_aware(
            datetime.combine((datetime(current_date.year, current_date.month, 1) + pd.offsets.MonthEnd()).date(),
                             datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_month, end_of_month))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user in the current month"},
                            status=status.HTTP_400_BAD_REQUEST)


        data['date'] = pd.to_datetime(data['date'])


        data.set_index('date', inplace=True)
        daily_data = data[self.metering_field].resample('D').mean().reset_index()


        all_days = pd.date_range(start=start_of_month, end=end_of_month, freq='D')
        daily_data = daily_data.set_index('date').reindex(all_days).reset_index()
        daily_data.columns = ['date', self.metering_field]


        daily_data[self.metering_field].fillna(0, inplace=True)


        plt.figure(figsize=(10, 8))
        plt.plot(daily_data['date'], daily_data[self.metering_field], marker='o')
        plt.title(self.plot_title)
        plt.xlabel('Date')
        plt.ylabel(self.plot_ylabel)
        plt.grid(True)


        plt.gca().set_xticks(daily_data['date'])
        plt.gca().set_xticklabels(daily_data['date'].dt.strftime('%d'))


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')
class GenerateMonthlyPlotForGlobalActivePower(BaseGenerateMonthlyPlot):
    metering_field = 'global_active_power'
    plot_title ='Average Daily Global Active Power for the Current Month'
    plot_ylabel = 'Global Active Power'

class GenerateMonthlyPlotForSubMetering1(BaseGenerateMonthlyPlot):
    metering_field = 'sub_metering_1'
    plot_title ='Average Daily Consumption for the kitchen for the Current Month'
    plot_ylabel = 'Kitchen consumption'

class GenerateMonthlyPlotForSubMetering2(BaseGenerateMonthlyPlot):
    metering_field = 'sub_metering_2'
    plot_title ='Average Daily Consumption for the Laundry Room for the Current Month'
    plot_ylabel = 'Laundry Room Consumption'

class GenerateMonthlyPlotForSubMetering3(BaseGenerateMonthlyPlot):
    metering_field = 'sub_metering_3'
    plot_title ='Average Daily Consumption for the Living Room for the Current Month'
    plot_ylabel = 'Living Room Consumption'

class ObjectDetect(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    def post(self, request):

        if 'file' not in request.data:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        uploaded_file = request.data['file']
        file_path = default_storage.save(f'tmp/{uploaded_file.name}', uploaded_file)

        try:
            model = YOLOv10('smart_home/Models/YoloV10/best.pt')
            results = model(file_path, conf=0.25)
            names = {
                0: 'Bed', 1: 'Chair', 2: 'Couch', 3: 'Dining table', 4: 'Laptop', 5: 'Microwave',
                6: 'Oven', 7: 'Refrigerator', 8: 'Sink', 9: 'Toaster', 10: 'Toilet', 11: 'TV'
            }
            answers = []

            for r in results:
                for box in r.boxes:
                    if box.cls.item() in names.keys():
                        answers.append(names[box.cls.item()])


            default_storage.delete(file_path)

            return Response({'results': answers}, status=status.HTTP_200_OK)

        except Exception as e:
            default_storage.delete(file_path)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GenerateConsumptionPlot(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, time_frame):
        user = request.user
        current_date = datetime.now().date()

        if time_frame == 'daily':
            start_date = make_aware(datetime.combine(current_date, datetime.min.time()))
            end_date = make_aware(datetime.combine(current_date, datetime.max.time()))
        elif time_frame == 'weekly':
            start_date = current_date - timedelta(days=current_date.weekday())
            end_date = start_date + timedelta(days=6)
            start_date = make_aware(datetime.combine(start_date, datetime.min.time()))
            end_date = make_aware(datetime.combine(end_date, datetime.max.time()))
        elif time_frame == 'monthly':
            start_date = current_date.replace(day=1)
            end_date = (start_date.replace(month=start_date.month % 12 + 1, day=1) - timedelta(days=1))
            start_date = make_aware(datetime.combine(start_date, datetime.min.time()))
            end_date = make_aware(datetime.combine(end_date, datetime.max.time()))
        else:
            return Response({"error": "Invalid time frame specified"}, status=status.HTTP_400_BAD_REQUEST)

        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_date, end_date))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the specified time frame"}, status=status.HTTP_400_BAD_REQUEST)

        sub_metering_sums = data[['sub_metering_1', 'sub_metering_2', 'sub_metering_3']].sum()
        labels = ['Kitchen', 'Laundry room', 'Living room']

        plt.figure(figsize=(10, 8))
        plt.bar(labels, sub_metering_sums.values, color=['blue', 'orange', 'green'])
        plt.title(f'Total Consumption by Sub-Metering {time_frame.capitalize()}')
        plt.xlabel('Places in home')
        plt.ylabel('Total Consumption (kW)')
        plt.grid(True)

        for i, v in enumerate(sub_metering_sums.values):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        return FileResponse(buffer, content_type='image/png')


