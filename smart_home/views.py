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

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_intensity, voltage]], columns=['Global_intensity', 'Voltage'])
        input_standardized = ct.transform(input_data)

        # Predict the global active power for the next day
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

        # Define current week range (assuming week starts on Monday)
        current_date = datetime.now().date()
        start_of_week = current_date - timedelta(days=current_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)

        # Convert to aware datetime objects
        start_of_week = make_aware(datetime.combine(start_of_week, datetime.min.time()))
        end_of_week = make_aware(datetime.combine(end_of_week, datetime.max.time()))

        # Fetch data for the current week
        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_week, end_of_week))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current week"}, status=status.HTTP_400_BAD_REQUEST)

        # Aggregate the data by day (mean, sum, etc.)
        data['date'] = pd.to_datetime(data['date']).dt.date
        daily_aggregated_data = data.groupby('date').mean()

        # Prepare weekly averaged input data
        weekly_averages = daily_aggregated_data.mean()
        global_intensity = weekly_averages['global_intensity']
        voltage = weekly_averages['voltage']

        # Load the model and column transformer
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'Adam': tf.keras.optimizers.Adam
        }
        model = tf.keras.models.load_model('smart_home/Models/Global_active_power/my_model.keras', custom_objects=custom_objects)
        ct = joblib.load('smart_home/Models/Global_active_power/column_transformer.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_intensity, voltage]], columns=['Global_intensity', 'Voltage'])
        input_standardized = ct.transform(input_data)

        # Predict the global active power for the next week
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

        # Define current month range
        current_date = datetime.now().date()
        start_of_month = current_date.replace(day=1)
        next_month = (start_of_month.replace(day=28) + timedelta(days=4)).replace(day=1)
        end_of_month = next_month - timedelta(days=1)

        # Convert to aware datetime objects
        start_of_month = make_aware(datetime.combine(start_of_month, datetime.min.time()))
        end_of_month = make_aware(datetime.combine(end_of_month, datetime.max.time()))

        # Fetch data for the current month
        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_month, end_of_month))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current month"}, status=status.HTTP_400_BAD_REQUEST)

        # Aggregate the data by day (mean, sum, etc.)
        data['date'] = pd.to_datetime(data['date']).dt.date
        daily_aggregated_data = data.groupby('date').mean()

        # Prepare monthly averaged input data
        monthly_averages = daily_aggregated_data.mean()
        global_intensity = monthly_averages['global_intensity']
        voltage = monthly_averages['voltage']

        # Load the model and column transformer
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'Adam': tf.keras.optimizers.Adam
        }
        model = tf.keras.models.load_model('smart_home/Models/Global_active_power/my_model.keras', custom_objects=custom_objects)
        ct = joblib.load('smart_home/Models/Global_active_power/column_transformer.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_intensity, voltage]], columns=['Global_intensity', 'Voltage'])
        input_standardized = ct.transform(input_data)

        # Predict the global active power for the next month
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

        # Define current day range
        current_date = datetime.now().date()
        start_of_day = make_aware(datetime.combine(current_date, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(current_date, datetime.max.time()))

        # Fetch data for the current day
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

        # Load the KNN model and the scaler
        loaded_model = joblib.load('smart_home/Models/Sub_metering_1/best_knn_model_sub_1.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_1/scaler_sub_1.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_2, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)

        # Predict the sub metering 1 for the next day
        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering_1": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)
class PredictSubMetering1Weekly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        # Define current week range (assuming week starts on Monday)
        current_date = datetime.now().date()
        start_of_week = current_date - timedelta(days=current_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)

        # Convert to aware datetime objects
        start_of_week = make_aware(datetime.combine(start_of_week, datetime.min.time()))
        end_of_week = make_aware(datetime.combine(end_of_week, datetime.max.time()))

        # Fetch data for the current week
        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_week, end_of_week))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current week"}, status=status.HTTP_400_BAD_REQUEST)

        # Aggregate the data (mean, sum, etc.)
        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_2 = aggregated_data['sub_metering_2']
        sub_metering_3 = aggregated_data['sub_metering_3']

        # Load the KNN model and the scaler
        loaded_model = joblib.load('smart_home/Models/Sub_metering_1/best_knn_model_sub_1.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_1/scaler_sub_1.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_2, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)

        # Predict the sub metering 1 for the next week
        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering_1": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)
class PredictSubMetering1Monthly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        # Define current month range
        current_date = datetime.now().date()
        start_of_month = current_date.replace(day=1)
        end_of_month = start_of_month.replace(day=1, month=start_of_month.month+1) - timedelta(days=1)

        # Convert to aware datetime objects
        start_of_month = make_aware(datetime.combine(start_of_month, datetime.min.time()))
        end_of_month = make_aware(datetime.combine(end_of_month, datetime.max.time()))

        # Fetch data for the current month
        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_month, end_of_month))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current month"}, status=status.HTTP_400_BAD_REQUEST)

        # Aggregate the data (mean, sum, etc.)
        aggregated_data = data.mean()
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_2 = aggregated_data['sub_metering_2']
        sub_metering_3 = aggregated_data['sub_metering_3']

        # Load the KNN model and the scaler
        loaded_model = joblib.load('smart_home/Models/Sub_metering_1/best_knn_model_sub_1.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_1/scaler_sub_1.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_2, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)

        # Predict the sub metering 1 for the next month
        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering_1": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering2Daily(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        # Define current day range
        current_date = datetime.now().date()
        start_of_day = make_aware(datetime.combine(current_date, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(current_date, datetime.max.time()))

        # Fetch data for the current day
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

        # Load the KNN model and the scaler
        loaded_model = joblib.load('smart_home/Models/Sub_metering_2/best_knn_model_sub_2.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_2/scaler_sub_2.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_1, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_1',
                                           'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)

        # Predict the sub metering 1 for the next day
        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering_2": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering2Weekly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        # Define current week range (assuming week starts on Monday)
        current_date = datetime.now().date()
        start_of_week = current_date - timedelta(days=current_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)

        # Convert to aware datetime objects
        start_of_week = make_aware(datetime.combine(start_of_week, datetime.min.time()))
        end_of_week = make_aware(datetime.combine(end_of_week, datetime.max.time()))

        # Fetch data for the current week
        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_week, end_of_week))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current week"}, status=status.HTTP_400_BAD_REQUEST)

        # Aggregate the data (mean, sum, etc.)
        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_1 = aggregated_data['sub_metering_1']
        sub_metering_3 = aggregated_data['sub_metering_3']

        # Load the KNN model and the scaler
        loaded_model = joblib.load('smart_home/Models/Sub_metering_2/best_knn_model_sub_2.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_2/scaler_sub_2.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_1, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)

        # Predict the sub metering 1 for the next week
        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering_2": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering2Monthly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        # Define current month range
        current_date = datetime.now().date()
        start_of_month = current_date.replace(day=1)
        end_of_month = start_of_month.replace(day=1, month=start_of_month.month+1) - timedelta(days=1)

        # Convert to aware datetime objects
        start_of_month = make_aware(datetime.combine(start_of_month, datetime.min.time()))
        end_of_month = make_aware(datetime.combine(end_of_month, datetime.max.time()))

        # Fetch data for the current month
        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_month, end_of_month))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current month"}, status=status.HTTP_400_BAD_REQUEST)

        # Aggregate the data (mean, sum, etc.)
        aggregated_data = data.mean()
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']
        sub_metering_1 = aggregated_data['sub_metering_1']
        sub_metering_3 = aggregated_data['sub_metering_3']

        # Load the KNN model and the scaler
        loaded_model = joblib.load('smart_home/Models/Sub_metering_2/best_knn_model_sub_2.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_2/scaler_sub_2.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_active_power, global_intensity, sub_metering_1, sub_metering_3]],
                                  columns=['Global_active_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_3'])
        input_scaled = scaler.transform(input_data)

        # Predict the sub metering 1 for the next month
        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering_2": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering3Daily(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        # Define current day range
        current_date = datetime.now().date()
        start_of_day = make_aware(datetime.combine(current_date, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(current_date, datetime.max.time()))

        # Fetch data for the current day
        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_day, end_of_day))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user on the current day"}, status=status.HTTP_400_BAD_REQUEST)


        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']


        # Load the KNN model and the scaler
        loaded_model = joblib.load('smart_home/Models/Sub_metering_3/best_knn_model.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_3/scaler.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_active_power, global_intensity]],
                                  columns=['Global_active_power', 'Global_intensity'])
        input_scaled = scaler.transform(input_data)

        # Predict the sub metering 1 for the next day
        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering_3": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class PredictSubMetering3Weekly(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        # Define current week range (assuming week starts on Monday)
        current_date = datetime.now().date()
        start_of_week = current_date - timedelta(days=current_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)

        # Convert to aware datetime objects
        start_of_week = make_aware(datetime.combine(start_of_week, datetime.min.time()))
        end_of_week = make_aware(datetime.combine(end_of_week, datetime.max.time()))

        # Fetch data for the current week
        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_week, end_of_week))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user during the current week"}, status=status.HTTP_400_BAD_REQUEST)

        # Aggregate the data (mean, sum, etc.)
        aggregated_data = data.mean()
        print(aggregated_data)
        global_active_power = aggregated_data['global_active_power']
        global_intensity = aggregated_data['global_intensity']


        # Load the KNN model and the scaler
        loaded_model = joblib.load('smart_home/Models/Sub_metering_3/best_knn_model.pkl')
        scaler = joblib.load('smart_home/Models/Sub_metering_3/scaler.pkl')

        # Prepare the input data for prediction
        input_data = pd.DataFrame([[global_active_power, global_intensity]],
                                  columns=['Global_active_power', 'Global_intensity'])
        input_scaled = scaler.transform(input_data)

        # Predict the sub metering 1 for the next week
        prediction = loaded_model.predict(input_scaled)

        result = {
            "predicted_sub_metering_3": prediction[0]
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
            "predicted_sub_metering_3": prediction[0]
        }

        return Response(result, status=status.HTTP_200_OK)

class GenerateCurrentDayPlotForGlobalActivePower(APIView):
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


        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_day) & (data['date'] <= end_of_day)]


        hourly_data = data.groupby(data['date'].dt.hour)['global_active_power'].mean().reset_index()


        plt.figure(figsize=(10, 6))
        plt.plot(hourly_data['date'], hourly_data['global_active_power'], marker='o')
        plt.title('Average Hourly Global Active Power for the Current Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Global Active Power')
        plt.grid(True)


        plt.xticks(hourly_data['date'], hourly_data['date'])


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')

class GenerateDailyPlotForSubMetering1(APIView):
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


        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_day) & (data['date'] <= end_of_day)]
        hourly_data = data.groupby(data['date'].dt.hour)['sub_metering_1'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        plt.plot(hourly_data['date'], hourly_data['sub_metering_1'], marker='o')
        plt.title('Average Hourly for the kitchen for the Current Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Consumption by kitchen')
        plt.grid(True)


        plt.xticks(hourly_data['date'], hourly_data['date'])


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')
class GenerateDailyPlotForSubMetering2(APIView):
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


        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_day) & (data['date'] <= end_of_day)]
        hourly_data = data.groupby(data['date'].dt.hour)['sub_metering_2'].mean().reset_index()


        plt.figure(figsize=(10, 6))
        plt.plot(hourly_data['date'], hourly_data['sub_metering_2'], marker='o')
        plt.title('Average Hourly for the laundry room for the Current Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Consumption by laundry room')
        plt.grid(True)


        plt.xticks(hourly_data['date'], hourly_data['date'])


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')

class GenerateDailyPlotForSubMetering3(APIView):
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


        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_day) & (data['date'] <= end_of_day)]
        hourly_data = data.groupby(data['date'].dt.hour)['sub_metering_3'].mean().reset_index()


        plt.figure(figsize=(10, 6))
        plt.plot(hourly_data['date'], hourly_data['sub_metering_3'], marker='o')
        plt.title('Average Hourly for the living room for the Current Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Consumption by living room')
        plt.grid(True)


        plt.xticks(hourly_data['date'], hourly_data['date'])


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')
class GenerateWeeklyPlotForGlobalActivePower(APIView):
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
            return Response({"error": "No data available for the user in the current week"}, status=status.HTTP_400_BAD_REQUEST)


        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_week) & (data['date'] <= end_of_week)]


        daily_data = data.groupby(data['date'].dt.date)['global_active_power'].mean().reset_index()


        plt.figure(figsize=(10, 6))
        plt.plot(daily_data['date'], daily_data['global_active_power'], marker='o')
        plt.title('Average Daily Global Active Power for the Current Week')
        plt.xlabel('Date')
        plt.ylabel('Global Active Power')
        plt.grid(True)


        plt.xticks(daily_data['date'], daily_data['date'].apply(lambda x: x.strftime('%Y-%m-%d')), rotation=45)


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')

class GenerateWeeklyPlotForSubMetering1(APIView):
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
            return Response({"error": "No data available for the user in the current week"}, status=status.HTTP_400_BAD_REQUEST)


        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_week) & (data['date'] <= end_of_week)]


        daily_data = data.groupby(data['date'].dt.date)['sub_metering_1'].mean().reset_index()


        plt.figure(figsize=(10, 6))
        plt.plot(daily_data['date'], daily_data['sub_metering_1'], marker='o')
        plt.title('Average Daily Consumption of the kitchen for the Current Week')
        plt.xlabel('Date')
        plt.ylabel('Kitchen consumption')
        plt.grid(True)


        plt.xticks(daily_data['date'], daily_data['date'].apply(lambda x: x.strftime('%Y-%m-%d')), rotation=45)


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')
class GenerateWeeklyPlotForSubMetering2(APIView):
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
            return Response({"error": "No data available for the user in the current week"}, status=status.HTTP_400_BAD_REQUEST)


        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_week) & (data['date'] <= end_of_week)]


        daily_data = data.groupby(data['date'].dt.date)['sub_metering_2'].mean().reset_index()


        plt.figure(figsize=(10, 6))
        plt.plot(daily_data['date'], daily_data['sub_metering_2'], marker='o')
        plt.title('Average Daily Consumption of the laundry room for the Current Week')
        plt.xlabel('Date')
        plt.ylabel('Laundry room consumption')
        plt.grid(True)


        plt.xticks(daily_data['date'], daily_data['date'].apply(lambda x: x.strftime('%Y-%m-%d')), rotation=45)


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')

class GenerateWeeklyPlotForSubMetering3(APIView):
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
            return Response({"error": "No data available for the user in the current week"}, status=status.HTTP_400_BAD_REQUEST)


        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_of_week) & (data['date'] <= end_of_week)]


        daily_data = data.groupby(data['date'].dt.date)['sub_metering_3'].mean().reset_index()


        plt.figure(figsize=(10, 6))
        plt.plot(daily_data['date'], daily_data['sub_metering_3'], marker='o')
        plt.title('Average Daily Consumption of the Living room for the Current Week')
        plt.xlabel('Date')
        plt.ylabel('Living room consumption')
        plt.grid(True)


        plt.xticks(daily_data['date'], daily_data['date'].apply(lambda x: x.strftime('%Y-%m-%d')), rotation=45)


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')
class GenerateMonthlyPlotForGlobalActivePower(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

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
        daily_data = data['global_active_power'].resample('D').mean().reset_index()


        all_days = pd.date_range(start=start_of_month, end=end_of_month, freq='D')
        daily_data = daily_data.set_index('date').reindex(all_days).reset_index()
        daily_data.columns = ['date', 'global_active_power']


        daily_data['global_active_power'].fillna(0, inplace=True)


        plt.figure(figsize=(10, 6))
        plt.plot(daily_data['date'], daily_data['global_active_power'], marker='o')
        plt.title('Average Daily Global Active Power for the Current Month')
        plt.xlabel('Date')
        plt.ylabel('Global Active Power')
        plt.grid(True)


        plt.gca().set_xticks(daily_data['date'])
        plt.gca().set_xticklabels(daily_data['date'].dt.strftime('%d'))


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')

class GenerateMonthlyPlotForSubMetering1(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

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
        daily_data = data['sub_metering_1'].resample('D').mean().reset_index()


        all_days = pd.date_range(start=start_of_month, end=end_of_month, freq='D')
        daily_data = daily_data.set_index('date').reindex(all_days).reset_index()
        daily_data.columns = ['date', 'sub_metering_1']


        daily_data['sub_metering_1'].fillna(0, inplace=True)


        plt.figure(figsize=(10, 6))
        plt.plot(daily_data['date'], daily_data['sub_metering_1'], marker='o')
        plt.title('Average Daily Consumption for the kitchen for the Current Month')
        plt.xlabel('Date')
        plt.ylabel('Global Active Power')
        plt.grid(True)


        plt.gca().set_xticks(daily_data['date'])
        plt.gca().set_xticklabels(daily_data['date'].dt.strftime('%d'))


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)


        return FileResponse(buffer, content_type='image/png')

class GenerateMonthlyPlotForSubMetering2(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

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
        daily_data = data['sub_metering_2'].resample('D').mean().reset_index()


        all_days = pd.date_range(start=start_of_month, end=end_of_month, freq='D')
        daily_data = daily_data.set_index('date').reindex(all_days).reset_index()
        daily_data.columns = ['date', 'sub_metering_2']


        daily_data['sub_metering_2'].fillna(0, inplace=True)


        plt.figure(figsize=(10, 6))
        plt.plot(daily_data['date'], daily_data['sub_metering_2'], marker='o')
        plt.title('Average Daily Consumption for the Laundry Room for the Current Month')
        plt.xlabel('Date')
        plt.ylabel('Laundry Room Consumption')
        plt.grid(True)


        plt.gca().set_xticks(daily_data['date'])
        plt.gca().set_xticklabels(daily_data['date'].dt.strftime('%d'))


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        return FileResponse(buffer, content_type='image/png')

class GenerateMonthlyPlotForSubMetering3(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user


        current_date = datetime.now().date()
        start_of_month = make_aware(datetime(current_date.year, current_date.month, 1))
        end_of_month = make_aware(datetime.combine((datetime(current_date.year, current_date.month, 1) + pd.offsets.MonthEnd()).date(), datetime.max.time()))


        queryset = Home_electricity_consumption.objects.filter(user=user, date__range=(start_of_month, end_of_month))
        data = pd.DataFrame.from_records(queryset.values())

        if data.empty:
            return Response({"error": "No data available for the user in the current month"},
                            status=status.HTTP_400_BAD_REQUEST)


        data['date'] = pd.to_datetime(data['date'])


        data.set_index('date', inplace=True)
        daily_data = data['sub_metering_3'].resample('D').mean().reset_index()
        all_days = pd.date_range(start=start_of_month, end=end_of_month, freq='D')
        daily_data = daily_data.set_index('date').reindex(all_days).reset_index()
        daily_data.columns = ['date', 'sub_metering_3']

        daily_data['sub_metering_3'].fillna(0, inplace=True)

        plt.figure(figsize=(10, 6))
        plt.plot(daily_data['date'], daily_data['sub_metering_3'], marker='o')
        plt.title('Average Daily Consumption for the Living Room for the Current Month')
        plt.xlabel('Date')
        plt.ylabel('Living Room Consumption')
        plt.grid(True)


        plt.gca().set_xticks(daily_data['date'])
        plt.gca().set_xticklabels(daily_data['date'].dt.strftime('%d'))


        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        return FileResponse(buffer, content_type='image/png')


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
                0: 'bed', 1: 'chair', 2: 'couch', 3: 'dining table', 4: 'laptop', 5: 'microwave',
                6: 'oven', 7: 'refrigerator', 8: 'sink', 9: 'toaster', 10: 'toilet', 11: 'tv'
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

