from rest_framework import serializers
from django.contrib.auth.models import User

from smart_home.models import Home_electricity_consumption


class CreateUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['email', 'username', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User(
            email=validated_data['email'],
            username=validated_data['username']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user

class HomeElectricityConsumptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Home_electricity_consumption
        fields = ['global_active_power', 'global_reactive_power', 'voltage',
                  'global_intensity', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3']

