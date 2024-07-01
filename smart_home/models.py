
from datetime import datetime

from django.contrib.auth.models import User
from django.db import models

# Create your models here.
class Home_electricity_consumption(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField(default=datetime.now())
    global_active_power = models.FloatField(default=0)
    global_reactive_power = models.FloatField(default=0)
    voltage = models.FloatField(default=0)
    global_intensity = models.FloatField(default=0)
    sub_metering_1 = models.FloatField(default=0)
    sub_metering_2 = models.FloatField(default=0)
    sub_metering_3 = models.FloatField(default=0)

    def __str__(self):
        return str(f'The user is {self.user} and the date is {self.date}')
