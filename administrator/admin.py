from django.contrib import admin
from .models import *
from doctor.models import Doctor
from patient.models import Patient
#import patient
# Register your models here.

admin.site.register([Administrator, Doctor, Patient])
