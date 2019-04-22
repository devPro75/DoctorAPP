from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Doctor(models.Model):
	name = models.CharField(max_length=255,null=False)
	phone = models.CharField(max_length=18,null=False)
	email = models.CharField(max_length=255,null=False)
	location = models.CharField(max_length=500,null=False)
	id = models.CharField(max_length=20,null=False,primary_key=True)
	password = models.CharField(max_length=255,null=False)
	user = models.ForeignKey(User, null=True, blank=True, on_delete="cascade")

