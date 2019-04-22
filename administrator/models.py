from django.db import models

# Create your models here.

class Administrator(models.Model):
	name=models.CharField(max_length=255,null=True)
	email=models.CharField(max_length=255,null=True)
	location=models.CharField(max_length=255,null=True)
	id = models.AutoField(primary_key=True)
	password = models.CharField(max_length=255,null=False)
