from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Patient(models.Model):
	name = models.CharField(max_length=255,null=False)
	phone = models.CharField(max_length=18,null=False)
	email = models.CharField(max_length=255,null=False)
	# location = models.CharField(max_length=500,null=True)
	id = models.AutoField(max_length=20,primary_key=True)
	password = models.CharField(max_length=255,default='123456')
	# patient = models.ForeignKey(Patient,on_delete=models.CASCADE)
	time = models.FloatField(null=True)
	concavity_se = models.FloatField(null=True)
	mean_radius = models.FloatField(null=True)
	concave_points_se = models.FloatField(null=True)
	mean_texture = models.FloatField(null=True)
	symmetry_se = models.FloatField(null=True)
	mean_perimeter = models.FloatField(null=True)
	fractal_dimension_se = models.FloatField(null=True)
	mean_area = models.FloatField(null=True)
	worst_radius = models.FloatField(null=True)
	mean_smoothness = models.FloatField(null=True)
	worst_texture = models.FloatField(null=True)
	mean_compact = models.FloatField(null=True)
	worst_perimeter = models.FloatField(null=True)
	mean_concavity = models.FloatField(null=True)
	worst_area = models.FloatField(null=True)
	mean_concave_points = models.FloatField(null=True)
	worst_smoothness = models.FloatField(null=True)
	mean_symmetry = models.FloatField(null=True)
	worst_compactness = models.FloatField(null=True)
	radius_se = models.FloatField(null=True)
	worst_concavity = models.FloatField(null=True)
	mean_fractal_dimension = models.FloatField(null=True)
	worst_concave_points = models.FloatField(null=True)
	texture_se = models.FloatField(null=True)
	worst_symmetry = models.FloatField(null=True)
	perimeter_se = models.FloatField(null=True)
	worst_fractal_dimension = models.FloatField(null=True)
	smoothness_se = models.FloatField(null=True)
	tumor_size = models.FloatField(null=True)
	compactness_se = models.FloatField(null=True)
	lymph_node_status = models.FloatField(null=True)
	Area_Se = models.FloatField(null=True)
