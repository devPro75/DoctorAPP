from django.shortcuts import render,redirect
from django.http import HttpResponse
import patient
import json
from . import classifier
import numpy as np
from sklearn.decomposition import PCA
from django.core.exceptions import PermissionDenied
import pandas
from . import models
from django.forms import formset_factory


# Create your views here.

# def user_is_loggedin_doctor(function):
# 	def wrap(request,*args,**kwargs):
# 		if request.session['usr'] != '':
# 			if request.session['usr']['group']=='doctor':
# 				return function(request, *args, **kwargs)
# 		else:
# 			raise PermissionDenied
# 			wrap.__doc__ == function.__doc__
# 	wrap.__name__ = function.__doc__
# 	return wrap

def user_is_loggedin_doctor(function):
	def wrap(request, *args, **kwargs):
		if request.session['usr'] !='':
			if request.session['usr']['group'] == 'doctors' or request.session['usr']['group']=='admin':
				return function(request, *args, **kwargs)
		else:
			raise PermissionDenied
			wrap.__doc__ = function.__doc__
	wrap.__name__ = function.__name__
	return wrap
def user_is_loggedin_admin(function):
	def wrap(request, *args, **kwargs):
		if request.session['usr'] !='':
			if request.session['usr']['group']=='admin':
				return function(request, *args, **kwargs)
		else:
			raise PermissionDenied
			wrap.__doc__ = function.__doc__
	wrap.__name__ = function.__name__
	return wrap

@user_is_loggedin_admin
def display_delete_patients_page(request):
	template = 'doctor/patient_list_menu.html'

	if request.method == 'GET':
		exec_key = request.GET['execkey']
		print(exec_key)
		if int(exec_key) == 12345:
			_id = request.GET['id']
			print(_id)

			try:
				patient_del = patient.models.Patient.objects.get(id=_id)
				patient_del.delete()

			except Exception as e:
				pass



	# return HttpResponse("Updated")
	patients = patient.models.Patient.objects.all()
	patients = [{'id': _patient.id, 'name': _patient.name, 'email': _patient.email, 'phone': _patient.phone}
	            for _patient in patients]
	return render(request, template, {'patients': patients})

@user_is_loggedin_doctor
def display_patients(request):
	#if request.method == 'GET':
	print("inside patient")
	if 'id' in request.GET.keys():
		print("inside if id ")
		patients = patient.models.Patient.objects.get(id=request.GET['id'])
		form = patient.forms.ReadOnlyPatientForm(instance=patients)
		return render(request,'doctor/patient_info_view.html',{'form':form})
	else:
		print(" inside else in ")
		patients = patient.models.Patient.objects.all()
		patients = [{'id':_patient.id,'name':_patient.name,'email':_patient.email,'phone':_patient.phone} for _patient in patients]
		return render(request,'doctor/patient_list_menu.html',{'patients':patients})

@user_is_loggedin_doctor
def display_add_patient_page(request):
	if request.method == 'GET':
		return render(request,'doctor/patient_add_new.html', {'form':patient.forms.DoctorAddPatientForm()})

	if request.method == 'POST':
		form = patient.forms.FullDoctorAddPatientForm(request.POST)
		print(request.POST)
		if form.is_valid():
			form.save()
		return redirect('/doctor/patients/add',{'status':'saved','form':patient.forms.DoctorAddPatientForm()})

@user_is_loggedin_doctor
def display_edit_patients_page(request):
	if request.method == 'GET':
		if 'id' in request.GET.keys():
			patients = patient.models.Patient.objects.get(id=request.GET['id'])
			form = patient.forms.FullDoctorAddPatientForm(instance=patients)
			return render(request,'doctor/patient_info_edit.html',{'form':form,'id':patients.id})
	if request.method == 'POST':
		_id = request.POST['id']
		# del request.POST['id']
		form = patient.forms.FullDoctorAddPatientForm(request.POST)
		if form.is_valid():
			_object = form.save(commit=False)
			_object.id = _id
			_object.save()
			# return HttpResponse("Updated")
			return redirect('/doctor/view/patients?id='+_id)

@user_is_loggedin_doctor
def display_patient_page_menu(request):
	
	if request.method == 'GET':
		if 'select_patient' in request.GET.keys():
			form = patient.forms.PatientsChoiceForm({'select_patient':request.GET['select_patient']})
			#form_factory = formset_factory(patient.forms.PatientsChoiceForm)
			#form = form_factory(request.GET)
			bar_data = patient.models.Patient.objects.get(id=request.GET['select_patient'])
			bar_data = [bar_data.__dict__[feature] for feature in patient.forms.DoctorAddPatientForm.Meta.fields]
		
			print("Prior to Area SE",request.GET['select_patient'])
			patient_res, non, recurrence = obtain_recurrence_ratio(request.GET['select_patient'])
			res =''
			if patient_res == 0:
				res = 'Non Recurrence'
			if patient_res == 1:
				res = 'Recurrence'
			pie_data = [non, recurrence]
			return render(request,'doctor/patients_result_menu.html',{'form':form,'pie_data':pie_data,'bar_data':bar_data,'result':res,'score':classifier.score *100})
		form = patient.forms.PatientsChoiceForm()
		return render(request,'doctor/patients_result_menu.html',{'form':form})

# this function is taking and predicting, but not using pca and all if you see		
def obtain_recurrence_ratio(_id):
	patients = patient.models.Patient.objects.all()
	features_list = list()
	fields = patient.forms.DoctorAddPatientForm.Meta.fields
	recurrence =0
	non = 0
	result_list =[]
	for p in patients:
		features_list = obtain_input_from_model(p).values
		# directly predicting, not transforming to pca and col umns are also chosen mismacth, means random
		# wait ok
		print("obtain_input_from_model(p):: :: : : ", obtain_input_from_model(p))
		print("****** features shape: ", features_list.shape)
		features_list = classifier.pca.transform(features_list)
		print("After pca features shape: ", features_list.shape)
		result = classifier.model.predict(features_list)
		if result[0]  == 1:
			recurrence +=1
		if result[0] == 0:
			non += 1
	_patient = patient.models.Patient.objects.get(id=_id)
	
	features_list = obtain_input_from_model(_patient).values
	features_list = classifier.pca.transform(features_list)
	result = classifier.model.predict(features_list)
	return result[0],non,recurrence

def obtain_input_from_model(model):
	print("arease-----s", model.Area_Se)
	df = pandas.DataFrame([{
			'ID':model.id,# missing
			'Time':model.time,
			'Mean_Radius':model.mean_radius,
			'Mean_Texture':model.mean_texture,
			'Mean_Perimeter':model.mean_perimeter,
			'Mean_Area': model.mean_area,
			'Mean_Smoothness':model.mean_smoothness,
			'Mean_Compactness':model.mean_compact,
			'Mean_Concavity':model.mean_concavity,
			'Mean_Concave_points':model.mean_concave_points,
			'Mean_Symmetry':model.mean_symmetry,
			'Mean_Fractal_dimension':model.mean_fractal_dimension,
			'Radius_SE':model.radius_se,
			'Texture_SE':model.texture_se,
			'Perimeter_SE':model.perimeter_se,
			'Area_Se' :model.Area_Se,
			'Smoothness_SE':model.smoothness_se,
			'Compactness_SE':model.compactness_se,
			'Concavity_SE':model.concavity_se,
			'Concave_points_SE':model.concave_points_se,
			'Symmetry_SE':model.symmetry_se,
			'Fractal_dimension_SE':model.fractal_dimension_se,
			'Worst_Radius':model.worst_radius,
			'Worst_Texture':model.worst_texture,
			'Worst_Perimeter':model.worst_perimeter,
			'Worst_Area':model.worst_area,
			'Worst_Smoothness':model.worst_smoothness,
			'Worst_Compactness':model.worst_compactness,
			'Worst_Concavity':model.worst_concavity,
			'Worst_Concave_points':model.worst_concave_points,
			'Worst_Symmetry':model.worst_symmetry,
			'Worst_Fractal_dimension':model.worst_fractal_dimension,
			'Tumor_size':model.tumor_size,
			'Lymph_node_status':model.lymph_node_status
		}])
	return df

def display_doctor_profile_page(request):
	if request.method == 'GET':
		print(request.session['usr'])
		doctor = models.Doctor.objects.get(id=request.session['usr']['id'])
		return render(request,'doctor/doctor_profile.html',{'id':doctor.id,'mail':doctor.email,'contact':doctor.phone,'location':doctor.location})
