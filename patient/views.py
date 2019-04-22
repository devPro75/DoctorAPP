from django.core.exceptions import PermissionDenied
from django.shortcuts import render
from . import forms
from . import models
from . import classifier
import doctor

# Create your views here.

def user_is_loggedin_patient(function):
	def wrap(request, *args, **kwargs):
		if request.session['usr'] !='':
			if request.session['usr']['group'] == 'patients' or request.session['usr']['group']=='admin':
				return function(request, *args, **kwargs)
		else:
			raise PermissionDenied
			wrap.__doc__ = function.__doc__
	wrap.__name__ = function.__name__
	return wrap

@user_is_loggedin_patient
def patients_register_page(request):
	if request.method == 'GET':
		return render(request, 'patient/register.html', {'form':forms.RegisterForm()})

	if request.method == 'POST':
		form = forms.RegisterForm(request.POST)
		if form.is_valid():
			form.save()

@user_is_loggedin_patient
def patients_view_profile(request):
	if request.method == 'GET':
		patient = models.Patient.objects.get(id=request.session['usr']['id'])
		return render(request,'patient/patient_profile.html',{'id':patient.id,'mail':patient.email,'phone':patient.phone})

@user_is_loggedin_patient
def patients_show_results(request):
	if request.method == 'GET':
		_patient = models.Patient.objects.get(id=request.session['usr']['id'])
		#_patient = models.Patient.objects.get(id=request.GET['select_patient'])
		bar_data = [_patient.__dict__[feature] for feature in forms.DoctorAddPatientForm.Meta.fields]
		input = doctor.views.obtain_input_from_model(_patient)
		print("Input shape: ", len(input.values[0]))
		import numpy as np
		input = np.array(input.values) + 0.
		input= input.reshape(input.size, 1)
		#print("there is your missing value", np.where(np.isnan(X))) # comment it
		temp_df = classifier.pca.transform(input)
		print("After applying pca")
		result = classifier.model.predict(temp_df)
		if result[0] == 1:
			res = 'Recurrence'
		if result[0] == 0:
			res = 'Non Recurrence' 
		return render(request,'patient/patients_result_menu.html',{'bar_data':bar_data,'result':res,'score':classifier.score*100})


