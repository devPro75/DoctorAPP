from django.shortcuts import render,redirect
from django.http import HttpResponse
import doctor
import patient
import patient.classifier
import administrator
# from django.contrib.auth import authenticate,login

# Create your views here.

def display_cancer_classification_page(request):
	if request.method=='GET':
		return render(request,template_name='entry.html')

def display_homepage(request):
	if request.method=='GET':
		return render(request,template_name='homepage.html') # where is this template ?

def display_about(request):
	if request.method=='GET':
		return render(request,template_name='about.html')

def display_contact(request):
	if request.method=='GET':
		return render(request,template_name='contact.html')

def logout(request):
	request.session['usr'] = ""
	return redirect('/home')

from django.contrib.auth import login, authenticate
from django.contrib.auth import get_user_model
User = get_user_model()
	
def display_login(request):
	if request.method == 'GET':
		return render(request, template_name='login.html')

	if request.method == 'POST':
		_id = request.POST['id']
		pwd = request.POST['password']
		try :
			job = request.POST['job'].strip()
		except :
			return HttpResponse("Press back button select the appropriate role")


		if job == 'Patient':
			print("in patient")
			user = authenticate(request, username='patient', password='Matrix@2019')
		else:
			user = authenticate(request, username=_id, password=pwd)

		if user is not None:
				login(request, user)

		#username = request.POST['username']

		res = []

		# user should be the user 
		# you should add a check for the password before this
		# Please you do it, I have no idea
		# user = User.objects.get(username=_id)
		#login(request,user)
		
		print(doctor.models.Doctor.objects.all())
		print(job)
		if job == 'Doctor':
			res = doctor.models.Doctor.objects.filter(id=_id, password=pwd)

		if job == 'Admin':
			res = administrator.models.Administrator.objects.filter(id=_id,password=pwd)

		if job == 'Patient':
			res = patient.models.Patient.objects.filter(name=_id, password='123456')

		if res.exists():
			if job == 'Doctor':
				request.session['usr'] = {'group':'doctors','id':res[0].id}

				return redirect('/doctor/view/patients')
			elif job == 'Admin':
				
				request.session['usr'] = {'group':'admin','id':res[0].id}
				return redirect('/administrator/doctor/list')
			elif job == 'Patient':
				request.session['usr'] = {'group':'patients','id':res[0].id}
				return redirect('/patient/information')
		else:
			print("no res")
		return HttpResponse("Nothing done")

def display_guest_entry_page(request):
	if request.method=='GET':
		return render(request,'guest/guest_entry.html',{'form':patient.forms.DoctorAddPatientForm()})

def handle_guest_entry(request):
	if request.method == 'POST':
		import numpy as np
		form = patient.forms.DoctorAddPatientForm(request.POST)
		model = form.save(commit='false')
		
		inputs = doctor.views.obtain_input_from_model(model)
		inputs = np.array(inputs.values) + 0.
		inputs= inputs.reshape(inputs.size, 1)
		inputs = patient.classifier.pca.transform(inputs)
		
		result = patient.classifier.model.predict(inputs)
		bar_data = [model.__dict__[feature] for feature in patient.forms.DoctorAddPatientForm.Meta.fields]
		res = ''
		if result[0] == 1:
			res = 'Recurrence'
		if result[0] == 0:
			res = 'Non Recurrence'
		return render(request,'guest/guest_result.html',{'bar_data':bar_data,'result':res,'score':patient.classifier.score*100})
		
def redirect_to_guest(request):
	if request.method == 'GET':
		return redirect('/guest/entry')