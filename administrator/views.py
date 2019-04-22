from . import forms
import doctor
from django.shortcuts import render,redirect
from django.core.exceptions import PermissionDenied

# Create your views here.

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
def display_add_new_doctor_page(request):
	if request.method == "GET":
		if 'id' in request.GET.keys():
			form = forms.AddDoctorForm(instance=doctor.models.Doctor.objects.get(id=request.GET['id']))
			return render(request, 'admin/doctor_add_new.html',{'form':form})
		return render(request, 'admin/doctor_add_new.html', {'form':forms.AddDoctorForm()})

	if request.method == "POST":
		form = forms.AddDoctorForm(request.POST)
		if form.is_valid():
			form.save()
			return redirect('/administrator/doctor/new')

@user_is_loggedin_admin
def display_doctors_page(request):
	if request.method == 'GET':
		if 'id' in request.GET.keys():
			_doctor = doctor.models.Doctor.objects.get(id=request.GET['id'])
			# doc_info = {'id':doc.id,'name':doc.name,'mail':doc.email,'phone':doc.phone}
			form = forms.ReadOnlyDoctorForm(instance=_doctor)
			return render(request,'admin/doctor_page.html',{'form':form})

@user_is_loggedin_admin
def display_doctors(request):
	if request.method == 'GET':
		doctors = doctor.models.Doctor.objects.all()
		doc_list = [{'name':doc.name,'id':doc.id} for doc in doctors]
		return render(request, 'admin/doctor_list.html', {'doctors': doc_list})
