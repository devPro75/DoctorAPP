from django import forms
from . import models

class RegisterForm(forms.ModelForm):
	class Meta:
		model = models.Patient
		fields = ['name', 'email', 'password', 'phone', ]

class DoctorAddPatientForm(forms.ModelForm):
	class Meta:
		model = models.Patient
		fields = ['time','concavity_se','mean_texture','mean_radius','concave_points_se',
	'symmetry_se','mean_perimeter','fractal_dimension_se','mean_area',
	'worst_radius','mean_smoothness','worst_texture','mean_compact','worst_perimeter',
	'mean_concavity','worst_area','mean_concave_points','worst_smoothness','mean_symmetry',
	'worst_compactness','radius_se','worst_concavity','mean_fractal_dimension','worst_concave_points',
	'texture_se','worst_symmetry','perimeter_se','worst_fractal_dimension','smoothness_se','tumor_size'
	,'compactness_se','lymph_node_status','Area_Se']
			

class FullDoctorAddPatientForm(forms.ModelForm):
	class Meta:
		model = models.Patient
		fields = ['name','email','phone','time','concavity_se','mean_texture','mean_radius','concave_points_se',
	'symmetry_se','mean_perimeter','fractal_dimension_se','mean_area',
	'worst_radius','mean_smoothness','worst_texture','mean_compact','worst_perimeter',
	'mean_concavity','worst_area','mean_concave_points','worst_smoothness','mean_symmetry',
	'worst_compactness','radius_se','worst_concavity','mean_fractal_dimension','worst_concave_points',
	'texture_se','worst_symmetry','perimeter_se','worst_fractal_dimension','smoothness_se','tumor_size'
	,'compactness_se','lymph_node_status','Area_Se']

class ReadOnlyPatientForm(FullDoctorAddPatientForm):
	def __init__(self, *args, **kwargs):
		super(ReadOnlyPatientForm, self).__init__(*args, **kwargs)
		instance = getattr(self, 'instance', None)
		if instance and instance.pk:
			for field in super(ReadOnlyPatientForm,self).Meta.fields:
				self.fields[field].widget.attrs['readonly'] = True


def patients():
	return [(patient.id, patient.name) for patient in models.Patient.objects.all().order_by('name')]
	#self.select_patient = forms.ChoiceField(choices=options)
	#super(PatientsChoiceForm, self).__init__(*args, **kwargs)
				
class PatientsChoiceForm(forms.Form):

	select_patient = forms.ChoiceField(choices=patients)
	
	
