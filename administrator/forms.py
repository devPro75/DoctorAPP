from django import forms
import doctor

class AddDoctorForm(forms.ModelForm):
	class Meta:
		model = doctor.models.Doctor
		fields = ['name','email','phone','location']

class ReadOnlyDoctorForm(AddDoctorForm):
	def __init__(self, *args, **kwargs):
		super(ReadOnlyDoctorForm,self).__init__(*args, **kwargs)
		instance = getattr(self, 'instance', None)
		if instance and instance.pk:
			for field in super(ReadOnlyDoctorForm,self).Meta.fields:
				self.fields[field].widget.attrs['readonly'] = True