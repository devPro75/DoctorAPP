from django.urls import path
from . import views

urlpatterns = [
	path('view/patients', views.display_patients),
	path('patients/add', views.display_add_patient_page),
	path('patients/edit', views.display_edit_patients_page),
	path('patients/delete', views.display_delete_patients_page),
	path('patients/predict', views.display_patient_page_menu),
	path('profile', views.display_doctor_profile_page)
]