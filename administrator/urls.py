from django.urls import path
from . import views

urlpatterns = [
	path('doctor/new',views.display_add_new_doctor_page),
	path('doctor/view',views.display_doctors_page),
	path('doctor/edit',views.display_add_new_doctor_page),
	path('doctor/list',views.display_doctors)
]