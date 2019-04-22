from django.urls import path
from . import views

urlpatterns = [
	path('register',views.patients_register_page),
	path('information',views.patients_view_profile),
	path('results',views.patients_show_results)
]