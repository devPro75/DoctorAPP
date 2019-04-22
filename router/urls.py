from django.urls import path
from . import views

urlpatterns = [
	path('classify', views.display_cancer_classification_page),
	path('home', views.display_homepage),
	path('about', views.display_about),
	path('contact', views.display_contact),
	path('login', views.display_login),
	path('logout', views.logout),
	path('guest/entry', views.display_guest_entry_page),
	path('guest/submit', views.handle_guest_entry),
	path('guest', views.redirect_to_guest),
	path('', views.display_homepage)
]