from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='chat_home'),
    path('ask/', views.ask, name='chat_ask'),
]
