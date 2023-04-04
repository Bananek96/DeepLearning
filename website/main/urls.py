from django.urls import path
from . import views

app_name = 'main'
urlpatterns = [
    path('', views.generate_speech, name='index'),
    path('tacotron2-api/', views.Tacotron2API.as_view(), name='tacotron2_api'),
]