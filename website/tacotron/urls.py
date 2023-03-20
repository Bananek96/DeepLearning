from django.urls import path
from . import views

app_name = 'tacotron'
urlpatterns = [
    path('', views.generate_audio, name='index'),
]
