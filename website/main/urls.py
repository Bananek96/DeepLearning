from django.urls import path
from . import views

app_name = 'main'
urlpatterns = [
    path('', views.GenerateSpeechView.as_view(), name='index'),
    path('voice', views.SpeechToText.as_view(), name='voice'),
]