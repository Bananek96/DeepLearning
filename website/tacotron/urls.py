from django.urls import path
from . import views

app_name = 'tacotron'
urlpatterns = [
    path('', views.tacotron2_view, name='index'),
]
