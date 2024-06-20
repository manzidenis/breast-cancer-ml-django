from django.urls import path

from . import views

urlpatterns = [
    path("", views.demHome, name="homeFunciton"),
    path("", views.demKing, name="KingFUnction"),
]
