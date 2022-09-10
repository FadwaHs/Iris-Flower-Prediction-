from django.urls import path 


from . import views # import views so we can use them in urls.

app_name='predict'


urlpatterns = [

    path('', views.predict , name ='predict'), #will call the method "index" in "views.py"
    path('predict/', views.predict_chances, name='submit_prediction'),
    path('results/', views.view_results, name='results'),
]