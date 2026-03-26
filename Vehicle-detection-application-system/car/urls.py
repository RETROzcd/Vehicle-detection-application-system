"""car2 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    # 校内车辆管理
    path('user/list/', views.user_list),
    path('user/add/', views.user_add),
    path('user/delete/', views.user_delete),
    path('user/<int:nid>/edit/', views.user_edit),
    path('user/<int:nid>/delete/', views.user_delete),
    # 管理员管理
    # path('admin/list/', views.admin_list),
    # path("admin/add/",views.admin_add),
    #登录注销
    path('login/', views.login),
    path('logout/', views.logout),
    # 校外车辆管理
    path('ForeignVehicle/list/', views.ForeignVehicle_list),
    path('ForeignVehicle/add/',views.ForeignVehicle_add),
    path('ForeignVehicle/<int:nid>/edit/',views.ForeignVehicle_edit),
    path('ForeignVehicle/<int:nid>/delete/', views.ForeignVehicle_delete),
    path('root/info/',views.root_info),
    #警告信息
    path('violation-warning/', views.violation_warning),
    #数据统计
    path('chart/list/',views.chart_list),
    #车辆管理对话框
    path('user1/list/',views.user1_list),
    path('number/store/',views.number_store),
]
