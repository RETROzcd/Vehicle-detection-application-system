from app import models
from django.db.models import Q
from django.shortcuts import render, redirect, HttpResponse
from app.utils.pagination import Pagination
from django import forms
from app.utils.encrypt import md5


def user_list(request):
    """校内车辆信息列表"""
    # for i in range(10):
    #
    #     models.Userinfo.objects.create(
    #         name='张三',
    #         gender='M',
    #         phone='123456789',
    #         license_plate_number='ABC123',
    #         college='ECE',
    #         violation_count=3
    #     )
    # data_dict = {}
    search_data = request.GET.get('q', "")
    # if search_data:
    #     data_dict["name__contains"] = search_data
    #     data_dict["license_plate_number__contains"] = search_data
    # queryset = models.Userinfo.objects.filter(**data_dict)
    condition1 = Q(name__contains=search_data)
    condition2 = Q(license_plate_number__contains=search_data)
    queryset = models.Userinfo.objects.filter(condition1 | condition2)

    page_object = Pagination(request, queryset)

    context = {"queryset": page_object.page_queryset,
               "search_data": search_data,
               "page_string": page_object.html()
               }

    return render(request, 'user_list.html', context)
def user1_list(request):
    """添加用户对话框"""
    form = UserModelForm()
    return render(request,'user1_list.html',{'form':form})

class UserModelForm(forms.ModelForm):
    class Meta:
        model = models.Userinfo
        fields = ["name", "gender", "phone", "college", "license_plate_number"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "gender": forms.Select(attrs={"class": "form-control"}),
            "phone": forms.TextInput(attrs={"class": "form-control"}),
            "college": forms.Select(attrs={"class": "form-control"}),
            "license_plate_number": forms.TextInput(attrs={"class": "form-control", })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 循环找到所有字段，并设置占位符为字段的标签
        for name, field in self.fields.items():
            # 获取字段的标签作为占位符
            label = field.label
            # 设置字段widget的占位符属性
            field.widget.attrs['placeholder'] = label


def user_add(request):
    """添加信息"""
    if request.method == "GET":
        form = UserModelForm()
        return render(request, 'user_model_form_add.html', {"form": form})
    # 用户post提交数据，必须数据校验
    form = UserModelForm(data=request.POST)
    if form.is_valid():
        form.save()
        return redirect('/user/list/')
    # 校验失败
    return render(request, 'user_model_form_add.html', {"form": form})


def user_edit(request, nid):
    """编辑信息"""
    if request.method == "GET":
        # 根据id去数据库获取要编辑的那一行数据
        row_object = models.Userinfo.objects.filter(id=nid).first()
        form = UserModelForm(instance=row_object)
        return render(request, 'user_edit.html', {"form": form})
    row_object = models.Userinfo.objects.filter(id=nid).first()
    form = UserModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        # 默认保存用户输入的所有数据
        form.save()
        return redirect('/user/list/')
    return render(request, 'user_edit.html', {"form": form})


def user_delete(request, nid):
    """删除信息"""
    models.Userinfo.objects.filter(id=nid).delete()
    return redirect('/user/list')


# def admin_list(request):
#     """管理员列表"""
#
#     queryset = models.Admin.objects.all()
#     page_object = Pagination(request, queryset)
#     context = {
#         "queryset": page_object.page_queryset,
#         "page_string": page_object.html(),
#     }
#     return render(request, 'admin_list.html', context)
#
#
# def admin_add(request):
#     """添加管理员"""
#     return render(request, 'admin_add.html')

class LoginForm(forms.Form):
    username = forms.CharField(
        label="用户名",
        widget=forms.TextInput,
        required=True
    )
    password = forms.CharField(
        label="密码",
        widget=forms.PasswordInput(render_value=True),
        required=True
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 循环ModelForm中的所有字段，给每个字段的插件设置
        for name, field in self.fields.items():
            # 字段中有属性，保留原来的属性，没有属性，才增加
            if field.widget.attrs:
                field.widget.attrs["class"] = "form-control"
                field.widget.attrs["placeholder"] = field.label
            else:
                field.widget.attrs = {
                    "class": "form-control",
                    "placeholder": field.label,
                }

    def cleaned_password(self):
        pwd = self.cleaned_data.get("password")
        return md5(pwd)


def login(request):
    """登录"""
    if request.method == 'GET':
        form = LoginForm()
        return render(request, 'login.html', {"form": form})

    form = LoginForm(data=request.POST)
    if form.is_valid():
        # 验证成功，获取到用户名和密码
        # print(form.cleaned_data)

        # 去数据库校验用户名和密码是否正确,获取用户对象、None
        admin_object = models.Admin.objects.filter(**form.cleaned_data).first()
        if not admin_object:
            form.add_error("password", "用户名或密码错误")
            return render(request, "login.html", {'form': form})
        # 用户名和密码正确
        # 网站生成随机字符串；写到用户浏览器的cookie;在写入到session中；
        request.session["info"] = {'id': admin_object.id, 'name': admin_object.username}
        return redirect("/user/list/")
    return render(request, 'login.html', {"form": form})


def logout(request):
    """注销"""

    request.session.clear()

    return redirect('/login/')


def ForeignVehicle_list(request):
    """校外车辆信息列表"""
    # for i in range(200):
    #
    # models.ForeignVehicle.objects.create(
    #     name='王五',
    #     gender='M',
    #     phone='138022222',
    #     license_plate_number='11111',
    #     violation_count=6,
    # )
    data_dict = {}
    search_data = request.GET.get('q', "")
    condition1 = Q(name__contains=search_data)
    condition2 = Q(license_plate_number__contains=search_data)
    queryset = models.ForeignVehicle.objects.filter(condition1 | condition2)

    page_object = Pagination(request, queryset)

    context = {"queryset": page_object.page_queryset,
               "search_data": search_data,
               "page_string": page_object.html()
               }
    return render(request, 'ForeignVehicle_list.html', context)


class ForeignModelForm(forms.ModelForm):
    class Meta:
        model = models.ForeignVehicle
        fields = ["name", "gender", "phone", "license_plate_number"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "gender": forms.Select(attrs={"class": "form-control"}),
            "phone": forms.TextInput(attrs={"class": "form-control"}),
            "license_plate_number": forms.TextInput(attrs={"class": "form-control", })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 循环找到所有字段，并设置占位符为字段的标签
        for name, field in self.fields.items():
            # 获取字段的标签作为占位符
            label = field.label
            # 设置字段widget的占位符属性
            field.widget.attrs['placeholder'] = label


def ForeignVehicle_add(request):
    """校外添加信息"""
    if request.method == "GET":
        form = ForeignModelForm()
        return render(request, 'ForeignVehicle_model_form_add.html', {"form": form})
    # 用户post提交数据，必须数据校验
    form = ForeignModelForm(data=request.POST)
    if form.is_valid():
        form.save()
        return redirect('/ForeignVehicle/list/')
    # 校验失败
    return render(request, 'ForeignVehicle_model_form_add.html', {"form": form})


def ForeignVehicle_edit(request, nid):
    """编辑信息"""
    if request.method == "GET":
        # 根据id去数据库获取要编辑的那一行数据
        row_object = models.ForeignVehicle.objects.filter(id=nid).first()
        form = ForeignModelForm(instance=row_object)
        return render(request, 'ForeignVehicle_edit.html', {"form": form})
    row_object = models.ForeignVehicle.objects.filter(id=nid).first()
    form = ForeignModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        # 默认保存用户输入的所有数据
        form.save()
        return redirect('/ForeignVehicle/list/')
    return render(request, 'ForeignVehicle_edit.html', {"form": form})


def ForeignVehicle_delete(request, nid):
    """删除信息"""
    models.ForeignVehicle.objects.filter(id=nid).delete()
    return redirect('/ForeignVehicle/list')


def root_info(request):
    """个人信息"""
    return render(request, 'root_info.html')


def violation_warning(request):
    """警告信息"""
    if not request.session.get('redirected', False):
        vehicles = models.Userinfo.objects.filter(violation_count__gt=3)
        context = {'vehicles': vehicles}
        request.session['redirected'] = True
        return redirect('violation-waring/')
    else:
        # 渲染页面
        vehicles = models.Userinfo.objects.filter(violation_count__gt=3)
        context = {'vehicles': vehicles}
        del request.session['redirected']  # 重定向后删除会话中的标志
        return render(request, 'violation_warning.html', context)

def chart_list(request):
    """数据统计页面"""
    return render(request,'chart_list.html')

def number_store(request):
    return render(request,'number_store.html')



