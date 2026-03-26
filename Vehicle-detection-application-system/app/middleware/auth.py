from django.utils.deprecation import MiddlewareMixin
from django.shortcuts import HttpResponse, redirect


class AuthMiddleware(MiddlewareMixin):

    def process_request(self, request):
        # 排除那些不需要登录就能访问的页面
        # request.path_info 获取当前用户请求的URL/login/

        if request.path_info == "/login/":
            return

        # 1、读取当前访问的用户session信息，如果能读到，说明已经登录过，就可以向后走
        info_dict = request.session.get("info")
        # print(info_dict)
        if info_dict:
            return
        # 2、没有登录过，重新回到登录界面
        return redirect("/login/")
