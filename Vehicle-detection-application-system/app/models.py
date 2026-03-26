from django.db import models


# Create your models here.
class Userinfo(models.Model):
    """ 人员及车辆信息表 """
    COLLEGE_CHOICES = (
        ('ECE', '电子与计算机工程学院'),
        ('CE', '土木与交通工程学院'),
        ('EM', '经济管理学院'),
        ('ME', '机械与电气工程学院'),
        ('AAS', '建筑与艺术设计学院'),
        ('PCE', '制药与化学工程学院'),
    )
    DEFAULT_COLLEGE = '其他'
    name = models.CharField(max_length=32, verbose_name='姓名')
    gender = models.CharField(max_length=1, choices=(('M', '男'), ('F', '女')), verbose_name='性别')
    phone = models.CharField(max_length=15, verbose_name='电话')
    license_plate_number= models.CharField(max_length=15, verbose_name='车牌号')
    college = models.CharField(max_length=50, choices=COLLEGE_CHOICES,default=DEFAULT_COLLEGE, verbose_name='学院信息')
    violation_count = models.PositiveIntegerField(default=0, verbose_name='违停次数')


class Admin(models.Model):
    """管理员"""
    username = models.CharField(verbose_name="用户名", max_length=32)
    password = models.CharField(verbose_name="密码", max_length=64)


class ForeignVehicle(models.Model):
    """ 外来车辆信息表 """

    name = models.CharField(max_length=32, verbose_name='姓名')
    gender = models.CharField(max_length=1, choices=(('M', '男'), ('F', '女')), verbose_name='性别')
    phone = models.CharField(max_length=15, verbose_name='电话')
    license_plate_number = models.CharField(max_length=15, verbose_name='车牌号')
    violation_count = models.PositiveIntegerField(default=0, verbose_name='违规次数')
