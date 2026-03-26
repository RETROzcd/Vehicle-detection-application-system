from django.conf import settings
import hashlib

def md5(data_string):
    # 使用settings.SECRET_KEY作为初始输入,创建一个MD5哈希对象
    obj = hashlib.md5(settings.SECRET_KEY.encode("utf-8"))
    # 使用data_string作为输入,更新MD5哈希对象
    obj.update(data_string.encode('utf-8'))
    # 返回MD5哈希值的十六进制字符串表示
    return obj.hexdigest()