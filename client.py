import base64
import socket
import time

import numpy as np
from PIL import Image


def client(send_data):
    host = '10.135.189.148'
    port = 1017
    s = socket.socket()
    s.connect((host, port))
    s.send(len(send_data).to_bytes(4, 'big'))
    time.sleep(0.1)
    s.send(send_data)
    ans = s.recv(1024)
    ans = ans.decode('utf8')
    print(ans)
    s.close()
    return ans


def read_img(img_path):
    img = Image.open('1729160875.jpg')
    with open(img_path, 'rb') as f:
        img_data = f.read()
        base64_img = base64.b64encode(img_data)
        return base64_img


if __name__ == '__main__':
    _img_data = read_img("D:\\android\\Server\\test_img_predict\\origin\\img_battery_9.jpg")
    client(_img_data)
