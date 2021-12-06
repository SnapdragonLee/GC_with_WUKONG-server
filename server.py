import socketserver
import socket
import json
import base64
import io
from PIL import Image, ImageFile
from ImgDetection.ImgModel import ModelClass
import numpy as np
import os
import cv2

MAX_REQUEST_SIZE = 4 * 1024 ** 2
model_loader = ModelClass('output/best_model')
classify_json_loader = json.load(open('ImgDetection/Classification_config.json', encoding='utf-8'))
ImageFile.LOAD_TRUNCATED_IMAGES = True


class mainHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            data = self.request.recv(4)
            img_size = int.from_bytes(data, 'big')
            print(img_size)
            data = bytes()
            less = img_size
            while less > 0:
                print("接收中")
                data = data + self.request.recv(img_size)
                less = img_size - len(data)
            if not data:
                break
            data = base64.b64decode(data)
            feedback, json_s = main_processing(data)
            print('发送信息如下')
            print(json_s)
            self.request.sendall(feedback)


def main_processing(rb_picture):
    byte_stream = io.BytesIO(rb_picture)

    ans = virtual_processing(byte_stream)
    """
    样例
    {
        "numbers": 1,
        "data": [
            {
                "name": "塑料袋",
                "type": "其他垃圾",
                "position": [1, 2]
            }]
    }
    """
    return bytes(json.dumps(ans, ensure_ascii=False), encoding='utf-8'), \
           json.dumps(ans, indent=4, ensure_ascii=False)


def PILImageToCV(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


def virtual_processing(byte_stream):
    img = Image.open(byte_stream)
    # img.show()
    print('get')
    img = PILImageToCV(img)
    return getImgPredict(img)


def getImgPredict(img: np.ndarray):
    modelPr = model_loader.prediction(img)
    # print(modelPr)
    json_Data = dict()
    if modelPr is None:
        json_Data['numbers'] = 0
    else:
        json_Data['numbers'] = len(modelPr)
        data_util = list()
        for key in modelPr:
            namedType = classify_json_loader.get(str(key)).split('_')
            attribute = dict()
            attribute['name'] = namedType[1]
            attribute['type'] = namedType[0]
            data_util.append(attribute)
        json_Data['data'] = data_util
    return json_Data


if __name__ == '__main__':
    host = socket.gethostbyname(socket.gethostname())  # '10.135.14.7'  #
    print(host)
    port = 1017
    server = socketserver.ThreadingTCPServer((host, port), mainHandler)
    server.serve_forever()
