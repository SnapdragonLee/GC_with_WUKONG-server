import socketserver
import socket
import json
import base64
import io
from PIL import Image

MAX_REQUEST_SIZE = 2 * 1024 ** 2


class mainHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            data = self.request.recv(MAX_REQUEST_SIZE)
            if not data:
                break
            data = base64.b64decode(data)
            feedback, json_s = main_processing(data)
            with open('example.json', 'w') as ex:
                ex.write(json_s)
            print('发送信息如下')
            print(json_s)
            self.request.sendall(feedback)


def main_processing(rb_picture):
    print('图片如下')
    byte_stream = io.BytesIO(rb_picture)
    virtual_processing(byte_stream)
    ans = {
        "numbers": 1,
        "data": [
            {
                "name": "塑料袋",
                "type": "其他垃圾",
                "position": [1, 2]
            }]
    }
    return bytes(json.dumps(ans), encoding='utf-8'), \
           json.dumps(ans, indent=4, ensure_ascii=False)


async def virtual_processing(byte_stream):
    img = Image.open(byte_stream)
    img.show()


if __name__ == '__main__':
    host = '10.136.138.201'  # socket.gethostbyname(socket.gethostname())
    print(host)
    port = 1017
    server = socketserver.ThreadingTCPServer((host, port), mainHandler)
    server.serve_forever()
