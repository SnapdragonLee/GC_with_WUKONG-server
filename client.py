import base64
import socket


def client(send_data):
    host = '10.136.138.201'
    port = 1017
    s = socket.socket()
    s.connect((host, port))
    s.send(send_data)
    ans = s.recv(1024)
    ans = ans.decode('utf8')
    print(ans)
    s.close()
    return ans


def read_img(img_path):
    with open(img_path, 'rb') as f:
        img_data = f.read()
        base64_img = base64.b64encode(img_data)
        return base64_img


if __name__ == '__main__':
    _img_data = read_img('aaaaa.jpg')
    client(_img_data)
