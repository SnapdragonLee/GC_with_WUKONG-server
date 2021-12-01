import os
import json

# from paddle.dataset.image import load_and_transform

'''
    Before using, you have 2 ways to run this:
    1. install the env manually: (Preferred)
    
        conda install python==3.7.11
        *** Remind that pip belongs to the virtual env, which can be checked by "pip -V" ***
            
        pip install ujson opencv-python pillow tqdm PyYAML visualdl
        pip install paddleslim==1.1.1
        pip install paddlex==1.3.10
        pip install paddlelite
        pip install Augmentor
        
        conda install paddlepaddle-gpu==2.2.0 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
        *** Do not use cudatoolkit=10.2, which will cause unknown error and exception ***
        
        pip install pandas pycocotools
        
        (Optional: pip install numpy==1.19.3, when I reinstall whole environment, I found numpy==1.21.4 can work normally)
        
    2. install with "./requirements.txt" (Has not been verified):
    all the package versions are dumped into "./requirements.txt", you can try "pip install -r ./requirements.txt" 
    to install the env.
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!Please Check that You are in virtual env, or you need using "python -m pip install -r ./requirements.txt to install"!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    The image prediction can be done in n ms (n < 150).
    OK then, enjoy!
    by LD
'''

import numpy as np
import threading

# checking your environment
import cv2

from ImgDetection.ImgModel import ModelClass

# if using test_model, you need to annotate $envDir$\Lib\site-packages\paddle\fluid\executor.py line 793
model_loader = ModelClass('output/best_model')

classify_json_loader = json.load(open('ImgDetection/Classification_config.json', encoding='utf-8'))


def getImgPredict(img: os.path or np.ndarray):
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


class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self, clientsocket, recvsize=1024 * 1024, encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("Starting a Thread.....")
        try:
            msg = ''
            while True:
                rec = self._socket.recv(self._recvsize)
                # decode
                msg += rec.decode(self._encoding)

                if msg.strip().endswith('OvrImg'):
                    msg = msg[:-6]
                    break

            # re = json.loads(msg)

            res = getImgPredict(msg)
            sendmsg = json.dumps(res)

            self._socket.send(("%s" % sendmsg).encode(self._encoding))

        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            # print(identifier)

        finally:
            self._socket.close()
        print("Thread stopped.....")

    def __del__(self):
        pass


if __name__ == "__main__":
    for i in range(1, 100):
        k = getImgPredict('test_img_predict/origin/test_2.png')
        print(k)
