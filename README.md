## Server分支

本分支为服务端代码，包括 `server.py` 服务端入口，`client.py` 客户端 DEMO。

`ImgDetection/` 为 PaddleX 识别代码，比 yolov 不知阴间多少。

`RAISR/` 为 RAISR 算法增强器。

`VoiceRec/` 为词语库，语音识别时用。

`output/` 为训练模型蒸馏输出，有 test 与 best 两个 model。

`test_img_predict/origin` 为预测图片测试样例。

`example.json` 为服务端与模型识别器交互格式。



### Startup

`pip install -r requirements.txt`

