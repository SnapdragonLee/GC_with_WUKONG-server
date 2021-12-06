# RAISR

在真正的安卓测试中不需要 RAISR 算法进行去模糊处理，因为处理时间不够迅速

因此转而采用另外的思路，即在深度学习模型训练中对采集的图像进行了增强，进而增加模糊图像的识别准确率。

需要包：opencv, scipy, matplotlib, numpy (主项目中已经使用了该包，详情请见 `../requirements.txt`)

测试完效果如下：
![image](./fig2.png)
