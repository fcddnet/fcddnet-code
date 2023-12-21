import cv2
import numpy as np

#读取图像
img = cv2.imread(r'F:\DeepLearning\data\VOC+NET\test2\1/ILSVRC2012_val_00023685.JPEG')

#生成秘钥
key = np.random.randint(0,256,img.shape,dtype=np.uint8)

#加密
secret = cv2.bitwise_xor(img,key)

#解密
truth = cv2.bitwise_xor(secret,key)

#显示图像
cv2.imshow('secret',secret)
cv2.imshow('truth',truth)
cv2.imwrite('jiami.jpeg',secret)
cv2.waitKey(0)