import numpy as np
import cv2

#读取图片
img = cv2.imread('city.jpg')
#二值化，canny检测
binaryImg = cv2.Canny(img,50,200)

#寻找轮廓
#也可以这么写：
#binary,contours, hierarchy = cv2.findContours(binaryImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#这样，可以直接用contours表示
h = cv2.findContours(binaryImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#提取轮廓
contours = h[0]
#打印返回值，这是一个元组
print(type(h))
#打印轮廓类型，这是个列表
print(type(h[1]))
#查看轮廓数量
print (len(contours))

#创建白色幕布
temp = np.ones(binaryImg.shape,np.uint8)*255
#画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
cv2.drawContours(temp,contours,-1,(0,255,0),3)

cv2.imshow("contours",temp)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread('test.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('imageshow', image)  # 显示返回值image，其实与输入参数的thresh原图没啥区别
cv2.waitKey(0)

img = cv2.drawContours(img, contours, -1, (0, 255, 0), 5)  # img为三通道才能显示轮廓
cv2.imshow('drawimg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()