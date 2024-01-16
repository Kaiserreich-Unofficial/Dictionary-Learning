import tiffile as tf
from model import Model
import cv2 as cv
import numpy as np

img_tf = tf.imread("LenaRGB.tif")
R = img_tf[:, :, 0]
G = img_tf[:, :, 1]
B = img_tf[:, :, 2]
model_R = Model(R)
model_G = Model(G)
model_B = Model(B)

if __name__ == "__main__":
    # RGB 图像的字典不要用 Model 自带的plot方法绘图，请使用 opencv merge 后转换成彩色空间
    # 红色通道
    print("I.处理 R 通道")
    model_R.extract_patches()
    model_R.svd_decomposition()
    model_R.iteration()
    # model_R.plot()
    # 绿色通道
    print("II.处理 G 通道")
    model_G.extract_patches()
    model_G.svd_decomposition()
    model_G.iteration()
    # model_G.plot()
    # 蓝色通道
    print("III.处理 B 通道")
    model_B.extract_patches()
    model_B.svd_decomposition()
    model_B.iteration()
    # model_B.plot()
    # 使用cv2.merge()函数将三个通道合并为一个三维矩阵
    img_mat = cv.merge((model_R.imD, model_G.imD, model_B.imD))
    model_R.plot(img_mat)
