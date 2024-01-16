import tiffile as tf
from model import Model

img_tf = tf.imread("Lena_BL.tif")
model_bl = Model(img_tf)
if __name__ == "__main__":
    model_bl.extract_patches()
    model_bl.svd_decomposition()
    model_bl.iteration()
    model_bl.plot(model_bl.imD)
