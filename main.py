#coding:utf-8

from PIL import Image
import numpy as np

img = Image.open('images/dog.jpg', 'r')
resize_img = img.resize((224, 224))
print np.asarray(resize_img).shape
