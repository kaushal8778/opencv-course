import cv2 as cv
import numpy as np
blank = np.zeros((500,500,3),dtype='uint8')  
blank[250:252,250:252] = 255,255,255
cv.imshow('blank',blank)
cv.waitKey(0)