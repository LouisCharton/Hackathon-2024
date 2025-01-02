import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import cv2

from skimage.filters import difference_of_gaussians, window, butterworth
from skimage.filters.rank import autolevel, enhance_contrast, otsu, gradient, entropy, equalize
from skimage.feature import canny
from skimage.exposure import adjust_log, rescale_intensity
from skimage.morphology import closing, disk, dilation, erosion, opening, skeletonize, reconstruction
from skimage.filters import threshold_otsu, threshold_li, gabor, hessian
# from skimage.restoration import denoise_bilateral

from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent  
sys.path.append(str(parent_dir))
from solution.helper import DATA_PATH

def processing(image: np.ndarray) -> np.ndarray:
     # F = np.zeros_like(image, dtype=np.uint16)
     # for theta in (0, 45, 90, 135):
     #      f, _ = gabor(image, frequency=0.05, theta=theta, bandwidth=3)
     #      cv2.imshow(f"gabor {theta}", f)
     #      F += f
     F = difference_of_gaussians(image, .2)
     F = rescale_intensity(F, out_range="uint8")
     F = adjust_log(F , 2)
     # F = equalize(F, disk(10))
     return F
     
if __name__ == "__main__":
     # image_path = DATA_PATH / "part_3/ mask_20241126-154623-554.png"   # Ganz i.O.
     # image_path = DATA_PATH / "part_20/mask_20241202-114431-044.png"   # Schrift ausgestanzt
     image_path = DATA_PATH / "part_22/mask_20241203-165823-809.png"   # Riesen Loch in der Mitte
     # image_path = DATA_PATH / "part_1/mask_20241203-084242-404.png"    # Beschissener Hintergrund
     image: np.ndarray = cv2.imread(image_path)
     
     SCALE = image.shape[:2]
     SCALE = SCALE[::-1]
          
     image = cv2.resize(image, (500,500))

     image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

     l, _, _ = cv2.split(image_lab)
     l = cv2.bilateralFilter(l, 10, 50, 50)
     l = enhance_contrast(l, disk(10))
     l[l==0] = l.mean()
     # l = equalize(l, disk(50))
     
     # F = processing(l)
     # F = cv2.bilateralFilter(F, 10, 100, 100)
     
     cv2.imshow("morph grad", gradient(l, disk(5)))
     
     cv2.imshow("Image", l)
     # cv2.imshow("Filtered Image", F)
     cv2.waitKey(0) 
