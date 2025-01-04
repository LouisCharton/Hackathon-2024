import pandas as pd
import numpy as np
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
     F = np.zeros_like(image, dtype=np.uint16)
     sig = 2
     lambd = .56 * sig
     N = 8
     for theta in np.linspace(0, 180 * (N-1)/N, N):
          kernel = cv2.getGaborKernel((15, 15), sig, theta, lambd, 3, 0, ktype=cv2.CV_32F)
          f = cv2.filter2D(image, cv2.CV_8UC3, kernel)
          # cv2.imshow(f"gabor {theta}", f)
          F += f
     F = rescale_intensity(F, out_range="uint8")
     return F

def fill_edge_profile(edge: np.ndarray) -> np.ndarray:
     # _, edge = cv2.threshold(edge, 200, 255, cv2.THRESH_BINARY)        
     contours, _ = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

     contours = sorted(contours, key=cv2.contourArea)[::-1]
     for i_, contour in enumerate(contours):
          color = 255 if i_ in (0,1) else 0
          # cv2.drawContours(edge, [contour], -1, color=color, thickness=cv2.FILLED)
          cv2.fillPoly(edge, [contour], color)
     return edge
     
     
if __name__ == "__main__":
     image_path = DATA_PATH / "part_3/mask_20241126-154623-554.png"    # Ganz i.O.
     # image_path = DATA_PATH / "part_20/mask_20241202-114431-044.png"   # Schrift ausgestanzt
     # image_path = DATA_PATH / "part_20/mask_20241202-164236-653.png"   # 
     # image_path = DATA_PATH / "part_22/mask_20241203-165823-809.png"   # Riesen Loch in der Mitte
     # image_path = DATA_PATH / "part_1/mask_20241203-084242-404.png"    # Beschissener Hintergrund
     # image_path = DATA_PATH / "part_27/mask_20241126-144214-401.png"   # Viele Löcher
     save_name = "ref_part_3.png"
     image: np.ndarray = cv2.imread(image_path)
     SCALE = image.shape[:2][::-1]
     image = cv2.resize(image, (1000,1000))
     image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
     l, _, _ = cv2.split(image_lab)
     
     l = cv2.copyMakeBorder(l, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
     l = cv2.bilateralFilter(l, 10, 100, 100)
     l = enhance_contrast(l, disk(5))
     l = difference_of_gaussians(l, 1.5)
     
     F = l.copy()
     F = rescale_intensity(F, out_range="uint8")
     F = enhance_contrast(F, disk(5))
     F = processing(F)
     F = enhance_contrast(F, disk(5))
     F = dilation(F, disk(5))
     F = erosion(F, disk(2))
     F = cv2.bilateralFilter(F, 1, 1000, 10)
     
     thresh = threshold_otsu(F)
     binary = F > thresh 
     binary = 255 * binary.astype(np.uint8)
     binary[l == 0] = 0
     binary = fill_edge_profile(binary)
     
     binary = binary[
          50:-50, 
          50:-50,
     ]
     # binary = closing(binary, disk(10))
     binary = cv2.resize(binary, SCALE)
     # cv2.imshow("morph grad", F)
     cv2.imshow("bin", binary)
     cv2.waitKey(0) 

     cv2.imwrite(save_name, binary)
     