import numpy as np
import cv2
from pathlib import Path
from typing import Self

from skimage.filters import difference_of_gaussians
from skimage.filters.rank import enhance_contrast
from skimage.exposure import rescale_intensity
from skimage.morphology import disk, dilation, erosion
from skimage.filters import threshold_otsu

from helper import DATA_PATH

BORDER_ROBUST: int = 50
IMAGE_RESIZE: tuple[int, int] = (1000, 1000)
GABOR_SIGMA: float = 2.
GABOR_LAMBDA_WEIGHT = .56
GABOR_N: int = 8
GABOR_KERNEL_SIZE: tuple[int, int] = (15, 15) 


def _normalize(image: np.ndarray) -> np.ndarray:
     return (image - np.min(image)) / (np.max(image) - np.min(image))     
          
class Part(object):
     image: np.ndarray
     x_ofs: int
     y_ofs: int
     scale: tuple[int, int]
     
     def __init__(self, image_path: str | Path) -> Self:
          image: np.ndarray = cv2.imread(image_path)
          image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
          l, _, _ = cv2.split(image_lab)
          image = l
          self.x_ofs, self.y_ofs, self.image = self.__align_to_boundary(image)
          self.scale = self.image.shape[:2][::-1]
     
     def detect(self) -> np.ndarray:
          part_image = self.__preprocessing(self.image)
          edges = self.__edge_detection(part_image)
          part = self.__edge_filling(part_image, edges)
          part = cv2.resize(part, self.scale)
          
          return _normalize(part)
          
     @staticmethod
     def __preprocessing(image: np.ndarray) -> np.ndarray:
          image = cv2.resize(image, IMAGE_RESIZE)
          # add robust border to avoid boundary effects
          image = cv2.copyMakeBorder(image, BORDER_ROBUST, BORDER_ROBUST, BORDER_ROBUST, BORDER_ROBUST, cv2.BORDER_CONSTANT)
          # Smooth broad features and increase local contrast to enhance high freq features
          image = cv2.bilateralFilter(image, 10, 100, 100)
          image = enhance_contrast(image, disk(5))
          # essentially high pass
          image = difference_of_gaussians(image, 1.5)
          # normalization
          image = rescale_intensity(image, out_range="uint8")
          image = enhance_contrast(image, disk(5))
          return image

     @staticmethod
     def __edge_detection(image: np.ndarray) -> np.ndarray:
          # Use 8 Gabor filters for actual edge detection
          F = np.zeros_like(image, dtype=np.uint16)
          sig = GABOR_SIGMA
          lambd = GABOR_LAMBDA_WEIGHT * sig
          N = GABOR_N
          for theta in np.linspace(0, 180 * (N-1)/N, N):
               kernel = cv2.getGaborKernel(
                    GABOR_KERNEL_SIZE, sig, theta, lambd, 3, 0, ktype=cv2.CV_32F
               )
               f = cv2.filter2D(image, cv2.CV_8UC3, kernel)
               F += f
          image = rescale_intensity(F, out_range="uint8")
          image = enhance_contrast(image, disk(5))
          # Weak morphological closing (erosion < dilation)
          image = dilation(image, disk(5))
          image = erosion(image, disk(2))
          # smooth colorspace (smooth local lighting)
          image = cv2.bilateralFilter(image, 1, 1000, 10)  
          return image

     @staticmethod
     def __edge_filling(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
          # convert to binary using otsu thresholding
          thresh = threshold_otsu(edges)
          binary = edges > thresh 
          # back to uint8
          thresholded = 255 * binary.astype(np.uint8)
          # set the border to 0, thus outer contour is not changing shape / size 
          thresholded[image == 0] = 0
          # fill the part contour
          contours, _ = cv2.findContours(
               thresholded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
          )
          contours = sorted(contours, key=cv2.contourArea)[::-1]
          for i_, contour in enumerate(contours):
               color = 255 if i_ in (0,1) else 0
               cv2.fillPoly(thresholded, [contour], color)
          # erase border
          thresholded = thresholded[
               BORDER_ROBUST:-BORDER_ROBUST, 
               BORDER_ROBUST:-BORDER_ROBUST,
          ]
          return thresholded

     @staticmethod
     def __align_to_boundary(image: np.ndarray) -> tuple[int, int, np.ndarray]:
          thresh = 0
          binary = 255 * (image > thresh).astype(np.uint8)
          contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
          # Find the bounding box of the largest contour
          largest_contour = max(contours, key=cv2.contourArea)
          x, y, w, h = cv2.boundingRect(largest_contour)
          # Crop the image using the bounding box
          image = image[y:y+h, x:x+w]
          return x, y, image

if __name__ == "__main__":
     # image_path = DATA_PATH / "part_3/mask_20241126-154623-554.png"    # Ganz i.O.
     # image_path = DATA_PATH / "part_20/mask_20241202-114431-044.png"   # Schrift ausgestanzt
     # image_path = DATA_PATH / "part_20/mask_20241202-164236-653.png"   # 
     # image_path = DATA_PATH / "part_22/mask_20241203-165823-809.png"   # Riesen Loch in der Mitte
     # image_path = DATA_PATH / "part_1/mask_20241203-084242-404.png"    # Beschissener Hintergrund
     # image_path = DATA_PATH / "part_27/mask_20241126-144214-401.png"   # Viele Löcher
     # image_path = DATA_PATH / "part_36/inverted_color/mask_20241204-113044-187i + 1.png"
     # image_path = DATA_PATH / "part_6/positional_variation/mask_20241204-112550-471i + 1.png" # Position
     image_path = "image.png"
     
     _P = Part(image_path)
     binary = _P.detect()
     print(_P.x_ofs, _P.y_ofs)
     cv2.imshow("bin", binary)
     cv2.waitKey(0) 
     cv2.imwrite("ref_part_27.png", 255*binary)
     