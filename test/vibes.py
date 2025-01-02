import cv2
import numpy as np
import random

from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent  
sys.path.append(str(parent_dir))
from solution.helper import DATA_PATH


image_path = DATA_PATH / "part_3/mask_20241126-154623-554.png"  # Ganz i.O.
# image_path = DATA_PATH / "part_20/mask_20241202-114431-044.png"   # Schrift ausgestanzt
# image_path = DATA_PATH / "part_22/mask_20241203-165823-809.png"   # Riesen Loch in der Mitte
# image_path = DATA_PATH / "part_1/mask_20241203-084242-404.png"    # Beschissener Hintergrund
image: np.ndarray = cv2.imread(image_path)


from skimage.filters import difference_of_gaussians, window
from skimage.filters.rank import autolevel, enhance_contrast, otsu, gradient, entropy, equalize
from skimage.feature import canny
from skimage.exposure import equalize_adapthist, adjust_log, rescale_intensity
from skimage.morphology import closing, disk, dilation, erosion, opening, skeletonize
from skimage.filters import threshold_otsu, threshold_li, gabor, butterworth

# wimage = image * window('hann', image.shape)
SCALE = image.shape[:2]
SCALE = SCALE[::-1]
    
image = cv2.resize(image, (1000,1000))

image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

l, _, _ = cv2.split(image_lab)
h, _, _ = cv2.split(image_hsv)

l = cv2.bilateralFilter(l, 10, 100, 100)
# l = equalize(l, disk(40))

filtered = l.copy()
filtered = difference_of_gaussians(filtered, .1, 2)
# filtered = butterworth(filtered, .02)
filtered = rescale_intensity(filtered, out_range="uint8")
# filtered = adjust_log(filtered, 3)

# thresh = threshold_li(filtered)
# filtered = 255 * (filtered < thresh).astype(np.uint8)
# filtered = 255 * skeletonize(filtered).astype(np.uint8)

# filtered = enhance_contrast(filtered, disk(5))

# edges = 255 * canny(filtered).astype(np.uint8)
# wfiltered = filtered * window('hann', image.shape)

cv2.imshow("Windowed Image", image)
cv2.imshow("Windowed Filtered Image", filtered)
# cv2.imshow("Windowed Filtered Image EDGES", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


exit()


l = cv2.bilateralFilter(l, 10, 60, 30)
l = equalize_adapthist(l)
thresh = threshold_otsu(l)
SMALL_DARK_FEATURES = 255 * (l > thresh).astype(np.uint8)

        
filtered = difference_of_gaussians(l, .1, 4)
# filtered = otsu(filtered, disk(5))
filtered = gradient(l, disk(3))
# filtered = entropy(filtered, disk(5))

filtered = enhance_contrast(filtered, disk(20))
# filtered = adjust_log(filtered, 2)

# filtered = enhance_contrast(filtered, disk(10))
filtered = equalize_adapthist(filtered)
# filtered = closing(filtered, disk(5))

# filtered = erosion(filtered, disk(2))
filtered = rescale_intensity(filtered)
# filtered = adjust_log(filtered, 2)
thresh = threshold_otsu(filtered)
filtered = 255 * (filtered > thresh).astype(np.uint8)

EDGE_FEATURE = filtered

######  PRE-PROCESSING  ######


cv2.imshow("Image", image)
cv2.imshow("Filtered Image", filtered)
# cv2.imshow("Otsu Thresh", edges)
# cv2.imshow("Small Features", SMALL_DARK_FEATURES)

cv2.waitKey(0)
cv2.destroyAllWindows()
