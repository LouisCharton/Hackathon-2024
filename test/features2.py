from helper import DATA_PATH

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import cv2

from skimage.filters import difference_of_gaussians, window
from skimage.filters.rank import autolevel, enhance_contrast, otsu, gradient, entropy, equalize
from skimage.feature import canny
from skimage.exposure import equalize_adapthist, adjust_log, rescale_intensity
from skimage.morphology import closing, disk, dilation, erosion, opening, skeletonize
from skimage.filters import threshold_otsu, threshold_li, gabor

def __optimal_grid(n: float) -> tuple[int, int]:
        sqrt_n = np.sqrt(n)
        rows = np.floor(sqrt_n)
        cols = np.ceil(sqrt_n)
        
        while rows * cols < n:
            if cols - rows > 1:
                rows += 1
            else:
                cols += 1
        return int(rows), int(cols)

def show_imgs(images: dict[str, np.ndarray]) -> None:
        n_img = len(images)
        grid_r, grid_c = __optimal_grid(n_img)
        grid = None
        for idx, (label, img) in enumerate(images.items()):
            if grid is None:
                img_h, img_w = img.shape[:2]
                grid = np.zeros((grid_r * img.shape[0], grid_c * img.shape[1], 3), dtype=np.uint8)
            
            if len(img.shape) != 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            img = cv2.putText(
                img=img, 
                text=label, 
                org=(int(img.shape[0]/15), int(img.shape[1]/5)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=.7, 
                color=(0,0,255), 
                thickness=2, 
                lineType=cv2.LINE_AA, 
                bottomLeftOrigin=False
            )
            row, col = divmod(idx, grid_c)
            grid[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w, :] = img
        
        cv2.imshow("IMAGES", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def pca_transform(image: np.ndarray, DATA: pd.DataFrame, N_COMP: int = 10) -> dict[str, np.ndarray]:
    pca = PCA(n_components=N_COMP)
    pca.fit(DATA.to_numpy())
    pca_feature_weights = pd.DataFrame(
        pca.components_, columns=DATA.columns, index=[f"PC{ii}" for ii in range(N_COMP)] 
    )
    # print(pca_feature_weights)
    pca_features = pd.DataFrame({
        col: DATA.to_numpy() @ pca_feature_weights.loc[col, :]
        for col in pca_feature_weights.index
    })
    # print(pca_features.info())
    print(f"PCA VARIANCE RATIO OF {N_COMP} Elements: {sum(pca.explained_variance_ratio_):.3f}/1")
    
    format = image.shape
    PCA_DATA = {} 
    for key, val in pca_features.items():
        X = rescale_intensity(val.to_numpy().reshape(format), out_range="uint8")
        PCA_DATA[key] = X
    return PCA_DATA


if __name__ == "__main__":
     # image_path = DATA_PATH / "part_3/ mask_20241126-154623-554.png"  # Ganz i.O.
    image_path = DATA_PATH / "part_20/mask_20241202-114431-044.png"   # Schrift ausgestanzt
    # image_path = DATA_PATH / "part_22/mask_20241203-165823-809.png"   # Riesen Loch in der Mitte
    # image_path = DATA_PATH / "part_1/mask_20241203-084242-404.png"    # Beschissener Hintergrund
    image: np.ndarray = cv2.imread(image_path)

    SCALE = image.shape[:2]
    SCALE = SCALE[::-1]
        
    image = cv2.resize(image, (1000,1000))

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    l, _, _ = cv2.split(image_lab)
    l = cv2.bilateralFilter(l, 10, 100, 100)
    l = equalize(l, disk(40))

    DATA = {}

    tmp = l.copy()
    tmp = difference_of_gaussians(tmp, .1, 2)
    tmp = rescale_intensity(tmp, out_range="uint8")
    DATA["gauss_diff"] = tmp.flatten()

    tmp = l.copy()
    tmp = canny(tmp)
    tmp = 255 * tmp.astype(np.uint8)
    DATA["canny"] = tmp.flatten()

    tmp = l.copy()
    tmp = gradient(tmp, disk(5))
    tmp = rescale_intensity(tmp, out_range="uint8")
    DATA["morph_grad"] = tmp.flatten()

    DATA = pd.DataFrame().from_dict(DATA)
    # tmp = l.copy()
    # tmp = entropy(tmp, disk(20))
    # tmp = rescale_intensity(tmp, out_range="uint8")
    # DATA["entropy"] = tmp

    # show_imgs(DATA)
    PCA_DATA = pca_transform(l, DATA, 3)
    show_imgs(PCA_DATA)
