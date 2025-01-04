from dataclasses import dataclass
from operator import attrgetter

import cv2
from scipy.ndimage import center_of_mass, distance_transform_edt, rotate
from scipy.optimize import minimize, OptimizeResult, basinhopping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

SQRT_2 = np.sqrt(2)

if __name__ == "__main__":
    from helper import SAMPLE_PATH, DATA_PATH
else:
    from .helper import SAMPLE_PATH, DATA_PATH

@dataclass
class CostParams:
    edge_weight: float
    edge_exp: float
    com_weight: float
    com_exp: float


def normalize(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def create_gripper_mask(shape: list[int], gripper, x, y, alpha) -> np.ndarray:
    gripper_mask = np.zeros(shape)
    gripper = gripper.astype(bool)
    gripper = rotate(gripper, alpha, reshape=True, order=0)
    gripper_h, gripper_w = gripper.shape
    gripper_cx, gripper_cy = gripper_w // 2, gripper_h // 2
    if gripper_mask.ndim == 2:
        gripper_mask[
            y - gripper_cy : y - gripper_cy + gripper_h,
            x - gripper_cx : x - gripper_cx + gripper_w,
        ] = gripper
    elif gripper_mask.ndim == 3:
        gripper_mask[
            :,
            y - gripper_cy : y - gripper_cy + gripper_h,
            x - gripper_cx : x - gripper_cx + gripper_w,
        ] = gripper
    else:
        raise NotImplemented()
    
    return gripper_mask


def overlay_gripper(
    vars: np.ndarray,
    gripper: np.ndarray,
    edge_cost_map: np.ndarray,
) -> np.ndarray:
    gripper_mask = create_gripper_mask(edge_cost_map.shape, gripper, vars[0], vars[1], vars[2])

    overlayed_map = edge_cost_map.copy()

    overlayed_map[gripper_mask == 0] = 0

    return overlayed_map


def target_function(
    vars: np.ndarray,
    gripper: np.ndarray,
    edge_cost_map: np.ndarray,
    com_x: int,
    com_y: int,
    cost_params: CostParams,
) -> float:
    edge_cost_map_overlay = overlay_gripper(np.floor(vars).astype(int), gripper, edge_cost_map)
    cost_edge = np.sum(edge_cost_map_overlay)
    dist = np.sqrt( (com_x - vars[0])**2 + (com_y - vars[1])**2 )
    cost_com = (cost_params.com_weight * dist) ** cost_params.com_exp

    return cost_edge + cost_com


def optimize(
    gripper: np.ndarray, 
    gripper_diag: int, 
    com_x: int, 
    com_y: int, 
    dist_from_edge: np.ndarray,
    cost_params: CostParams
) -> OptimizeResult:

    edge_cost_map = (cost_params.edge_weight * dist_from_edge) ** cost_params.edge_exp
    part_shape = edge_cost_map.shape

    res = basinhopping(
        target_function,
        np.array([com_x, com_y, 0]),
        minimizer_kwargs={
            "method": "Nelder-Mead",
            "args": (gripper, edge_cost_map, com_x, com_y, cost_params),
            "bounds": [(gripper_diag,part_shape[1]-gripper_diag),(gripper_diag,part_shape[0]-gripper_diag),(-1440, 1440)],
        },
        T=128,
        niter=6,
        stepsize=128,
        target_accept_rate=0.5,
        stepwise_factor=0.9,
    )
    
    return res


def show_res(gripper, height_map, res):
    fig, ax = plt.subplots(2)
    x, y, alpha = res
    gripper_mask = create_gripper_mask(height_map.shape, gripper, int(x), int(y), alpha)

    print(np.max(height_map))
    ax[0].contourf(height_map)
    ax[1].contourf(gripper_mask)
    plt.show()


if __name__ == "__main__":
    # part = cv2.imread(f"{SAMPLE_PATH}/reference24.png")
    part = cv2.imread("ref_part_27.png")
    part = cv2.cvtColor(part, cv2.COLOR_RGB2GRAY)
    _, part = cv2.threshold(part, 1, 255, cv2.THRESH_BINARY)
    part = normalize(part)

    gripper = cv2.imread(f"{SAMPLE_PATH}/gripper_2.png")
    gripper = cv2.cvtColor(gripper, cv2.COLOR_RGB2GRAY)
    gripper[gripper > 0] = 1

    gripper_size = np.sum(gripper)
    gripper_r = np.max(gripper.shape)/2
    gripper_diag = np.uint32(np.ceil(np.linalg.norm(gripper.shape, 2)))
    part_border = cv2.copyMakeBorder(part, gripper_diag, gripper_diag, gripper_diag, gripper_diag, cv2.BORDER_CONSTANT, None, 0)
    
    com_y, com_x = center_of_mass(part_border)
    dist_from_edge = distance_transform_edt(1 - part_border)
    
    cost_params = CostParams(1.5, 2.0, 0.001 * gripper_size, 1.2)
    best = optimize(gripper, gripper_diag, com_x, com_y, dist_from_edge, cost_params)

    x = int(best.x[0])
    y = int(best.x[1])
    alpha = best.x[2]

    gripper_mask = create_gripper_mask(part_border.shape, gripper, x, y, alpha)
    res = part_border.copy()
    res[gripper_mask == 1] = 0.5
    res[int(com_y)-5:int(com_y)+5, int(com_x)-5:int(com_x+5)] = 0.5
    cv2.imshow("Result", res)
    cv2.waitKey()


