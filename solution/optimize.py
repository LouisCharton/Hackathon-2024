from dataclasses import dataclass

import cv2
from scipy.ndimage import center_of_mass, distance_transform_edt, rotate
from scipy.optimize import OptimizeResult, basinhopping
import numpy as np


COST_EDGE_FACTOR = 1.5
COST_EDGE_EXP = 2.0
COST_COM_FACTOR = 0.001
COST_COM_EXP = 1.2
BASIN_TEMP = 128
BASIN_NITER = 6
BASIN_STEPSIZE = 128
BASIN_TAR = 0.5
BASIN_SF = 0.9


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


class Optimizer:
    def _create_gripper_mask(shape: list[int], gripper, x, y, alpha) -> np.ndarray:
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

    def _overlay_gripper(
        vars: np.ndarray,
        gripper: np.ndarray,
        edge_cost_map: np.ndarray,
    ) -> np.ndarray:
        gripper_mask = Optimizer._create_gripper_mask(edge_cost_map.shape, gripper, vars[0], vars[1], vars[2])

        overlayed_map = edge_cost_map.copy()

        overlayed_map[gripper_mask == 0] = 0

        return overlayed_map

    def _cost_function(
        vars: np.ndarray,
        gripper: np.ndarray,
        edge_cost_map: np.ndarray,
        com_x: int,
        com_y: int,
        cost_params: CostParams,
    ) -> float:
        edge_cost_map_overlay = Optimizer._overlay_gripper(np.floor(vars).astype(int), gripper, edge_cost_map)
        cost_edge = np.sum(edge_cost_map_overlay)
        dist = np.sqrt( (com_x - vars[0])**2 + (com_y - vars[1])**2 )
        cost_com = (cost_params.com_weight * dist) ** cost_params.com_exp

        return cost_edge + cost_com

    def optimize(
        part: np.ndarray,
        gripper: np.ndarray, 
    ) -> OptimizeResult:
        gripper_size = np.sum(gripper)
        gripper_diag = np.uint32(np.ceil(np.linalg.norm(gripper.shape, 2)))

        part_border = cv2.copyMakeBorder(part, gripper_diag, gripper_diag, gripper_diag, gripper_diag, cv2.BORDER_CONSTANT, None, 0)
        
        com_y_border, com_x_border = center_of_mass(part_border)
        dist_from_edge = distance_transform_edt(1 - part_border)
        
        cost_params = CostParams(COST_EDGE_FACTOR, COST_EDGE_EXP, COST_COM_FACTOR * gripper_size, COST_COM_FACTOR)

        edge_cost_map = (cost_params.edge_weight * dist_from_edge) ** cost_params.edge_exp
        part_shape = edge_cost_map.shape

        res = basinhopping(
            Optimizer._cost_function,
            np.array([com_x_border, com_y_border, 0]),
            minimizer_kwargs={
                "method": "Nelder-Mead",
                "args": (gripper, edge_cost_map, com_x_border, com_y_border, cost_params),
                "bounds": [(gripper_diag,part_shape[1]-gripper_diag),(gripper_diag,part_shape[0]-gripper_diag),(-1440, 1440)],
            },
            T=BASIN_TEMP,
            niter=BASIN_NITER,
            stepsize=BASIN_STEPSIZE,
            target_accept_rate=BASIN_TAR,
            stepwise_factor=BASIN_SF,
        )
        
        # remove border
        res.x[0] = res.x[0] - gripper_diag 
        res.x[1] = res.x[1] - gripper_diag
        return res


def _normalize(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def show_res(part, gripper, res):
    gripper_mask = Optimizer._create_gripper_mask(part.shape, gripper, int(res.x[0]), int(res.x[1]), res.x[2])
    res = part.copy()
    res[gripper_mask == 1] = 0.5
    com_y, com_x = center_of_mass(part)
    res[int(com_y)-5:int(com_y)+5, int(com_x)-5:int(com_x+5)] = 0.5
    cv2.imshow("Result", res)
    cv2.waitKey()


if __name__ == "__main__":
    # part = cv2.imread(f"{SAMPLE_PATH}/reference24.png")
    part = cv2.imread("REF_PART2.png")
    part = cv2.cvtColor(part, cv2.COLOR_RGB2GRAY)
    _, part = cv2.threshold(part, 1, 255, cv2.THRESH_BINARY)
    part = _normalize(part)

    gripper = cv2.imread(f"{SAMPLE_PATH}/gripper_2.png")
    gripper = cv2.cvtColor(gripper, cv2.COLOR_RGB2GRAY)
    gripper[gripper > 0] = 1

    
    best = Optimizer.optimize(part, gripper)
    show_res(part, gripper, best)

    


