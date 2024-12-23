from functools import partial

import cv2
from scipy.ndimage import center_of_mass, distance_transform_edt, rotate
from scipy.optimize import minimize, OptimizeResult
import numpy as np
import matplotlib.pyplot as plt


SQRT_2 = np.sqrt(2)

# Parameters
ROT_STEP_SCALING = 2**7
TRANS_STEP_SCALING = 1

if __name__ == "__main__":
    from helper import SAMPLE_PATH
else:
    from .helper import SAMPLE_PATH


def normalize(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def create_height_maps(
    input: np.ndarray,
    com_x: float,
    com_y: float,
    weight_com: list[float] = [0.01,],
    exp_com: list[float] = [0.5,],
    weight_edge: list[float] = [0.01,],
    exp_edge: list[float] = [1.5,],
) -> np.ndarray:
    
    if not (len(weight_com) == len(exp_com) == len(weight_edge) == len(exp_edge)):
        raise IndexError("parameter lists have to be of the same length")

    dist_from_edge = distance_transform_edt(1 - input)

    cols, rows = np.indices(np.shape(input))
    dist_from_com = np.sqrt((cols - com_x) ** 2 + (rows - com_y) ** 2)

    maps = []
    for i in range(len(weight_com)):
        edge_cost = (weight_edge[i] * dist_from_edge) ** exp_edge[i]
        com_cost = (weight_com[i] * dist_from_com) ** exp_com[i]
        maps.append(edge_cost + com_cost)

    return maps



def overlay_gripper(
    vars: np.ndarray,
    gripper: np.ndarray,
    base_img: np.ndarray,
) -> np.ndarray:
    x, y, alpha = vars
    gripper_mask = np.zeros(base_img.shape)
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

    result_overlay = (
        base_img * gripper_mask
    )  # Could maybe be optimized for runtime

    return result_overlay


def target_function(
    vars: np.ndarray,
    gripper: np.ndarray,
    height_map: np.ndarray,
):
    result_overlay = overlay_gripper(np.floor(vars).astype(int), gripper, height_map)
    result_sum = np.sum(result_overlay)

    return result_sum


def optimize(
    gripper: np.ndarray, height_map: np.ndarray, com_x: int, com_y: int
) -> OptimizeResult:
    gripper_r = np.max(gripper.shape)/2
    gripper_diag = np.linalg.norm(gripper.shape, 2)
    rot_step = np.atan(1/gripper_r) * ROT_STEP_SCALING
    res = minimize(
        target_function,
        np.array([com_x, com_y, 0]),
        (gripper, height_map),
        "SLSQP",
        jac=None,
        #callback=partial(show_res, gripper, height_map),
        bounds=[(gripper_diag,height_map.shape[1]-gripper_diag),(gripper_diag,height_map.shape[0]-gripper_diag),(-180, 180)],
        options={"disp": True, "eps": (TRANS_STEP_SCALING, TRANS_STEP_SCALING, rot_step)},
    )
    return res


def show_res(gripper, height_map, res):
    fig, ax = plt.subplots(2)
    ax[0].contourf(height_map, vmin=0, vmax=10)
    out = overlay_gripper(np.floor(np.array([res[0], res[1], res[2]])).astype(int), gripper, height_map)
    ax[1].contourf(out, vmin=0, vmax=10)
    plt.show()


if __name__ == "__main__":
    part = cv2.imread(f"{SAMPLE_PATH}/reference23.png")
    part = cv2.cvtColor(part, cv2.COLOR_RGB2GRAY)
    _, part = cv2.threshold(part, 1, 255, cv2.THRESH_BINARY)
    part = normalize(part)

    gripper = cv2.imread(f"{SAMPLE_PATH}/gripper_2.png")
    gripper = cv2.cvtColor(gripper, cv2.COLOR_RGB2GRAY)
    gripper[gripper > 0] = 1

    com_x, com_y = center_of_mass(part)
    height_maps = create_height_maps(
        part, 
        com_x, 
        com_y,
        [0.01, 0.1, 0.1],
        [0.5, 0.5, 0.5],
        [0.01, 0.01, 0.01],
        [1.5, 1.75, 2],
    )
    # grad_map = np.stack(np.gradient(target_map))


    res = optimize(gripper, height_maps[0], com_x, com_y)
    show_res(gripper, height_maps[0], res.x)





##########################################################################
# ABSTELLGLEIS
##########################################################################



# def target_grad(
#     vars: np.ndarray,
#     gripper: np.ndarray,
#     target_map: np.ndarray,
#     grad_map: np.ndarray,
# ):
#     gripper_size = np.sum(gripper) / 100

#     grad_overlay = overlay_gripper(np.floor(vars).astype(int), gripper, grad_map)
#     total_force = np.sum(grad_overlay, axis=(1,2))

#     x0, y0, alpha = vars

#     x, y = np.meshgrid(np.arange(grad_overlay.shape[2]), np.arange(grad_overlay.shape[1]))
#     rx = x - x0
#     ry = y - y0

#     torque = rx * grad_overlay[0,:,:] - ry * grad_overlay[1,:,:]
#     gripper_r = np.max(gripper.shape)/2
#     total_torque = np.sum(torque) / gripper_r

#     total = np.concatenate((total_force, [total_torque]))
#     print(total)
#     return total


