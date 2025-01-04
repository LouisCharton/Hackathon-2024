from pathlib import Path
from argparse import ArgumentParser

from rich.progress import track
import pandas as pd
import cv2
import cairosvg
import numpy as np


if __name__ == "__main__":
    from helper import DATA_PATH 
    from part_detection import Part
    from optimize import Optimizer, show_res
else:
    from .helper import DATA_PATH
    from .part_detection import Part
    from .optimize import Optimizer, show_res

def optimal_gripper_solution(
    part_image_path: Path, gripper_image_path: Path
) -> tuple[float, float, float]:
    """Compute the solution for the given part and gripper images.

    :param part_image_path: Path to the part image
    :param gripper_image_path: Path to the gripper image
    :return: The x, y and angle of the gripper
    """
    _PART = Part(part_image_path)
    DETECTED = _PART.detect()
    X_OFS = _PART.x_ofs
    Y_OFS = _PART.y_ofs
    
    if gripper_image_path.suffix == ".svg":
        buf = cairosvg.svg2png(url=str(gripper_image_path), dpi=25.4, scale=1)
        png_array = np.frombuffer(buf, dtype=np.uint8)
        GRIPPER = cv2.imdecode(png_array, cv2.IMREAD_UNCHANGED)
    else:
        GRIPPER = cv2.imread(gripper_image_path)
        
    GRIPPER = cv2.cvtColor(GRIPPER, cv2.COLOR_BGR2GRAY)
    GRIPPER[GRIPPER > 0] = 1

    best = Optimizer.optimize(DETECTED, GRIPPER)
    X, Y, ALPHA = best.x    
    
    # DETECTED = cv2.copyMakeBorder(DETECTED, 100, 100, 100, 100, cv2.BORDER_CONSTANT)
    
    # best.x[0] += 100
    # best.x[1] += 100
    # show_res(DETECTEÖD, GRIPPER, best)
    return X+X_OFS, Y+Y_OFS, ALPHA


def main():
    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    # read the input csv file
    input_df = pd.read_csv(args.input)

    # compute the solution for each row
    results = []
    for _, row in track(
        input_df.iterrows(),
        description="Computing the solutions for each row",
        total=len(input_df),
    ):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        x, y, angle = optimal_gripper_solution(part_image_path, gripper_image_path)
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle])

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
