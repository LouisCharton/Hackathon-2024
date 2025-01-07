# Hackathon 2024 - Submission of Group *HEINERCV*

Team members:
- Robert Knobloch 
- Louis Charton 
    
## Description
The solution approach is 100% knowledge based, as the data seemed to unreliable (noisy, bad resolution) for pretrained models
-> e.g. **HED** (Hollistically Nested Edge Detection) (Xie et al., 2015) https://github.com/s9xie/hed

- For part detection, heavy preprocessing was needed for any edge detecting algorithm to work. We resulted in applying a pipeline consisting of a constant rescaling of any image given; Bilateral Filtering to reduce noise and reduce overall very high frequency noise. A chain of Rank filters for contrast enhancement, low frequency Cutting using a difference of gaussians Filter to slightly enhance high frequency content (e.g. edges) proved to be a robust preprocessing pipeline, considering the quality of provided data.
- The edge Detection mainly consists in filtering with 8 Gabor Kernels arranged in a semicircle, tuned for high frequencies. The parameters are kept constant due to the rescaling of the image. 
- After that, a series of morphological filters are applied to close gaps and remove small filter noise artefacts. The image is thresholded by using otsus method and the resulting binary contour is filled and used as a mask for the gripper.
- For the position optimization, the gripper mask is converted to png and functions as a integral mask for our costfunction. the value of this spatial integral is to be minimized. The cost function is a linear weighted overlay of:
    - The euclidean distance of the gripper center to the center of gravity of the part mask of order N
    - The distance of any point to the nearest point on the Part (Weight for Holes) of order N
    
    Repeating this optimization for different weights and different orders of the cost function parts, in addition to random starting coordinates (basin hopping) yields a nearly optimal result for the identified partmask provided.

The result of the part detection algorithm is non-stochastic and the optimizer yields nearly optimal results all while using a stochastic optimization approach (basinhopping)   

## How to Run
After installing all requirements according to requirements.txt run below: 

`python solution/main.py path/to/input/tasks.csv output/solutions.csv` 

## ... and other things you want to tell us
- The rather complex part detection is necessary, because traditional algorithms like canny have dramatically unreliable results with the provided data.
- The challenges were mainly the stability of the optimizer, poor computation performance (high cost) and the reliability of the part detection. 
- We did not want to create labeled Data (too much and too various) and pretrained DNN approaches like HED (see above) did not work properly for thresholding in most cases. We found no clear identifiers for a clustering approach, even after performing a PCA of hundereds of features making use of what felt like the whole scikit-image library.
- Thus we decided on the knowledge based approach which serves as a compromise, yielding decent results for the most images, even including noisy backgrounds, poor contrast, high noise etc.   