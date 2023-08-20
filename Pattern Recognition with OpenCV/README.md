**Overview:**
The problem statement is for Pattern Recognition - using image processing library OpenCV.
Pattern recognition is a process in which a system or algorithm identifies and classifies patterns within data. In the context of image processing, pattern recognition involves identifying specific features or objects within an image based on certain criteria. Let's break down the steps involved in pattern recognition using the given scenario:

Gray-level Image and Object Digitization:
The initial image is in gray-scale, representing the original picture with two distinct objects – a rectangle and a circle. Gray-level images represent pixel intensities with varying shades of gray, usually ranging from 0 (black) to 255 (white). The first step is to convert the image into a digital format, where each pixel's intensity is represented by a numerical value.

Binary Image after Histogram Analysis and Thresholding:
In order to simplify the image and separate the objects from the background, histogram analysis and thresholding are applied. The histogram shows the frequency distribution of pixel intensities in the image. Thresholding involves selecting a value that separates pixels into two categories – those below the threshold (background) and those above (objects). This process results in a binary image, where pixels are either 0 (background) or 1 (object).

Connectivity Analysis and Region Labeling:
After thresholding, you have a binary image with isolated objects represented as connected groups of foreground pixels (1s). Connectivity analysis is applied to identify these connected regions. One common approach is to use techniques like connected component labeling or contour tracing to segment the image into separate regions. Each region is assigned a unique label or number.

Compute Attributes and Recognize Objects:
Once regions are labeled and isolated, various attributes or features are computed for each region. These attributes might include:
Area: Number of pixels within the region.
Perimeter: Sum of the lengths of the boundary pixels.
Centroid: The center of mass of the region.
Circularity: A measure of how closely the shape resembles a circle.
After computing these attributes, the next step is to classify the objects based on the attributes. By comparing the computed attributes against predefined criteria, we can determine whether a region corresponds to a circle or a square

Convolution with a Gaussian filter is a common technique used in image processing to perform smoothing or blurring operations on an image. In the context of our scenario, we want to use a Gaussian filter to remove salt and pepper noise from a JPEG image containing objects (rectangle and circle). Here's how we can do it step by step:

Salt and Pepper Noise:
Salt and pepper noise is a type of noise that randomly turns some pixels in the image to either the maximum (salt) or minimum (pepper) pixel intensity values. This creates isolated bright and dark pixels that do not correspond to the actual image content.

Gaussian Filter:
A Gaussian filter is a type of linear filter that's used for blurring or smoothing an image. It's characterized by its Gaussian distribution, which assigns more weight to pixels closer to the center and less weight to pixels farther away. The Gaussian filter helps to average out pixel values and reduce high-frequency noise.

The convolution operation involves sliding the Gaussian filter over the image, pixel by pixel, and computing a weighted average of the pixel values under the filter. This average replaces the original pixel value, resulting in a smoother version of the image.
Applying Gaussian Filter:
To apply a Gaussian filter to your JPEG image and remove salt and pepper noise, follow these steps:
a. Choose the Filter Size: The size of the Gaussian filter (also known as the kernel size) determines the extent of blurring. A larger kernel size results in more smoothing. However, larger kernels can also cause loss of fine details. A common choice is a 3x3 or 5x5 kernel.
b. Choose the Standard Deviation (σ): The standard deviation controls the spread of the Gaussian distribution. A smaller σ leads to a narrower, sharper Gaussian curve, while a larger σ results in a broader curve and more extensive blurring.
c. Convolution Operation: For each pixel in the image, place the Gaussian filter centered on that pixel. Multiply the filter coefficients by the corresponding pixel values and sum up the results. The sum becomes the new pixel value for that location.
d. Repeat for All Pixels: Slide the filter over the entire image, pixel by pixel, applying the convolution operation at each location.

Result:
After applying the Gaussian filter, the salt and pepper noise will be reduced, and the image will appear smoother. However, some fine details might also be slightly blurred due to the filtering process.

**Implementation:**
!pip install opencv-python

from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Upload the 'screenshot.jpg' image to your Colab environment
uploaded = files.upload()

# Read the uploaded image using OpenCV
img = cv2.imread(next(iter(uploaded)), cv2.IMREAD_GRAYSCALE)

# Display the original image:
from google.colab.patches import cv2_imshow
cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://github.com/manishbafna22/Artificial-Intelligence/assets/115042164/bee1b152-63a3-4261-a814-b74cfc67fc64)







**Presentation Link:**
[Pattern Recognition with OpenCV](https://docs.google.com/presentation/d/1DJ9uKgSjfBMHG-bX82cLwUOQYm6O40wSChepEhOr2Wk/edit?usp=sharing)
