**Overview:**

Pattern Recognition - using image processing library OpenCV.
The problem statement is to recognize the patterns in an image file

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


# Upload the 'screenshot.jpg' image to your Colab environment
uploaded = files.upload()

# Read the uploaded image using OpenCV
img = cv2.imread(next(iter(uploaded)), cv2.IMREAD_GRAYSCALE)

# Display the original image:
from google.colab.patches import cv2_imshow
cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

![image](https://github.com/manishbafna22/Artificial-Intelligence/assets/115042164/d80c61bc-5bd7-4364-8194-55e18c5bb91b)

# Apply Gaussian blur for noise reduction
blurred_img = cv2.GaussianBlur(img, (7, 7), 0)

# Display the OpenCV version
print("OpenCV Version:", cv2.__version__)

# Display the blurred image:
from google.colab.patches import cv2_imshow
cv2_imshow(blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

![image](https://github.com/manishbafna22/Artificial-Intelligence/assets/115042164/29374e94-538f-4e5d-8b30-46d6f0fbaf6e)


# Plot the histogram
plt.hist(img.ravel(), 256, [0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Grayscale Image')
plt.show()

![image](https://github.com/manishbafna22/Artificial-Intelligence/assets/115042164/c77d90fe-341e-46aa-a1b5-f4ed2e704171)

# Global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
 img, 0, th2,
 blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding(v=127)',
 'Original Noisy Image','Histogram',"Otsu's Thresholding",
 'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
 plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
 plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
 plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
 plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
 plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
 plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

![image](https://github.com/manishbafna22/Artificial-Intelligence/assets/115042164/a1c871cd-b883-4117-831a-9a151a0f9fa8)

def connected_component_label(path):
 # Getting the input image
 img = cv2.imread(next(iter(uploaded)), cv2.IMREAD_GRAYSCALE)
 # Converting those pixels with values 1-127 to 0 and others to 1
 img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
 # Applying cv2.connectedComponents()
 num_labels, labels = cv2.connectedComponents(img)
 # Map component labels to hue val, 0-179 is the hue range in OpenCV
 label_hue = np.uint8(179*labels/np.max(labels))
 blank_ch = 255*np.ones_like(label_hue)
 labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
 # Converting cvt to BGR
 labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
 # set bg label to black
 labeled_img[label_hue==0] = 0
 #Showing Image after Component Labeling
 plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
 plt.axis('off')
 plt.title("Image after Component Labeling")
 plt.show()
connected_component_label(img)

![image](https://github.com/manishbafna22/Artificial-Intelligence/assets/115042164/17556d38-226f-4c17-baa2-fc564838e31c)

# Pattern recognition (Original Image):
from google.colab.patches import cv2_imshow

font = cv2.FONT_HERSHEY_COMPLEX
#img = cv2.imread('Screenshot1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread(next(iter(uploaded)), cv2.IMREAD_GRAYSCALE)
_, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 4:
        cv2.putText(img, "Square", (x, y), font, 1, (255))
    else:
        cv2.putText(img, "Circle", (x, y), font, 1, (255))

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

cv2_imshow(img)

![image](https://github.com/manishbafna22/Artificial-Intelligence/assets/115042164/833d7e83-a421-4cf7-b912-0c91cbe915e3)

# Pattern recognition: Remove Salt and Pepper Noise (Filtered Image):
from google.colab.patches import cv2_imshow

font = cv2.FONT_HERSHEY_COMPLEX
img = cv2.imread(next(iter(uploaded)), cv2.IMREAD_GRAYSCALE)
# Apply Gaussian blur for noise reduction
filtered_imag = cv2.GaussianBlur(img, (7, 7), 0)

_, threshold = cv2.threshold(filtered_imag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 4:
        cv2.putText(filtered_imag, "Square", (x, y), font, 1, (255))
    else:
        cv2.putText(filtered_imag, "Circle", (x, y), font, 1, (255))

filtered_imag = cv2.cvtColor(filtered_imag, cv2.COLOR_GRAY2BGR)
cv2.drawContours(filtered_imag, contours, -1, (255, 0, 0), 2)

cv2_imshow(filtered_imag)
cv2.waitKey(0)
cv2.destroyAllWindows()

![image](https://github.com/manishbafna22/Artificial-Intelligence/assets/115042164/d1b56f73-0231-485c-9458-94a7152084e8)


**Presentation Link:**

[Pattern Recognition with OpenCV](https://docs.google.com/presentation/d/1DJ9uKgSjfBMHG-bX82cLwUOQYm6O40wSChepEhOr2Wk/edit?usp=sharing)
