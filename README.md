# Ex2-Convolution-Edge-Detection

## Introduction
This is a collection of functions for image processing in Python. It includes functions for convolving 1D and 2D signals with a given kernel, calculating gradients of an image, blurring images using a Gaussian kernel, edge detection, hough circles and bilateral filter.

## Dependencies
Python 3.x
OpenCV
Numpy

## Functions

#### 1D and 2D Convolution
- `conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray`: Convolve a 1-D array with a given kernel.
- `conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray`: Convolve a 2-D array with a given kernel.
- `convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray)`: Calculate gradient of an image.

#### Gaussian Filter
- `get_gaussian_kernel(size, sigma)`: Compute a Gaussian kernel of a given size and sigma value.
- `blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray`: Blur an image using a Gaussian kernel.
- `blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray`: Blur an image using a Gaussian kernel using OpenCV built-in functions.

#### Edge Detection
- `edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray`: Detect edges using the "ZeroCrossing" method.
- `edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray`: Detect edges using the "ZeroCrossingLOG" method.

#### Feature Extraction
- `houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:`: The Hough Circle Transform is a feature extraction technique used in image analysis and computer vision to detect circular objects in an image. It works by mapping the edge points of an image to a 2D array, where each cell in the array represents a potential circle center. By applying a voting procedure, the algorithm can identify the most likely circle parameters.

#### Non-Linear Filtering
- `bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):`: The Bilateral Filter is a non-linear filter used to smooth images while preserving their edges. It works by taking a weighted average of the pixels in the image, where the weights depend on the difference in intensity between neighboring pixels. By using a Gaussian function to compute the weights, the Bilateral Filter can smooth an image while preserving the edges.

## Examples
#### Edge Detection
![image](https://github.com/lior2k/Ex2-Convolution-Edge-Detection/assets/92747945/5c0000d6-63e2-490b-9560-06f1ca8fdc5e)

#### Feature Extraction
![image](https://github.com/lior2k/Ex2-Convolution-Edge-Detection/assets/92747945/9d21aa2d-8154-45e4-87f6-10075eb66f4f)
