
import numpy as np
import cv2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206284960


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    kernel = np.flip(k_size)
    padding = kernel.size - 1
    padded_signal = np.pad(in_signal, pad_width=padding)
    convolved_signal = np.zeros(padding + in_signal.size)
    for i in range(convolved_signal.size):
        convolved_signal[i] = padded_signal[i:i + padding + 1] @ kernel
    return convolved_signal


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    kernel = np.flip(kernel)
    padding = (int(kernel.shape[0] / 2), int(kernel.shape[1] / 2))
    padded_image = np.pad(in_image, pad_width=((padding[0], padding[0]), (padding[1], padding[1])))
    convolved_image = np.zeros_like(in_image)
    for i in range(convolved_image.shape[0]):
        for j in range(convolved_image.shape[1]):
            convolved_image[i][j] = (padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel).sum()
    return convolved_image


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    v = np.array([[1, 0, -1]])
    Ix = conv2D(in_image, v)
    Iy = conv2D(in_image, v.T)
    dir = np.arctan2(Iy, Ix).astype(np.float64)
    mag = np.sqrt(Ix ** 2 + Iy ** 2)
    return dir, mag


def get_gaussian_kernel(size, sigma):
    # Compute binomial coefficients for the given size
    coeffs = np.ones(size)
    for i in range(1, size):
        coeffs[i] = coeffs[i - 1] * (size - i) / i

    # Compute the 1D Gaussian kernel
    kernel = np.zeros(size)
    for i in range(size):
        kernel[i] = coeffs[i] * np.exp(-0.5 * ((i - (size - 1) / 2) / sigma) ** 2)

    # Compute the 2D Gaussian kernel by convolving the 1D kernel with itself
    kernel_2d = np.outer(kernel, kernel)
    kernel_2d /= kernel_2d.sum()

    return kernel_2d


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gkernel = get_gaussian_kernel(k_size, 1.0)
    return conv2D(in_image, gkernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    return cv2.GaussianBlur(in_image, (k_size, k_size), 0)


def zeroCrossings(img: np.ndarray) -> np.ndarray:
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    edges_image = np.zeros_like(img)
    rows, cols = edges_image.shape[0], edges_image.shape[1]
    for i in range(rows - 1):
        for j in range(cols - 1):
            neighbors = [binary_img[i - 1, j - 1], binary_img[i - 1, j], binary_img[i - 1, j + 1],
                         binary_img[i, j - 1], binary_img[i, j + 1],
                         binary_img[i + 1, j - 1], binary_img[i + 1, j], binary_img[i + 1, j + 1]]
            if any([np.sign(binary_img[i, j]) != np.sign(neighbor) for neighbor in neighbors]):
                edges_image[i, j] = 255
    return edges_image


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    laplacian_img = cv2.Laplacian(img, cv2.CV_64F)
    return zeroCrossings(laplacian_img)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    blurred_img = blurImage2(img, 11)
    laplacian_img = cv2.Laplacian(blurred_img, cv2.CV_64F)
    return zeroCrossings(laplacian_img)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    circles = []
    candidate_circles = []
    height, width = img.shape
    accumulator = np.zeros((height, width, max_radius - min_radius + 1))
    img = cv2.GaussianBlur(img, (11, 11), 1)
    edge_image = cv2.Canny((img * 255).astype(np.uint8), 255 / 3, 255)
    edge_indices = []
    for i in range(height):
        for j in range(width):
            if edge_image[i, j] == 255:
                edge_indices.append((i, j))

    for r in range(min_radius, max_radius + 1):
        for theta in range(1, 361, 5):
            x = int(r * np.cos(theta * np.pi / 180))
            y = int(r * np.sin(theta * np.pi / 180))
            candidate_circles.append((x, y, r))

    for i, j in edge_indices:
        for x, y, r in candidate_circles:
            a = i - x
            b = j - y
            if 0 <= a < height and 0 <= b < width:
                accumulator[a, b, r - min_radius] += 1

    threshold = np.median([np.amax(accumulator[:, :, radius]) for radius in range(max_radius - min_radius + 1)])
    for i in range(height):
        for j in range(width):
            for r in range(max_radius-min_radius+1):
                if accumulator[i, j, r] > threshold:
                    circles.append((j, i, r + min_radius))

    return circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    result_img = np.zeros_like(in_image)
    padding = (int(np.floor(k_size / 2),), int(np.floor(k_size / 2),))
    padded_img = np.pad(in_image, pad_width=padding)
    kernel = cv2.getGaussianKernel(k_size, sigma_space)
    height, width = in_image.shape
    for x in range(height):
        for y in range(width):
            pivot_v = in_image[x, y]
            neighbor_hood = padded_img[x:x + k_size, y:y + k_size]
            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            combo = kernel * diff_gau
            result = (combo * neighbor_hood / combo.sum()).sum()
            result_img[x][y] = result

    return result_img, cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
