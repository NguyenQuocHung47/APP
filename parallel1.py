import cv2
from numba import cuda
import math

# HSV to RGB conversion mising (?)

def create_gaussian_kernel(kernel_size, sigma):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


@cuda.jit
def apply_gaussian_blur_to_channel_kernel(channel, kernel, result, kernel_size):
    i, j = cuda.grid(2)
    if i < channel.shape[0] and j < channel.shape[1]:
        pad = kernel_size // 2
        value = 0.0
        for ki in range(-pad, pad + 1):
            for kj in range(-pad, pad + 1):
                ni = min(max(i + ki, 0), channel.shape[0] - 1)
                nj = min(max(j + kj, 0), channel.shape[1] - 1)
                value += channel[ni, nj] * kernel[ki + pad, kj + pad]
        result[i, j] = value


@cuda.jit
def detect_edges_kernel(channel, edges, threshold):
    i, j = cuda.grid(2)
    if 1 <= i < channel.shape[0] and 1 <= j < channel.shape[1]:
        if (abs(channel[i, j] - channel[i-1, j]) > threshold) or (abs(channel[i, j] - channel[i, j-1]) > threshold):
            edges[i, j] = 1


@cuda.jit
def low_pass_filter_kernel(edges, filtered_edges, neighbor_threshold):
    i, j = cuda.grid(2)
    if 1 <= i < edges.shape[0]-1 and 1 <= j < edges.shape[1]-1:
        if edges[i, j] == 1:
            neighbor_count = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if not (di == 0 and dj == 0):
                        neighbor_count += edges[i+di, j+dj]
            if neighbor_count < neighbor_threshold:
                filtered_edges[i, j] = 0
            else:
                filtered_edges[i, j] = 1


@cuda.jit
def calculate_additive_magnitude_kernel(value_channel, result):
    # Only use 1 thread to compute
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        flat_values = value_channel.reshape(-1)
        max_v = np.max(flat_values)
        min_v = np.min(flat_values)
        avg_v = np.mean(flat_values)
        mid_v = (max_v + min_v) / 2.0
        result[0] = (max_v / 8.0) * (avg_v / mid_v)


@cuda.jit
def sharpen_edges_kernel(hsv_image, edges, additive_magnitude, scale, output_hsv):
    i, j = cuda.grid(2)
    if 1 <= i < hsv_image.shape[0]-1 and 1 <= j < hsv_image.shape[1]-1:
        if edges[i, j] == 1:
            local_sum = 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    local_sum += hsv_image[i+di, j+dj, 2]
            local_mean = local_sum / 9.0
            v = hsv_image[i, j, 2]
            delta = 0
            if local_mean > v:
                delta = -scale * additive_magnitude * (v / local_mean)
            else:
                delta = scale * additive_magnitude * (local_mean / v)
            new_v = v + delta
            output_hsv[i, j, 2] = min(max(new_v, 0.0), 1.0)


@cuda.jit
def hsv_image_to_rgb_batch_kernel(hsv_image, result):
    i, j = cuda.grid(2)
    if i < hsv_image.shape[0] and j < hsv_image.shape[1]:
        h, s, v = hsv_image[i, j, 0], hsv_image[i, j, 1], hsv_image[i, j, 2]
        h = h * 360.0
        c = v * s
        x = c * (1 - abs(((h / 60.0) % 2) - 1))
        m = v - c

        r, g, b = 0.0, 0.0, 0.0
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        elif 300 <= h < 360:
            r, g, b = c, 0, x

        result[i, j, 0] = (r + m) * 255.0
        result[i, j, 1] = (g + m) * 255.0
        result[i, j, 2] = (b + m) * 255.0

# Main function
def process_image_on_gpu(input_path, output_path, edge_threshold=0.1, neighbor_threshold=2, scale=0.5):
    img = cv2.imread(input_path)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

    height, width = hsv_image.shape[:2]

    value_channel = hsv_image[:, :, 2]

    kernel_size = 5
    sigma = 1.0
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # Allocate memory on device
    d_value_channel = cuda.to_device(value_channel)
    d_blurred_channel = cuda.device_array_like(value_channel)
    d_edges = cuda.device_array_like(value_channel).astype(np.int32)
    d_filtered_edges = cuda.device_array_like(value_channel).astype(np.int32)
    d_hsv_image = cuda.to_device(hsv_image)
    d_output_hsv = cuda.device_array_like(hsv_image)
    d_rgb_image = cuda.device_array_like(img)
    d_kernel = cuda.to_device(kernel)
    d_additive_magnitude = cuda.device_array(1, dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid = (math.ceil(width / threadsperblock[0]), math.ceil(height / threadsperblock[1]))

    # Apply Gaussian blur
    apply_gaussian_blur_to_channel_kernel[blockspergrid, threadsperblock](d_value_channel, d_kernel, d_blurred_channel, kernel_size)

    # Detect edges
    detect_edges_kernel[blockspergrid, threadsperblock](d_value_channel, d_edges, edge_threshold)

    # Filter weak edges
    low_pass_filter_kernel[blockspergrid, threadsperblock](d_edges, d_filtered_edges, neighbor_threshold)

    # Calculate additive magnitude
    calculate_additive_magnitude_kernel[1, 1](d_value_channel, d_additive_magnitude)

    # Sharpen edges
    sharpen_edges_kernel[blockspergrid, threadsperblock](d_hsv_image, d_filtered_edges, d_additive_magnitude[0], scale, d_output_hsv)

    # Convert to RGB
    hsv_image_to_rgb_batch_kernel[blockspergrid, threadsperblock](d_output_hsv, d_rgb_image)

    # Copy back and save
    output_rgb = d_rgb_image.copy_to_host().astype(np.uint8)
    cv2.imwrite(output_path, output_rgb)
