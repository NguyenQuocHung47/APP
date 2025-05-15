import cv2
import math
import numpy as np
from numba import cuda
from numba import config
import time
config.CUDA_ENABLE_PYNVJITLINK = 1
from PIL import Image

@cuda.jit
def rgb_to_hsv_kernel(rgb_image, hsv_image):
    i, j = cuda.grid(2)
    if i < rgb_image.shape[0] and j < rgb_image.shape[1]:
        r = rgb_image[i, j, 2]
        g = rgb_image[i, j, 1]
        b = rgb_image[i, j, 0]

        maxc = max(r, g, b)
        minc = min(r, g, b)
        v = maxc

        if minc == maxc:
            h = 0.0
            s = 0.0
        else:
            s = (maxc - minc) / maxc
            rc = (maxc - r) / (maxc - minc)
            gc = (maxc - g) / (maxc - minc)
            bc = (maxc - b) / (maxc - minc)

            if r == maxc:
                h = bc - gc
            elif g == maxc:
                h = 2.0 + rc - bc
            else:
                h = 4.0 + gc - rc

            h = (h / 6.0) % 1.0

        hsv_image[i, j, 0] = h
        hsv_image[i, j, 1] = s
        hsv_image[i, j, 2] = v

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
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        flat_values = value_channel.reshape(-1)
        max_v = 0.0
        min_v = 1.0
        sum_v = 0.0
        for i in range(flat_values.shape[0]):
            val = flat_values[i]
            max_v = max(max_v, val)
            min_v = min(min_v, val)
            sum_v += val
        avg_v = sum_v / flat_values.shape[0]
        mid_v = (max_v + min_v) / 2.0
        result[0] = (max_v / 8.0) * (avg_v / mid_v)

@cuda.jit
def sharpen_edges_kernel(hsv_image, edges, additive_magnitude, scale, output_hsv):
    i, j = cuda.grid(2)
    if 1 <= i < hsv_image.shape[0]-1 and 1 <= j < hsv_image.shape[1]-1:
        output_hsv[i, j, 0] = hsv_image[i, j, 0]
        output_hsv[i, j, 1] = hsv_image[i, j, 1]
        output_hsv[i, j, 2] = hsv_image[i, j, 2]
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
        h = h * 360.0  # Scale to 0-360
        h_i = int(h / 60.0) % 6
        f = (h / 60.0) - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:  # h_i == 5
            r, g, b = v, p, q

        # OpenCV uses BGR order
        result[i, j, 2] = b * 255.0  # B
        result[i, j, 1] = g * 255.0  # G
        result[i, j, 0] = r * 255.0  # R

# Main GPU processing function

def process_image_on_gpu(input_path, output_path, edge_threshold=0.1, neighbor_threshold=2, scale=0.5):
    timings = {}
    start_time = time.time()  # Start time for the entire process
    img = cv2.imread(input_path)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    height, width = hsv_image.shape[:2]
    value_channel = hsv_image[:, :, 2]
    img = cv2.imread(input_path)
    d_rgb_image = cuda.to_device(img.astype(np.float32))
    d_hsv_image = cuda.device_array((img.shape[0], img.shape[1], 3), dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid = (
    math.ceil(img.shape[0] / threadsperblock[0]),
    math.ceil(img.shape[1] / threadsperblock[1])
    )
    rgb_to_hsv_start= time.time()
    rgb_to_hsv_kernel[blockspergrid, threadsperblock](d_rgb_image, d_hsv_image)
    cuda.synchronize()
    rgb_to_hsv_end = time.time()
    timings['RGB to HSV'] = rgb_to_hsv_end - rgb_to_hsv_start
    # Prepare data
    kernel_size = 5
    sigma = 1.0
    kernel = create_gaussian_kernel(kernel_size, sigma)

    d_value_channel = cuda.to_device(np.ascontiguousarray(value_channel))
    d_blurred_channel = cuda.device_array_like(value_channel)
    d_edges = cuda.device_array(value_channel.shape, dtype=np.int32)
    d_filtered_edges = cuda.device_array(value_channel.shape, dtype=np.int32)
    d_hsv_image = cuda.to_device(hsv_image)
    d_output_hsv = cuda.device_array_like(hsv_image)
    d_rgb_image = cuda.device_array_like(img)
    d_kernel = cuda.to_device(kernel)
    d_additive_magnitude = cuda.device_array(1, dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid = (
        math.ceil(height / threadsperblock[0]),
        math.ceil(width / threadsperblock[1])
    )

    # Gaussian Blur
    blur_start = time.time()  # Start time for Gaussian blur
    apply_gaussian_blur_to_channel_kernel[blockspergrid, threadsperblock](d_value_channel, d_kernel, d_blurred_channel, kernel_size)
    cuda.synchronize()
    blur_end = time.time()  # End time for Gaussian blur
    timings['Gaussian Blur'] = blur_end - blur_start

    # Edge Detection
    edge_start = time.time()  # Start time for edge detection
    detect_edges_kernel[blockspergrid, threadsperblock](d_blurred_channel, d_edges, edge_threshold)
    cuda.synchronize()
    edge_end = time.time()  # End time for edge detection
    timings['Edge Detection'] = edge_end - edge_start

    # Edge Filtering
    filter_start = time.time()  # Start time for edge filtering
    low_pass_filter_kernel[blockspergrid, threadsperblock](d_edges, d_filtered_edges, neighbor_threshold)
    cuda.synchronize()
    filter_end = time.time()  # End time for edge filtering
    timings['Edge Filtering'] = filter_end - filter_start

    # Additive Magnitude
    additive_start = time.time()  # Start time for additive magnitude calculation
    calculate_additive_magnitude_kernel[1, 1](d_value_channel, d_additive_magnitude)
    cuda.synchronize()
    additive_end = time.time()  # End time for additive magnitude calculation
    timings['Additive Magnitude'] = additive_end - additive_start

    # Sharpen Edges
    sharpen_start = time.time()  # Start time for sharpening edges
    sharpen_edges_kernel[blockspergrid, threadsperblock](d_hsv_image, d_filtered_edges, d_additive_magnitude[0], scale, d_output_hsv)
    cuda.synchronize()
    sharpen_end = time.time()  # End time for sharpening edges
    timings['Sharpen Edges'] = sharpen_end - sharpen_start

    # HSV to RGB
    hsv2rgb_start = time.time()  # Start time for HSV to RGB conversion
    hsv_image_to_rgb_batch_kernel[blockspergrid, threadsperblock](d_output_hsv, d_rgb_image)
    cuda.synchronize()
    hsv2rgb_end = time.time()  # End time for HSV to RGB conversion
    timings['HSV to RGB'] = hsv2rgb_end - hsv2rgb_start

    output_rgb = d_rgb_image.copy_to_host().astype(np.uint8)
    cv2.imwrite(output_path, output_rgb)

    end_time = time.time()  # End time for the entire process
    total_time = end_time - start_time
    print("Processing completed in {:.4f} seconds.".format(total_time))

    # Print the time taken for each step
    for step, duration in timings.items():
        print(f"{step} : {duration:.4f} seconds")

input_path = 'images/image1.jpg'  # 4000x6000 pixel
output_path = 'outputs/parallel/sharpened_image1_gpu.jpg'  # 4000x6000 pixel
process_image_on_gpu(input_path, output_path, edge_threshold=0.1, neighbor_threshold=2, scale=0.5)
