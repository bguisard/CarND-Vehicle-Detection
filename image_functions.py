import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage import transform
import bcolz


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def read_all_imgs(path, color=cv2.IMREAD_COLOR):

    images = []
    filenames = []

    filelist = os.listdir(path)
    for file in filelist:

        try:
            img = cv2.imread(path + file, color)
        except:
            img = None

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            filenames.append(file)

    return images, filenames


def HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def HLS_RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_HLS2RGB)


def HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def HSV_RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def GRAY(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def calibrate_camera(images, pattern):
    """
    Function to calibrate camera based on multiple images of a checkerboard pattern.

    Source:
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

    Inputs:
    images - Numpy array of all calibration images
    pattern - tuple with the Checkerboard pattern from calibration images in the format (r, c) where
              r is the number of crossings per row and c is the number of crossings per column

    """
    # termination criteria to refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Create a meshgrid of points
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    # Create Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for img in images:
        gray = GRAY(img)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern, None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Now that we have our object points and corners we can perform our camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


def undistort_image(image, camera_params):

    ret, mtx, dist, rvecs, tvecs = camera_params

    dst = cv2.undistort(image, mtx, dist, None, mtx)

    return dst


def correct_and_plot(image, camera_params):
    implot = plt.figure(figsize=(16, 16))

    ax1 = implot.add_subplot(1, 2, 1)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('Original Image')

    ax2 = implot.add_subplot(1, 2, 2)
    ax2.grid(False)
    ax2.axis('off')
    ax2.imshow(undistort_image(image, camera_params))
    ax2.set_title('Undistorted Image')

    plt.show()


def plot_two(img1, img2, title1='Img_1', title2='Img_2', cmap1=None, cmap2=None):
    implot = plt.figure(figsize=(16, 16))

    ax1 = implot.add_subplot(1, 2, 1)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1)

    ax2 = implot.add_subplot(1, 2, 2)
    ax2.grid(False)
    ax2.axis('off')
    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2)

    plt.show()


def color_mask(img, return_bin=False):
    """

    This function will convert an image to HSV color space and apply two different masks:
    one for the white and another one for yellow and will return a color masked image in the HSV color space.

    """

    hsv = HSV(img)

    # White Filter:
    lower_w = np.array([0, 0, 200])
    upper_w = np.array([180, 25, 255])

    hsv_w_mask = cv2.inRange(hsv, lower_w, upper_w)

    hsv_w = cv2.bitwise_and(hsv, hsv, mask=hsv_w_mask)

    # Yellow Filter
    lower_y = np.array([15, 75, 75])
    upper_y = np.array([35, 255, 255])

    hsv_y_mask = cv2.inRange(hsv, lower_y, upper_y)

    hsv_y = cv2.bitwise_and(hsv, hsv, mask=hsv_y_mask)

    color_masked = cv2.add(hsv_y, hsv_w)

    if return_bin:
        gray = GRAY(color_masked)
        binary = np.zeros_like(gray)
        binary[(gray > 0)] = 1
        return binary

    return color_masked


def new_color_mask(img):
    """

    New color mask fucntion that combines three different color space filters
    to increasee detection accuracy.

    """

    hsv = HSV(img)
    hls = HLS(img)
    binary = np.zeros_like(img[:, :, 0])

    # Yellow filter
    lower_yellow = (15, 75, 75)
    upper_yellow = (35, 255, 255)
    hsv_y_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_hls_yellow = (10, 0, 20)
    upper_hls_yellow = (30, 255, 255)
    hls_y_mask = cv2.inRange(hls, lower_hls_yellow, upper_hls_yellow)

    # White filter
    hsv_sensitivity = 75
    lower_hsv_white = (0, 0, 255 - hsv_sensitivity)
    upper_hsv_white = (180, 25, 255)
    hsv_w_mask = cv2.inRange(hsv, lower_hsv_white, upper_hsv_white)

    hls_sensitivity = 75
    lower_hls_white = (0, 255 - hls_sensitivity, 0)
    upper_hls_white = (255, 255, hls_sensitivity)
    hls_w_mask = cv2.inRange(hls, lower_hls_white, upper_hls_white)

    rgb_sensitivity = 55
    lower_rgb_white = (255 - rgb_sensitivity, 255 - rgb_sensitivity, 255 - rgb_sensitivity)
    upper_rgb_white = (255, 255, 255)
    rgb_w_mask = cv2.inRange(img, lower_rgb_white, upper_rgb_white)

    bit_layer = binary | hsv_y_mask | hls_y_mask | hsv_w_mask | hls_w_mask | rgb_w_mask

    return bit_layer


def sobel(img, orient='x', thres=(20, 100), input_format='RGB'):

    # Check if image has 3 channels as the only accepted formats are RGB or HSV
    try:
        img.shape[2] == 3
    except:
        print ('ERROR: input format should be either RGB or HSV')
        return

    # Check if user selected a valid input format
    if input_format == 'RGB':
        gray = GRAY(img)
    elif input_format == 'HSV':
        gray = GRAY(HSV_RGB(img))
    else:
        print ('ERROR: input format should be either RGB or HSV')
        return

    # Sobel transform as suggested by lessons
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Thresholding
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thres[0]) & (scaled_sobel <= thres[1])] = 1

    return binary_output


def complex_sobel(image, sobel_kernel=5, params={'thres_gradx': (0, 255),
                                                 'thres_mag': (0, 255),
                                                 'thres_dir': (0, np.pi / 2),
                                                 'thres_s': (0, 255)}):

    result = np.zeros(image.shape[:2], dtype=np.uint8)

    gray = GRAY(image)
    hsv = HSV(image)
    s = hsv[:, :, 1]

    for c in [gray, s]:
        gradx = abs_sobel_thresh(c, orient='x', sobel_kernel=sobel_kernel, thresh=params['thres_gradx'])
        mag_bin = mag_thresh(c, sobel_kernel=sobel_kernel, thresh=params['thres_mag'])
        dir_bin = dir_thresh(c, sobel_kernel=sobel_kernel, thresh=params['thres_dir'])
        result[((gradx == 1)) | ((mag_bin == 1) & (dir_bin == 1))] = 1

    s_bin = np.zeros_like(s)
    s_bin[(s >= params['thres_s'][0]) & (s <= params['thres_s'][1])] = 1

    result[(s_bin == 1)] = 1
    return result


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # Thresholding
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return mag_binary


def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi / 2)):

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Thresholding
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary


def perspective_transform(image, calc_points=True):

    img_w = image.shape[1]
    img_h = image.shape[0]

    src = np.zeros((4, 2), dtype=np.float32)

    if calc_points:
        src[0] = [np.uint(img_w * 0.375), np.uint(img_h * 2 / 3)]
        src[1] = [np.uint(img_w * 0.625), np.uint(img_h * 2 / 3)]
        src[2] = [img_w, img_h]
        src[3] = [0, img_h]
    else:
        # Coordinates for the ROI based on the fixed camera mount and fixed car position in this exercise:
        src[0] = [480., 480.]                    # [ img width * 0.375, img_height * 2/3]
        src[1] = [800., 480.]                    # [ img width * 0.625, img_height * 2/3]
        src[2] = [1280., 720.]                   # [ img width * 1,     img_height * 1  ]
        src[3] = [0., 720.]                      # [ img width * 0,     img_height * 1  ]

    # Calculate the destination points
    dst = np.zeros((4, 2), dtype=np.float32)
    dst[0] = [0., 0.]
    dst[1] = [img_w, 0.]
    dst[2] = [img_w, img_h]
    dst[3] = [0., img_h]

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (img_w, img_h))
    return warped, M, Minv


def new_perspective_transform(image):

    img_w = image.shape[1]
    img_h = image.shape[0]

    src = np.zeros((4, 2), dtype=np.float32)

    # New method:
    src[0] = [np.uint(img_w / 2) - 55, np.uint(img_h / 2) + 100]
    src[1] = [np.uint(img_w / 2 + 55), np.uint(img_h / 2) + 100]
    src[2] = [np.uint(img_w * 5 / 6) + 60, img_h]
    src[3] = [(np.uint(img_w / 6) - 10), img_h]

    # Calculate the destination points
    dst = np.zeros((4, 2), dtype=np.float32)
    dst[0] = [np.uint(img_w / 4), 0]
    dst[1] = [np.uint(img_w * 3 / 4), 0]
    dst[2] = [np.uint(img_w * 3 / 4), img_h]
    dst[3] = [np.uint(img_w / 4), img_h]

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (img_w, img_h))
    return warped, M, Minv


def plot_img_mask_from_folder(filename, figsize=(24, 24), orient=0,
                              img_path='./datasets/deeplearning/udacity/',
                              mask_path='./datasets/deeplearning/masks/mask_'):

    """
    This function will plot the original image, it's mask and the masked image.

    It's meant to help us visualize that everything is in order before we
    proceed to training our model.

    """
    if orient == 0:
        rows = 3
        cols = 1
    else:
        rows = 1
        cols = 3

    img = cv2.imread(img_path + filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_mask = cv2.imread(mask_path + filename, cv2.IMREAD_GRAYSCALE)
    img_mask = np.reshape(img_mask, (np.shape(img_mask)[0], np.shape(img_mask)[1], 1))

    plt.figure(figsize=figsize)
    plt.subplot(rows, cols, 1)
    plt.title('Image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(rows, cols, 2)
    plt.title('Vehicle Mask')
    plt.imshow(img_mask[:, :, 0])
    plt.axis('off')

    plt.subplot(rows, cols, 3)
    plt.imshow(cv2.bitwise_and(img, img, mask=img_mask))
    plt.title('Masked Result')
    plt.axis('off')
    plt.show()


def plot_img_mask(img, img_mask, figsize=(24, 24), orient=0):

    """
    This function will plot the original image, it's mask and the masked image.

    It's meant to help us visualize that everything is in order before we
    proceed to training our model.

    """
    if orient == 0:
        rows = 3
        cols = 1
    else:
        rows = 1
        cols = 3

    plt.figure(figsize=figsize)
    plt.subplot(rows, cols, 1)
    plt.title('Image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(rows, cols, 2)
    plt.title('Vehicle Mask')
    plt.imshow(img_mask[:, :, 0])
    plt.axis('off')

    plt.subplot(rows, cols, 3)
    plt.imshow(cv2.bitwise_and(img, img, mask=img_mask))
    plt.title('Masked Result')
    plt.axis('off')
    plt.show()


def Motionblur(img, kernel_sz=3):
    imshape = img.shape
    kernel_mb = np.zeros((kernel_sz, kernel_sz))
    kernel_mb[int((kernel_sz - 1) / 2), :] = np.ones(kernel_sz)
    kernel_mb = kernel_mb / kernel_sz
    blur = cv2.filter2D(img, -1, kernel_mb)
    return blur.reshape(*imshape)


def Affine(img, img_mask, scale_limit=0.05, angle_limit=8., shear_limit=5., trans_limit=0.02):
    h, w = img.shape[:2]
    trans_limit_x = int(w * trans_limit)
    trans_limit_y = int(h * trans_limit)
    centering = np.array((h, w)) / 2. - 0.5
    scale = random.uniform(1 - scale_limit, 1 + scale_limit)
    angle = np.deg2rad(random.uniform(-angle_limit, angle_limit))
    shear = np.deg2rad(random.uniform(-shear_limit, shear_limit))
    trans_x = random.uniform(-trans_limit_x, trans_limit_x)
    trans_y = random.uniform(-trans_limit_y, trans_limit_y)

    center = transform.SimilarityTransform(translation=-centering)
    tform = transform.AffineTransform(scale=(scale, scale),
                                      rotation=angle,
                                      shear=shear,
                                      translation=(trans_x, trans_y))
    recenter = transform.SimilarityTransform(translation=centering)

    aug_img = transform.warp(img, (center + (tform + recenter)).inverse, mode='edge')
    aug_mask = transform.warp(img_mask, (center + (tform + recenter)).inverse, mode='edge')

    return aug_img, aug_mask
