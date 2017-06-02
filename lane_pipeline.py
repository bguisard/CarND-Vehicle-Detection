import cv2
import numpy as np
from image_functions import color_mask, sobel, perspective_transform
from image_functions import new_color_mask, new_perspective_transform
from laneline import Line


def find_line_raw(warped, nwindows=9, return_img=False, plot_boxes=False, plot_line=False):

    """
    This function finds lane lines based on the sliding window algorithm from the lessons. It's "raw" because
    it doesn't receive any previous information about where the lane lines may be and have to look exclusively at
    the warped binary image to find it.

    The default is to not return the image, only the equation for the lane lines, but one can set return_img to True
    to debug.

    """

    img_h = warped.shape[0]

    if return_img:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped, warped, warped)) * 255

    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[np.uint(img_h / 2):, :], axis=0)

    # Find peak points for left and right halves of histogram
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if return_img and plot_boxes:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        left_fit = [[0]]
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        right_fit = [[0]]

    if return_img:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        if plot_line:
            ploty = np.linspace(0, img_h - 1, img_h)
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            cv2.polylines(out_img, np.int32([np.column_stack((left_fitx, ploty))]), False, (255, 255, 0), 2)
            cv2.polylines(out_img, np.int32([np.column_stack((right_fitx, ploty))]), False, (255, 255, 0), 2)

        return left_fit, right_fit, out_img

    return left_fit, right_fit


def find_line_recursive(warped, left_fit, right_fit, margin=100, return_img=False, plot_boxes=False, plot_line=False):

    """
    This is the recursive version of the lane finder. It needs information from the previous frame in order to
    locate the lane lines.

    Margin controls how wide from the previous lane line equation we are going to look

    """

    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        left_fit = [np.array([0])]
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        right_fit = [np.array([0])]

    if return_img:

        img_h = warped.shape[0]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((warped, warped, warped)) * 255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_h - 1, img_h)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        if plot_line:
            cv2.polylines(out_img, np.int32([np.column_stack((left_fitx, ploty))]), False, (255, 255, 0), 2)
            cv2.polylines(out_img, np.int32([np.column_stack((right_fitx, ploty))]), False, (255, 255, 0), 2)

        if plot_boxes:

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))

            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            out_img = result.copy()

        return left_fit, right_fit, out_img

    return left_fit, right_fit


def get_curvature(poly, y_eval=0):

    A, B, C = poly
    R = ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / np.absolute(2 * A)

    return R


def curv_to_m(R, y_mpp=30 / 720, x_mpp=3.7 / 700):
    """ Converts our curvature to m based on meters per pixel for y and x """
    return R / (x_mpp / (y_mpp ** 2))


def get_curvature_m(poly, imsize=(720, 1280), y_eval=0, y_mpp=30 / 720, x_mpp=3.7 / 700):
    """
    Helper function I used to make sure the transformation above works. I will use the math version in the pipeline
    as it is quicker.
    """
    img_h = imsize[0]

    A, B, C = poly
    ploty = np.linspace(0, img_h - 1, img_h)
    leftx = np.array([A * y**2 + B * y + C for y in ploty])

    poly_converted = np.polyfit(ploty * y_mpp, leftx * x_mpp, 2)

    A, B, C = poly_converted

    R = ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / np.absolute(2 * A)

    return R


def get_lane_curvature(leftline, rightline, y_eval=0, y_mpp=30 / 720, x_mpp=3.7 / 700):

    assert np.all(leftline.ploty == rightline.ploty)
    assert np.all(leftline.ppm == rightline.ppm)

    Rx = rightline.bestx * x_mpp
    Rl = leftline.bestx * x_mpp

    Ymid = leftline.ploty * y_mpp
    Xmid = np.mean((Rx, Rl), 0)

    midlane = np.polyfit(Ymid, Xmid, 2)

    return ((1 + (2 * midlane[0] * y_eval + midlane[1]) ** 2) ** 1.5) / np.absolute(2 * midlane[0])


def get_lane_offset(left_fit, right_fit, imsize=(720, 1280), x_mpp=3.7 / 700):

    img_h = imsize[0]

    img_center = imsize[1] / 2
    left_lane_start = left_fit[0] * img_h ** 2 + left_fit[1] * img_h + left_fit[2]
    right_lane_start = right_fit[0] * img_h ** 2 + right_fit[1] * img_h + right_fit[2]
    lane_center = np.mean((left_lane_start, right_lane_start))

    dist = (img_center - lane_center) * x_mpp

    if dist == 0:
        side = ' '
    elif dist > 0:
        side = 'right'
    else:
        side = 'left'

    return np.absolute(dist), side


def plot_lanelines(img, warped, left_fit, right_fit, Minv, out_img=None, R=None, lane_offset=None):

    img_h = img.shape[0]
    img_w = img.shape[1]

    # Create an image to draw the lines on
    if out_img is None:
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    else:
        color_warp = out_img.copy()

    ploty = np.linspace(0, img_h - 1, img_h)

    A, B, C = left_fit
    leftx = np.array([A * y**2 + B * y + C for y in ploty])

    A, B, C = right_fit
    rightx = np.array([A * y**2 + B * y + C for y in ploty])

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_w, img_h))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    # Add Text
    font = cv2.FONT_HERSHEY_SIMPLEX

    if R:
        text = 'Lane Radius of curvature: {:.2f}km'.format(R / 1000)
        result = cv2.putText(result, text, (50, 50), font, 1, (255, 255, 255), 2)

    if lane_offset:
        text = 'Vehicle is {:.2f}m {} of lane center'.format(lane_offset[0], lane_offset[1])
        result = cv2.putText(result, text, (50, 100), font, 1, (255, 255, 255), 2)

    return result


def test_img_pipeline(img):
    # Apply color masking
    color_masked = color_mask(img)

    # Apply sobel
    sobel_ = sobel(color_masked, input_format='HSV')

    # Perspective Transform
    warped, M, Minv = perspective_transform(sobel_)

    # Find lane lines
    left_fit, right_fit, out_img = find_line_raw(warped, return_img=True)

    # Find curvature
    Rl = curv_to_m(get_curvature(left_fit, 360))
    Rr = curv_to_m(get_curvature(right_fit, 360))
    R = curv_to_m((Rl + Rr) / 2)

    # Find lane offset
    lane_offset = get_lane_offset(left_fit, right_fit)

    result = plot_lanelines(img, warped, left_fit, right_fit, Minv, out_img, R=R, lane_offset=lane_offset)

    return result


def find_lane_lines(leftline, rightline, warped, nwindows=9, margin=100, return_img=False, plot_boxes=False, plot_line=False, verbose=0):

    """
    This function finds lane lines on images using two different approaches.

    1 - If is the first iteration of leftlane and rightlane it will use a slower
        algorithm to find lane lines looking through all nonzero points of the
        warped binary.

    2 - If it's not the first iteration of the lane class objects it will use
        a window to search the points based on the best fit that the lane class
        objects have.

    Inputs:
    leftlane - object of Lane class
    rightlane - object of Lane class
    warped - Warped binary of image that went trough our pipeline.

    Outputs:
    Updates - leftlane and rightlane
    out_img - [Optional] - Returns a top view of image with pixels found and
                           boxes of search [optional] and lane lines [optional]
    """

    # We initialize the function assuming the lines were never detected before
    use_robust = True
    use_recursive = False

    # Storing image size
    img_h = warped.shape[0]

    if verbose > 0:
        print ('Processing Image: {}'.format(leftline.iter))
    leftline.iter += 1
    rightline.iter += 1

    # And check to see if we should change our assumption
    # First is to check if the lanes were detected on the last iteration
    # if both were detected, we will use the recursive version.
    if leftline.detected and rightline.detected:
        use_robust = False
        use_recursive = True
        if verbose > 0:
            print ("Lanes were detected on last iteration. Using recursive")

    # We will do the same if any line was found in the previous 5 iterations:
    if use_recursive is False and leftline.last_detected < 5 and rightline.last_detected < 5:
        use_robust = False
        use_recursive = True
        if verbose > 0:
            print ("Lanes were detected in the near past. Using recursive")

    if leftline.use_robust or rightline.use_robust:
        use_robust = True
        use_recursive = False

    if use_robust:
        use_recursive = False  # Just a safety parameter to ensure recursive wont run after this module
        if verbose > 0:
            print ("Locating lane lines using robust mode")

        if return_img:
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((warped, warped, warped)) * 255

        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped[np.uint(img_h / 2):, :], axis=0)

        # Find peak points for left and right halves of histogram
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(img_h / nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        if leftx_base > 200:
            leftx_current = leftx_base
        else:
            leftx_current = 250

        if rightx_base > 800:
            rightx_current = rightx_base
        else:
            rightx_current = 900

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if return_img and plot_boxes:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a polynomial to each side
        leftline.fitpoly((lefty, leftx))
        rightline.fitpoly((righty, rightx))

        if return_img:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            if plot_line:
                if np.nansum(leftline.bestx > 0):
                    cv2.polylines(out_img, np.int32([np.column_stack((leftline.bestx, leftline.ploty))]), False, (255, 255, 0), 2)
                if np.nansum(rightline.bestx > 0):
                    cv2.polylines(out_img, np.int32([np.column_stack((rightline.bestx, rightline.ploty))]), False, (255, 255, 0), 2)

            return leftline.best_fit, rightline.best_fit, out_img

        return leftline.best_fit, rightline.best_fit

    if use_recursive:
        if verbose > 0:
            print ("Locating lane lines using recursive mode")

        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        pl = np.poly1d(leftline.best_fit)
        pr = np.poly1d(rightline.best_fit)

        left_lane_inds = (nonzerox > (pl(nonzeroy) - margin)) & (nonzerox < (pl(nonzeroy) + margin))
        right_lane_inds = (nonzerox > (pr(nonzeroy) - margin)) & (nonzerox < (pr(nonzeroy) + margin))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a polynomial to each side
        leftline.fitpoly((lefty, leftx))
        rightline.fitpoly((righty, rightx))

        if return_img:

            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((warped, warped, warped)) * 255
            window_img = np.zeros_like(out_img)

            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            if plot_line:
                if np.nansum(leftline.bestx > 0):
                    cv2.polylines(out_img, np.int32([np.column_stack((leftline.bestx, leftline.ploty))]), False, (255, 255, 0), 2)
                if np.nansum(rightline.bestx > 0):
                    cv2.polylines(out_img, np.int32([np.column_stack((rightline.bestx, rightline.ploty))]), False, (255, 255, 0), 2)

            if plot_boxes:

                # Generate a polygon to illustrate the search window area
                # And recast the x and y points into usable format for cv2.fillPoly()
                left_line_window1 = np.array([np.transpose(np.vstack([leftline.bestx - margin, leftline.ploty]))])
                left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([leftline.bestx + margin, leftline.ploty])))])
                left_line_pts = np.hstack((left_line_window1, left_line_window2))

                right_line_window1 = np.array([np.transpose(np.vstack([rightline.bestx - margin, rightline.ploty]))])
                right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rightline.bestx + margin, rightline.ploty])))])
                right_line_pts = np.hstack((right_line_window1, right_line_window2))

                # Draw the lane onto the warped blank image
                cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
                cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
                result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
                out_img = result.copy()

            return leftline.best_fit, rightline.best_fit, out_img

        return leftline.best_fit, rightline.best_fit


def export_frames(clip, path, filename):
    f = 0
    for frame in clip.iter_frames():
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path + '/video_frames/{}_frame_{}.jpg'.format(filename, str(f + 10000)), img)
        f += 1


def video_pipeline(img, leftline, rightline, use_sobel=True):
    """
    This function applies our pipeline to the frames of a video stream that is captured
    using VideoFileClip.fl_image. The user needs to initialize two instances of lanelines
    befor using this pipeline and call them leftline and rightline.
    """

    if use_sobel is False:
        # Apply color masking
        binary = color_mask(img, return_bin=True)
    else:
        # Apply color masking and sobel
        color_masked = color_mask(img)
        binary = sobel(color_masked, input_format='HSV')

    # Perspective Transform
    warped, M, Minv = perspective_transform(binary)

    # Find lane lines
    left_fit, right_fit, out_img = find_lane_lines(leftline, rightline, warped, nwindows=9, margin=100, return_img=True, plot_boxes=False, plot_line=False, verbose=0)

    # Find curvature
    R = get_lane_curvature(leftline, rightline)

    # Find lane offset
    lane_offset = get_lane_offset(left_fit, right_fit)

    result = plot_lanelines(img, warped, left_fit, right_fit, Minv, out_img, R=R, lane_offset=lane_offset)

    return result


def new_video_pipeline(img, leftline, rightline):
    """
    This function applies our pipeline to the frames of a video stream that is captured
    using VideoFileClip.fl_image. The user needs to initialize two instances of lanelines
    befor using this pipeline and call them leftline and rightline.

    To achieve more robust results this pipeline starts with the perspective
    transform and relies in color thresholding from several different color spaces
    instead of Sobel transform.
    """

    # Perspective Transform
    warped, M, Minv = new_perspective_transform(img)

    # Color masking
    # warped_bin = new_color_mask(warped)
    warped_bin = color_mask(warped, return_bin=True)

    # Find lane lines
    left_fit, right_fit, out_img = find_lane_lines(leftline, rightline, warped_bin, nwindows=9, margin=100, return_img=True, plot_boxes=False, plot_line=False, verbose=0)

    # Find curvature
    R = get_lane_curvature(leftline, rightline, img.shape[0] // 2)

    # Find lane offset
    lane_offset = get_lane_offset(left_fit, right_fit)

    result = plot_lanelines(img, warped, left_fit, right_fit, Minv, out_img, R=R, lane_offset=lane_offset)

    return result
