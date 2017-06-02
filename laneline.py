import numpy as np
from collections import deque


class Line():
    def __init__(self, poly_order=2, image_size=(720, 1280), ppm=(30 / 720, 3.7 / 700), avg_lines=1, wght='linear',
                 name='line', smoothing=0.0, dist_thres=0, slope_thres=0, curv_thres=0):

        # Storing line name
        self.name = name

        # Variable to validate line
        self.valid_fit = False

        # Storing smoothing factors
        self.smoothing = smoothing
        self.avg_lines = avg_lines
        self.dist_thres = dist_thres
        self.slope_thres = slope_thres
        self.curv_thres = curv_thres

        # Polynomial order
        self.poly_order = poly_order

        # Boolean to force our algo to use the more robust lane finding method
        self.use_robust = True

        # Create weight vector for weighted moving average
        if wght == 'linear':
            self.weights = np.array([1] * avg_lines)
        else:
            self.weights = np.linspace(avg_lines, 1, avg_lines) / avg_lines

        self.weights2 = np.linspace(avg_lines, 1, avg_lines, dtype='int')

        # Save image size parameters and pixel per meter conversion
        self.img_h = image_size[0]
        self.img_w = image_size[1]
        self.ppm = ppm

        # Calculate ploty once and store it
        self.ploty = np.linspace(0, self.img_h - 1, self.img_h)

        # was the line detected in the last iteration?
        self.detected = False

        # How many iterations ago was the last detection?
        self.last_detected = 100

        # x values of the last fit of the line
        self.current_xfitted = []

        # polynomial coefficients for the most recent fit
        self.current_fit = np.array([np.nan] * (self.poly_order + 1), dtype='float')

        # polynomial coefficients for the last n iterations stored in deque
        self.all_fits = deque(maxlen=avg_lines)

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([np.nan] * (self.poly_order + 1), dtype='float')

        # x values of the fitted line over the last n iterations
        self.bestx = []

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([np.nan] * (self.poly_order + 1), dtype='float')

        # Values for detected line pixels queued for the past N=avg_lines iterations
        self.allx = deque(maxlen=avg_lines)
        self.ally = deque(maxlen=avg_lines)

        # radius of curvature of the line in pixel units
        self.radius_of_curvature = np.nan

        # distance in pixel units of vehicle center from the line
        self.line_base_pos = np.nan

        # iter counter
        self.iter = 0

    def fitpoly(self, pts):
        self.allx.append(pts[1])
        self.ally.append(pts[0])

        # We try to fit a polynomial to the provided points
        try:
            poly = np.polyfit(self.ally[-1], self.allx[-1], self.poly_order)
        except:
            poly = np.array([0] * (self.poly_order + 1), dtype='float')

        self.valid_fit = self.validate_fit(poly)
        self.diffs = self.current_fit - self.best_fit

        # update self parameters based on whether we found a fit or not
        if np.sum(poly) == 0 or self.valid_fit is False:
            # print ("{} lane not detected".format(self.name))
            self.detected = False
            self.last_detected += 1
            self.current_fit = np.array([np.nan] * (self.poly_order + 1), dtype='float')

            # We will not call update_best_fit since we don't have any useful fit in this iteration
            # but we still need to erase the last xfitted values
            self.current_xfitted = np.asanyarray([], dtype='float')

            if self.last_detected >= 5:
                self.use_robust = True
        else:
            self.detected = True
            self.use_robust = False
            self.last_detected = 0
            self.current_fit = poly
            self.update_best_fit()

    def fit_x(self, poly):

        p = np.poly1d(poly)

        return np.array([p(y) for y in self.ploty])

    def update_best_fit(self):

        self.current_xfitted = self.fit_x(self.current_fit)
        self.all_fits.append(self.current_fit)

        self.best_fit = self.smooth_fit()

        self.bestx = self.fit_x(self.best_fit)

        self.radius_of_curvature = self.get_radius(self.bestx, self.img_h - 1)

        self.line_base_pos = self.bestx[-1] - self.img_w / 2

    def get_radius(self, x_pts, y_eval=0):
        """
        For this step it's easier to just refit a 2nd order polynomial to the points
        converted to real life measures.
        """

        A, B, C = np.polyfit(self.ploty * self.ppm[0], x_pts * self.ppm[1], 2)
        R = ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / np.absolute(2 * A)

        return R

    def smooth_fit(self):

        if np.nansum(self.best_fit) > 0 and self.last_detected < 10:
            new_fit = np.zeros_like(self.best_fit)
            avg_fit = np.average(self.all_fits, 0, self.weights[:len(self.all_fits)])

            for i, (cb, cn) in enumerate(zip(self.best_fit, avg_fit)):
                new_fit[i] = self.smoothing * cb + (1 - self.smoothing) * cn

            return new_fit
        else:
            return self.current_fit

    def average_fit(self):
        all_x = np.array([], dtype='int')
        all_y = np.array([], dtype='int')

        for i, (xpts, ypts) in enumerate(zip(self.allx, self.ally)):
            x = np.tile(xpts, self.weights2[i])
            all_x = np.append(all_x, x)
            y = np.tile(ypts, self.weights2[i])
            all_y = np.append(all_y, y)

        avg_fit = np.polyfit(all_y, all_x, self.poly_order)

        return avg_fit

    def validate_fit(self, poly, y_eval=0):

        # If our best_fit is NaN we will take this line as a good line.
        if np.nansum(self.best_fit) == 0:
            return True

        fittedx = self.fit_x(poly)

        # Check curvature of current line vs best line
        if self.curv_thres > 0:
            curv_valid = False
            new_fit_curv = self.get_radius(fittedx, self.img_h - 1)

            if abs(new_fit_curv / self.radius_of_curvature - 1) <= self.curv_thres:
                curv_valid = True
        else:
            curv_valid = True

        # Check slope from the midpoint of current line vs bestline
        if self.slope_thres > 0:
            mid_h = np.floor_divide(self.img_h, 2)
            slope_valid = False
            best_fit_slope = (self.bestx[-1] - self.bestx[mid_h]) / (self.ploty[-1] - self.ploty[mid_h])
            new_fit_slope = (fittedx[-1] - fittedx[mid_h]) / (self.ploty[-1] - self.ploty[mid_h])

            if abs(new_fit_slope / best_fit_slope - 1) <= self.slope_thres:
                slope_valid = True
        else:
            slope_valid = True

        # Check distances between 5 different points
        if self.dist_thres > 0:
            distances = np.array([np.absolute(self.bestx[y] - fittedx[y]) for y in np.linspace(0, 719, 5, dtype='int')])
            dist_valid = np.all(distances < self.dist_thres)
        else:
            dist_valid = True

        if (curv_valid and slope_valid and dist_valid):
            return True
        else:
            return False
