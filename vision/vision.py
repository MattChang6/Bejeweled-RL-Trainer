import cv2 as cv
import numpy as np
from hsvfilter import hsvFilter


class Vision:
    # constants
    TRACKBAR_WINDOW = "Trackerbars"

    # properties
    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None

    # constructor
    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        if needle_img_path:
            # load the image we're trying to match
            self.needle_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)

            if self.needle_img.shape[2] == 4:
                # Convert RGBA to RGB
                self.needle_img = cv.cvtColor(self.needle_img, cv.COLOR_BGRA2BGR)

            # Save the dimensions of the needle image
            self.needle_w = self.needle_img.shape[1]
            self.needle_h = self.needle_img.shape[0]

        self.method = method

    def find(self, haystack_img, threshold=0.75, max_results=10):
        # run the OpenCV algorithm
        result = cv.matchTemplate(haystack_img, self.needle_img, self.method)

        # Get the all the positions from the match result that exceed our threshold
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))
        # print(locations)

        if not locations:
            return np.array([], dtype=np.int32).reshape(0, 4)

        # First we need to create the list of [x, y, w, h] rectangles
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            # Add every box to the list twice in order to retain single (non-overlapping) boxes
            rectangles.append(rect)
            rectangles.append(rect)

        # "Relative difference between sides of the rectangles to merge them into a group."
        rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
        # print(rectangles)

        if len(rectangles) > max_results:
            rectangles = rectangles[:max_results]

        return rectangles

    def get_click_points(self, rectangles):
        points = []

        # Loop over all the rectangles
        for (x, y, w, h) in rectangles:
            # Determine the center position
            center_x = x + int(w / 2)
            center_y = y + int(h / 2)
            # Save the points
            points.append((center_x, center_y))

        return points

    def draw_rectangles(self, haystack_img, rectangles):
        line_color = (0, 255, 0)
        line_type = cv.LINE_4

        for (x, y, w, h) in rectangles:
            # Determine the box position
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # Draw the box
            cv.rectangle(haystack_img, top_left, bottom_right, color=line_color,
                         lineType=line_type, thickness=2)
        return haystack_img

    def draw_crosshairs(self, haystack_img, points):
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS

        for (center_x, center_y) in points:
            cv.drawMarker(haystack_img, (center_x, center_y), marker_color, marker_type)

        return points

    def init_controls_gui(self):
        cv.namedWindow(self.TRACKBAR_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.TRACKBAR_WINDOW, 350, 700)

        # required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass

        # create trackbars for bracketing.
        # OpenCV scale for HSV is H: 0-179, S: 0-255, V: 0-255
        cv.createTrackbar('HMin', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('HMax', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        # Set default value for Max HSV trackbars
        cv.setTrackbarPos('HMax', self.TRACKBAR_WINDOW, 179)
        cv.setTrackbarPos('SMax', self.TRACKBAR_WINDOW, 255)
        cv.setTrackbarPos('VMax', self.TRACKBAR_WINDOW, 255)

        # trackbars for increasing/decreasing saturation and value
        cv.createTrackbar('SAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('SSub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VSub', self.TRACKBAR_WINDOW, 0, 255, nothing)

        # returns an HSV filter object based on the control GUI values

    def get_hsv_filter_from_controls(self):
        # Get current positions of all trackbars
        hsv_filter = hsvFilter()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', self.TRACKBAR_WINDOW)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', self.TRACKBAR_WINDOW)
        hsv_filter.vMin = cv.getTrackbarPos('VMin', self.TRACKBAR_WINDOW)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', self.TRACKBAR_WINDOW)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', self.TRACKBAR_WINDOW)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', self.TRACKBAR_WINDOW)
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', self.TRACKBAR_WINDOW)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', self.TRACKBAR_WINDOW)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', self.TRACKBAR_WINDOW)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', self.TRACKBAR_WINDOW)
        return hsv_filter

    def apply_hsv_filter(self, original_image, hsv_filter=None):
        hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

        # if not given a defined filter, use filter values from gui
        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        # add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, -hsv_filter.vSub)
        hsv = cv.merge([h, s, v])

        # apply min and max HSV values to display
        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])

        # apply thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # convert back to BGR for imshow() display
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img

    def shift_channel(self, c, amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c
