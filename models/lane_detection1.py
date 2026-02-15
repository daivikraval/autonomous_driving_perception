import cv2
import numpy as np

class LaneSmoother:
    def __init__(self, n=5):
        self.n = n  # number of frames to average over
        self.left_fits = []
        self.right_fits = []

    def add_fit(self, left_fit, right_fit):
        if left_fit is not None and right_fit is not None:
            self.left_fits.append(left_fit)
            self.right_fits.append(right_fit)
            if len(self.left_fits) > self.n:
                self.left_fits.pop(0)
                self.right_fits.pop(0)

    def get_smoothed_fit(self):
        if not self.left_fits or not self.right_fits:
            return None, None
        left_avg = np.mean(self.left_fits, axis=0)
        right_avg = np.mean(self.right_fits, axis=0)
        return left_avg, right_avg

def combined_threshold(img):
    # Sobel x gradient threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    # S channel threshold (HLS)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 180) & (s_channel <= 255)] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 1)
    masked = cv2.bitwise_and(img, mask)
    return masked

def warp_perspective(img):
    h, w = img.shape[:2]
    src = np.float32([
        [w*0.43, h*0.65],
        [w*0.58, h*0.65],
        [w*0.1, h*0.95],
        [w*0.9, h*0.95]
    ])
    dst = np.float32([
        [w*0.2, 0],
        [w*0.8, 0],
        [w*0.2, h],
        [w*0.8, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    minpix = 50
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    if len(left_lane_inds) == 0 or len(right_lane_inds) == 0:
        return None, None, None, None

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped, lane_smoother):
    pixels = find_lane_pixels(binary_warped)
    if pixels is None or any(len(p) == 0 for p in pixels):
        return None

    leftx, lefty, rightx, righty = pixels

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    lane_smoother.add_fit(left_fit, right_fit)
    smooth_left_fit, smooth_right_fit = lane_smoother.get_smoothed_fit()
    if smooth_left_fit is None or smooth_right_fit is None:
        return None

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = smooth_left_fit[0]*ploty**2 + smooth_left_fit[1]*ploty + smooth_left_fit[2]
    right_fitx = smooth_right_fit[0]*ploty**2 + smooth_right_fit[1]*ploty + smooth_right_fit[2]

    return ploty, left_fitx, right_fitx

def draw_lane(original_img, binary_warped, left_fitx, right_fitx, ploty, Minv):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int32([pts]), (0,255,0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result

lane_smoother = LaneSmoother(n=5)

def detect_lanes(frame):
    combined_binary = combined_threshold(frame)
    height, width = combined_binary.shape

    vertices = np.array([[
        (0, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width, height)
    ]], dtype=np.int32)

    masked = region_of_interest(combined_binary, vertices)
    masked = (masked * 255).astype(np.uint8)

    warped, M, Minv = warp_perspective(masked)

    lane_detection = fit_polynomial(warped, lane_smoother)

    if not lane_detection:
        return frame

    ploty, left_fitx, right_fitx = lane_detection

    if ploty is None or left_fitx is None or right_fitx is None:
        return frame

    result = draw_lane(frame, warped, left_fitx, right_fitx, ploty, Minv)

    return result
