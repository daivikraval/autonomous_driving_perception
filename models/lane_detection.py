import cv2
import numpy as np
from ultralytics import YOLO

# Smoothing class for lane polynomial coefficients using EMA
class StableLaneSmoother:
    def __init__(self, alpha=0.3):
        # Initialize smoothed coefficients for left and right lanes
        self.left_fit = None
        self.right_fit = None
        self.alpha = alpha  # EMA smoothing factor (0 < alpha < 1)

    def update(self, left_fit_new, right_fit_new):
        # Update left lane polynomial using EMA, or assign if not initialized
        if left_fit_new is not None:
            if self.left_fit is None:
                self.left_fit = left_fit_new
            else:
                self.left_fit = self.alpha * left_fit_new + (1 - self.alpha) * self.left_fit

        # Update right lane polynomial similarly
        if right_fit_new is not None:
            if self.right_fit is None:
                self.right_fit = right_fit_new
            else:
                self.right_fit = self.alpha * right_fit_new + (1 - self.alpha) * self.right_fit

        return self.left_fit, self.right_fit

# Global lane smoother instance
lane_smoother = StableLaneSmoother(alpha=0.3)

def region_of_interest(img, vertices):
    # Mask image outside the ROI defined by vertices
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lane_curves(img, left_fit, right_fit, y_min, y_max, color=(0,255,0)):
    # Draw lane polynomial curves over the image
    plot_y = np.linspace(y_min, y_max, num=y_max-y_min)

    # Calculate x coordinates for left lane curve if fit exists
    if left_fit is not None:
        left_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    else:
        left_x = None

    # Calculate x coordinates for right lane curve if fit exists
    if right_fit is not None:
        right_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]
    else:
        right_x = None

    # If either polynomial fit is missing, skip drawing
    if left_x is None or right_x is None:
        return img

    # Assemble points for polygon connecting left/right lanes
    pts_left = np.array([np.transpose(np.vstack([left_x, plot_y]))], dtype=np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))], dtype=np.int32)
    pts = np.hstack((pts_left, pts_right))

    # Overlay filled polygon and blend with input image
    overlay = np.zeros_like(img)
    cv2.fillPoly(overlay, [pts], color)
    combined = cv2.addWeighted(img, 0.9, overlay, 0.4, 0)
    return combined

def pipeline(image):
    # Main lane detection pipeline for a video/image frame
    height, width = image.shape[:2]
    roi_vertices = [(0, height), (width//2, height//2), (width, height)]  # Define ROI triangle

    # Convert image to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Canny edge detector

    # Apply ROI mask to edge image
    cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Detect line segments with Hough Transform
    lines = cv2.HoughLinesP(
        cropped,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=80,
        maxLineGap=30
    )

    # Lists to hold lane line endpoints (left and right)
    left_x, left_y, right_x, right_y = [], [], [], []

    # Process each detected line for lane classification
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # Skip vertical lines
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.4:
                    continue  # Ignore near-horizontal lines
                if slope < 0:
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])
                else:
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])

    # Vertical range for drawing lane polygon
    y_min = int(height * 0.65)
    y_max = height

    left_fit = None
    right_fit = None

    # Fit quadratic polynomial to left and right lane points if enough detected
    if len(left_x) > 10 and len(left_y) > 10:
        left_fit = np.polyfit(left_y, left_x, 2)
    if len(right_x) > 10 and len(right_y) > 10:
        right_fit = np.polyfit(right_y, right_x, 2)

    # Apply smoothing to lane polynomial parameters
    left_fit_smooth, right_fit_smooth = lane_smoother.update(left_fit, right_fit)

    # Draw the detected/smoothed lane curves on the input image
    result = draw_lane_curves(image, left_fit_smooth, right_fit_smooth, y_min, y_max)

    return result

def estimate_distance(bbox_width):
    # Rough estimation of distance to detected object using basic pinhole camera formula
    focal_length = 1000      # Example focal length (pixels)
    known_width = 2.0        # Known width of car (meters)
    if bbox_width == 0:
        return float('inf')  # Avoid division by zero
    distance = (known_width * focal_length) / bbox_width
    return distance

# Initialize YOLO model for object detection (cars)
model = YOLO('yolov8n.pt')

def detect_lanes(frame):
    # Lane detection plus car localization and distance estimation
    resized_frame = cv2.resize(frame, (1280, 720))    # Resize for consistency
    lane_frame = pipeline(resized_frame)              # Draw lanes on frame

    # Run YOLO detection
    results = model(resized_frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])   # Bounding box coordinates
            conf = float(box.conf[0])                 # Confidence score
            cls = int(box.cls[0])                     # Class index
            # Check if detected object is a car with high confidence
            if model.names[cls] == 'car' and conf >= 0.5:
                label = f'{model.names[cls]} {conf:.2f}'
                # Draw rectangle and label for car
                cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(lane_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                bbox_width = x2 - x1                   # Width of bounding box
                distance = estimate_distance(bbox_width)   # Estimate distance
                distance_label = f'Distance: {distance:.2f}m'
                cv2.putText(lane_frame, distance_label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return lane_frame
