import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define Fuzzy Variables for Linearity Classification
linearity = ctrl.Antecedent(np.arange(0, 101, 1), 'linearity')  # 0: Highly Curved, 100: Straight
lane_type = ctrl.Consequent(np.arange(0, 101, 1), 'lane_type')  # 0: Highly Curved, 100: Straight Lane

# Membership Functions for Linearity
linearity['highly_curved'] = fuzz.trimf(linearity.universe, [0, 0, 40])
linearity['slightly_curved'] = fuzz.trimf(linearity.universe, [20, 50, 80])
linearity['straight'] = fuzz.trimf(linearity.universe, [60, 100, 100])

# Membership Functions for Lane Type
lane_type['highly_curved'] = fuzz.trimf(lane_type.universe, [0, 0, 40])
lane_type['slightly_curved'] = fuzz.trimf(lane_type.universe, [20, 50, 80])
lane_type['straight'] = fuzz.trimf(lane_type.universe, [60, 100, 100])

# Fuzzy Rules
rule1 = ctrl.Rule(linearity['highly_curved'], lane_type['highly_curved'])
rule2 = ctrl.Rule(linearity['slightly_curved'], lane_type['slightly_curved'])
rule3 = ctrl.Rule(linearity['straight'], lane_type['straight'])

# Fuzzy Control System
lane_type_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
lane_type_system = ctrl.ControlSystemSimulation(lane_type_ctrl)

# Function to Classify Lane Type
def classify_lane(image):
    # Preprocessing: Assume the lane is already detected in binary format (white = lane, black = background)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Fit a line to the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate linearity score
        # vx, vy represent the direction vector of the line
        angle = np.arctan2(vy, vx) * 180 / np.pi
        linearity_score = 100 - abs(angle)  # Smaller angles indicate straighter lines

        # Input linearity score into fuzzy system
        lane_type_system.input['linearity'] = linearity_score
        lane_type_system.compute()
        lane_score = lane_type_system.output['lane_type']
        
        # Determine lane type
        if lane_score < 40:
            print("Detected Lane Type: Highly Curved")
            lane_type_label = "Highly Curved"
        elif 40 <= lane_score < 80:
            print("Detected Lane Type: Slightly Curved")
            lane_type_label = "Slightly Curved"
        else:
            print("Detected Lane Type: Straight")
            lane_type_label = "Straight"
        
        return lane_type_label, image
    else:
        print("No Lane Detected!")
        return "No Lane", image

# Example Usage
image_path = 'F:\Soft computing\PROJECT - 1\Main-Lane-Detection\output_image1.jpg'  # Input a binary image with detected lanes
binary_lane_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Binary format (0 or 255)
result, processed_image = classify_lane(binary_lane_image)

# Display Result
print("Lane Classification Result:", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
