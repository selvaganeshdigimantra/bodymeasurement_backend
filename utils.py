import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# Dineshraj
def get_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:

        return None, "Image not found"
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None, "No landmarks detected"
    return results.pose_landmarks.landmark, image.shape

def get_pixel_distance(p1, p2, image_size):
    h, w = image_size[:2]
    return np.linalg.norm(np.array([p1.x*w, p1.y*h]) - np.array([p2.x*w, p2.y*h]))



def measure_front_dimensions(front_image_path, person_height_cm):
    landmarks, shape = get_pose_landmarks(front_image_path)
    if landmarks is None:
        return None, shape

    h, w = shape[:2]

    # Pixels per cm using height (nose → ankle midpoint)
    top = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    ankle_mid = ((left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2)

    height_px = np.linalg.norm(
        np.array([top.x * w, top.y * h]) - np.array([ankle_mid[0] * w, ankle_mid[1] * h])
    )
    pixels_per_cm = height_px / person_height_cm

    def width(p1, p2):
        return get_pixel_distance(landmarks[p1], landmarks[p2], shape) / pixels_per_cm

    # --- Core widths ---
    shoulder_width = width(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    chest_width = shoulder_width * 0.8   # approximate ratio
    hip_width = width(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)

    # --- Front length: shoulder → hip ---
    front_length_px = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h -
                          landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h)
    front_length_cm = front_length_px / pixels_per_cm

    # --- Outseam: hip → knee → ankle (average both sides) ---
    def leg_outseam(side):
        hip = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP")].x * w,
                        landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP")].y * h])
        knee = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE")].x * w,
                         landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE")].y * h])
        ankle = np.array([landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE")].x * w,
                          landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE")].y * h])
        return np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle)

    outseam_px = (leg_outseam("LEFT") + leg_outseam("RIGHT")) / 2
    outseam_cm = outseam_px / pixels_per_cm + 5

    # --- Better crotch approximation for inseam ---
    hip_mid = (
        (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2,
        (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2,
    )
    knee_mid = (
        (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x) / 2,
        (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2,
    )

    # Crotch point ~20% from knees → hips
    crotch_x = knee_mid[0] + 0.7 * (hip_mid[0] - knee_mid[0])
    crotch_y = knee_mid[1] + 0.7 * (hip_mid[1] - knee_mid[1])

    # Inseam: crotch → ankle midpoint
    inseam_length = np.linalg.norm(
        np.array([crotch_x * w, crotch_y * h]) -
        np.array([ankle_mid[0] * w, ankle_mid[1] * h])
    ) / pixels_per_cm

    return {
        "shoulder_width_cm": round(shoulder_width, 2),
        "chest_width_cm": round(chest_width, 2),
        "hip_width_cm": round(hip_width, 2),
        "front_length_cm": round(front_length_cm, 2),
        "inseam_length_cm": round(inseam_length, 2),
        "outseam_length_cm": round(outseam_cm, 2),
        "pixels_per_cm": pixels_per_cm
    }, None






def measure_depths(side_image_path, pixels_per_cm):
    landmarks, shape = get_pose_landmarks(side_image_path)
    if landmarks is None:
        return None, shape
    h, w = shape[:2]

    def depth(p_front, p_back):
        return get_pixel_distance(landmarks[p_front], landmarks[p_back], shape) / pixels_per_cm

    # Chest depth: shoulder front-back approximation
    chest_depth = depth(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER) * 0.5

    # Waist depth: hip midpoint thickness
    waist_depth = depth(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP) * 0.55

    # Hip depth: hip thickness lower section
    hip_depth = depth(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP) * 0.65

    return {
        "chest_depth_cm": round(chest_depth, 2),
        "waist_depth_cm": round(waist_depth, 2),
        "hip_depth_cm": round(hip_depth, 2)
    }, None


def combine_measurements(front_data, side_data, gender):
    # Direct anthropometric ratio for chest/bust (based on shoulder width)
    chest_circ = front_data["shoulder_width_cm"] * 2.3

    # Waist and hip using ellipse approximation
    waist_circ = estimate_circumference(front_data["hip_width_cm"] * 0.95,
                                        side_data["waist_depth_cm"])
    hip_circ = estimate_circumference(front_data["hip_width_cm"],
                                      side_data["hip_depth_cm"])

    result = {
        "shoulder_width_cm": round(front_data["shoulder_width_cm"], 2),
        "waist_circumference_cm": round(waist_circ, 2),
        "hip_circumference_cm": round(hip_circ, 2),
        "front_length_cm": round(front_data["front_length_cm"], 2),
        "inseam_length_cm": round(front_data["inseam_length_cm"], 2),
        "outseam_length_cm": round(front_data["outseam_length_cm"], 2)
    }

    # ✅ Add gender-specific chest/bust
    if gender.lower() == "women":
        result["bust_circumference_cm"] = round(chest_circ, 2)
    else:
        result["chest_circumference_cm"] = round(chest_circ, 2)

    return result


def estimate_circumference(width_cm, depth_cm):
    """
    Approximate ellipse perimeter (Ramanujan’s formula).
    width_cm = horizontal diameter
    depth_cm = sagittal diameter
    """
    return np.pi * (3*(width_cm + depth_cm) -
                    np.sqrt((3*width_cm + depth_cm) * (width_cm + 3*depth_cm)))

