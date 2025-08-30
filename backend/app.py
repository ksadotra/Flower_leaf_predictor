import os
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64

# --- Initialize Flask App ---
app = Flask(__name__)
# Allow requests specifically from your GitHub Pages site
CORS(app, resources={r"/predict": {"origins": "https://flower-leaf-predictor-frontend.onrender.com"}}) # Enable Cross-Origin Resource Sharing

# --- Configuration ---
MODEL_FILE = 'rf (1).pkl'
SCALER_FILE = 'scaler (1).pkl'
CONFIDENCE_THRESHOLD = 25.0

# --- CRITICAL: Feature Extraction Functions (MUST BE IDENTICAL TO TRAINING) ---
# NOTE: These complex feature extraction functions remain unchanged as they are essential for the model.
def extract_features_no_segmentation(image):
    all_features = {}
    h, w, _ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    sobel_edges = cv2.magnitude(cv2.Sobel(blurred_image,cv2.CV_64F,1,0,ksize=3), cv2.Sobel(blurred_image,cv2.CV_64F,0,1,ksize=3))
    adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    dilated_veins = cv2.dilate(adaptive_thresh, np.ones((3,3),np.uint8), iterations=1)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r, _ = cv2.mean(image)
    h_mean, s_mean, v_mean, _ = cv2.mean(hsv_image)
    _, bgr_std = cv2.meanStdDev(image)
    _, hsv_std = cv2.meanStdDev(hsv_image)
    all_features.update({'b_mean':b,'g_mean':g,'r_mean':r,'b_std':bgr_std[0][0],'g_std':bgr_std[1][0],'r_std':bgr_std[2][0],'h_mean':h_mean,'s_mean':s_mean,'v_mean':v_mean,'h_std':hsv_std[0][0],'s_std':hsv_std[1][0],'v_std':hsv_std[2][0]})
    glcm = graycomatrix(gray_image, distances=[1,2,3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    all_features.update({'contrast':np.mean(graycoprops(glcm,'contrast')), 'dissimilarity':np.mean(graycoprops(glcm,'dissimilarity')), 'homogeneity':np.mean(graycoprops(glcm,'homogeneity')), 'energy':np.mean(graycoprops(glcm,'energy')), 'correlation':np.mean(graycoprops(glcm,'correlation')), 'ASM':np.mean(graycoprops(glcm,'ASM'))})
    main_contour = np.array([[[0,0]], [[0,h-1]], [[w-1,h-1]], [[w-1,0]]])
    shape_moments = cv2.moments(main_contour)
    hu_moments = cv2.HuMoments(shape_moments)
    for i in range(7):
        hu = hu_moments[i][0]
        all_features[f'vein_hu_moment_{i+1}'] = -np.sign(hu) * np.log10(np.abs(hu)) if hu != 0 else 0
    total_pixel_count = w * h
    edge_pixel_count = np.count_nonzero(sobel_edges)
    vein_pixel_count = cv2.countNonZero(dilated_veins)
    all_features['edge_density'] = edge_pixel_count / total_pixel_count
    all_features['vein_density'] = vein_pixel_count / total_pixel_count
    area, perimeter = float(w*h), float(2*(w+h))
    all_features.update({'aspect_ratio':float(w)/h if h!=0 else 0, 'rectangularity':1.0, 'circularity':(4*np.pi*area)/(perimeter**2) if perimeter!=0 else 0, 'solidity':1.0, 'perimeter_area_ratio':perimeter/area if area!=0 else 0, 'centroid_x':0.5, 'centroid_y':0.5})
    return all_features

def extract_leaf_features_tiered(image_data):
    try:
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image data: {e}")
        return None

    final_mask = None
    image_to_process = original_image
    
    hsv_image_full = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40]); upper_green = np.array([95, 255, 255])
    green_mask_full = cv2.inRange(hsv_image_full, lower_green, upper_green)
    green_pixel_count = cv2.countNonZero(green_mask_full)
    total_pixels = original_image.shape[0] * original_image.shape[1]
    green_percentage = green_pixel_count / total_pixels

    if 0.05 < green_percentage < 0.50:
        contours_green, _ = cv2.findContours(cv2.morphologyEx(green_mask_full, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_green:
            main_contour = max(contours_green, key=cv2.contourArea)
            final_mask = np.zeros_like(green_mask_full)
            cv2.drawContours(final_mask, [main_contour], -1, 255, -1)
    
    if final_mask is None:
        target_size = 600
        h, w, _ = original_image.shape
        scale = target_size / max(h, w)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            image_to_process = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image_to_process = original_image

        hsv_image = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 60, 60]); upper_yellow = np.array([35, 255, 255])
        lower_brown = np.array([5, 40, 20]); upper_brown = np.array([25, 200, 200])
        lower_black = np.array([0, 0, 0]); upper_black = np.array([180, 255, 50])

        resized_green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
        black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

        hint_mask = cv2.bitwise_or(resized_green_mask, yellow_mask)
        hint_mask = cv2.bitwise_or(hint_mask, brown_mask)
        hint_mask = cv2.bitwise_or(hint_mask, black_mask)
        
        contours_hint, _ = cv2.findContours(cv2.morphologyEx(hint_mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_hint:
            main_contour_hint = max(contours_hint, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour_hint)
            rect = (x, y, w, h)
            mask = np.zeros(image_to_process.shape[:2], np.uint8)
            bgdModel, fgdModel = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(image_to_process, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                grabcut_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
                final_mask = cv2.bitwise_and(grabcut_mask, hint_mask)
            except: final_mask = None
        else: final_mask = None

    if final_mask is None or cv2.countNonZero(final_mask) < 500:
        features = extract_features_no_segmentation(original_image)
        return features

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(final_mask)
    cv2.drawContours(final_mask, [main_contour], -1, 255, -1)
    segmented_image = cv2.bitwise_and(image_to_process, image_to_process, mask=final_mask)
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    all_features = {}
    hsv_leaf_only = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
    b, g, r, _ = cv2.mean(segmented_image, mask=final_mask)
    h, s, v, _ = cv2.mean(hsv_leaf_only, mask=final_mask)
    _, bgr_std = cv2.meanStdDev(segmented_image, mask=final_mask)
    _, hsv_std = cv2.meanStdDev(hsv_leaf_only, mask=final_mask)
    all_features.update({'b_mean':b,'g_mean':g,'r_mean':r,'b_std':bgr_std[0][0],'g_std':bgr_std[1][0],'r_std':bgr_std[2][0],'h_mean':h,'s_mean':s,'v_mean':v,'h_std':hsv_std[0][0],'s_std':hsv_std[1][0],'v_std':hsv_std[2][0]})
    glcm = graycomatrix(gray_image, distances=[1,2,3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    all_features.update({'contrast':np.mean(graycoprops(glcm,'contrast')), 'dissimilarity':np.mean(graycoprops(glcm,'dissimilarity')), 'homogeneity':np.mean(graycoprops(glcm,'homogeneity')), 'energy':np.mean(graycoprops(glcm,'energy')), 'correlation':np.mean(graycoprops(glcm,'correlation')), 'ASM':np.mean(graycoprops(glcm,'ASM'))})
    dilated_veins = cv2.dilate(cv2.adaptiveThreshold(cv2.GaussianBlur(gray_image, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2), np.ones((3,3),np.uint8))
    vein_contours, _ = cv2.findContours(dilated_veins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_moments = cv2.moments(main_contour)
    if vein_contours:
        vein_contour = max(vein_contours, key=cv2.contourArea)
        shape_moments = cv2.moments(vein_contour)
    hu_moments = cv2.HuMoments(shape_moments)
    for i in range(7):
        hu = hu_moments[i][0]
        all_features[f'vein_hu_moment_{i+1}'] = -np.sign(hu) * np.log10(np.abs(hu)) if hu != 0 else 0
    leaf_pixel_count = cv2.countNonZero(final_mask)
    vein_pixel_count = cv2.countNonZero(dilated_veins)
    all_features['vein_density'] = vein_pixel_count / leaf_pixel_count if leaf_pixel_count > 0 else 0
    sobel_edges = cv2.magnitude(cv2.Sobel(gray_image,cv2.CV_64F,1,0), cv2.Sobel(gray_image,cv2.CV_64F,0,1))
    edge_pixel_count = np.count_nonzero(sobel_edges)
    all_features['edge_density'] = edge_pixel_count / leaf_pixel_count if leaf_pixel_count > 0 else 0
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    x, y, w, h = cv2.boundingRect(main_contour)
    all_features.update({'aspect_ratio':float(w)/h if h!=0 else 0, 'rectangularity':area/(w*h) if (w*h)!=0 else 0, 'circularity':(4*np.pi*area)/(perimeter**2) if perimeter!=0 else 0})
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    all_features['solidity'] = float(area)/hull_area if hull_area!=0 else 0
    all_features['perimeter_area_ratio'] = perimeter/area if area!=0 else 0
    moments = cv2.moments(main_contour)
    cx,cy = 0.5,0.5
    if moments['m00'] != 0:
        cx = float(moments['m10']/moments['m00'])/w
        cy = float(moments['m01']/moments['m00'])/h
    all_features.update({'centroid_x':cx, 'centroid_y':cy})

    return all_features

# --- MODIFIED Prediction Function ---
def predict_leaf_disease(image_data):
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except Exception as e:
        return {"error": f"Failed to load model files: {e}"}

    # MODIFICATION: The function no longer expects a segmented image to be returned.
    features = extract_leaf_features_tiered(image_data)
    if features is None:
        return {"error": "Could not extract features from the image."}

    feature_df = pd.DataFrame([features])
    try:
        feature_df = feature_df[scaler.get_feature_names_out()]
    except Exception as e:
        return {"error": f"Feature mismatch. Ensure the training features match prediction features. Details: {e}"}

    features_scaled = scaler.transform(feature_df)
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    
    predicted_class = prediction[0]
    confidence_score = np.max(probability) * 100

    # MODIFICATION: Implement the confidence threshold logic.
    final_prediction_class = ""
    if confidence_score < CONFIDENCE_THRESHOLD:
        final_prediction_class = "Unknown Leaf"
    else:
        final_prediction_class = predicted_class.upper()

    # MODIFICATION: Create the result dictionary without confidence or segmented image.
    result = {
        "prediction": final_prediction_class
    }
    
    # MODIFICATION: The function now only returns the result dictionary.
    return result

# --- MODIFIED Flask Route ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        image_data = file.read()
        
        # MODIFICATION: The function call now expects only one return value.
        prediction_result = predict_leaf_disease(image_data)
        
        if "error" in prediction_result:
            return jsonify(prediction_result), 500
        
        # MODIFICATION: Directly return the result. No need to add segmented image.
        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
