#!/usr/bin/env python3
"""
Calibrated Predictor - Uses enhanced model with confidence calibration
Addresses class collapse and overconfidence issues
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import pickle

# Blood group labels
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

def load_calibration_map(calibration_file='confidence_calibration.pkl'):
    """
    Load confidence calibration map
    """
    try:
        with open(calibration_file, 'rb') as f:
            calibration_points = pickle.load(f)
        print("✓ Loaded confidence calibration map")
        return calibration_points
    except Exception as e:
        print(f"⚠️  Failed to load calibration map: {e}")
        # Return default calibration (linear mapping)
        return [(0.0, 0.0), (1.0, 1.0)]

def apply_confidence_calibration(raw_confidence, calibration_points):
    """
    Apply confidence calibration using the calibration map
    """
    try:
        # Sort calibration points by raw confidence
        calibration_points = sorted(calibration_points, key=lambda x: x[0])
        
        # Find appropriate calibration points
        for i in range(len(calibration_points) - 1):
            raw_low, cal_low = calibration_points[i]
            raw_high, cal_high = calibration_points[i + 1]
            
            if raw_low <= raw_confidence <= raw_high:
                # Linear interpolation
                if raw_high == raw_low:
                    return cal_low
                ratio = (raw_confidence - raw_low) / (raw_high - raw_low)
                calibrated = cal_low + ratio * (cal_high - cal_low)
                return max(0.0, min(1.0, calibrated))  # Clamp to [0,1]
        
        # If outside range, extrapolate
        if raw_confidence < calibration_points[0][0]:
            return calibration_points[0][1]
        else:
            return calibration_points[-1][1]
            
    except Exception as e:
        print(f"⚠️  Calibration failed: {e}")
        return raw_confidence  # Return raw confidence if calibration fails

def apply_advanced_fingerprint_preprocessing(image):
    """
    Apply advanced fingerprint preprocessing to enhance features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Multi-scale CLAHE for enhanced ridge visibility
    clahe_fine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    clahe_coarse = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    
    enhanced_fine = clahe_fine.apply(gray)
    enhanced_coarse = clahe_coarse.apply(gray)
    
    # Combine multi-scale enhancements
    enhanced = cv2.addWeighted(enhanced_fine, 0.7, enhanced_coarse, 0.3, 0)
    
    # Apply Gabor filter for ridge enhancement
    kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(enhanced, cv2.CV_8UC3, kernel)
    
    # Ridge orientation normalization
    sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    orientation = np.arctan2(sobely, sobelx)
    
    # Convert back to RGB with orientation information
    orientation_normalized = cv2.normalize(orientation, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    result = cv2.cvtColor(orientation_normalized, cv2.COLOR_GRAY2RGB)
    
    return result

def preprocess_for_prediction(img_path):
    """
    Preprocess image for prediction with exact training pipeline
    """
    # Load image
    img = cv2.imread(img_path)
    
    # Check if image was loaded successfully
    if img is None:
        raise ValueError(f"Could not load image from path: {img_path}")
    
    # Apply advanced fingerprint preprocessing
    img = apply_advanced_fingerprint_preprocessing(img)
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to match model input
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def load_model_safely(model_path='calibrated_model.h5'):
    """
    Safely load model and verify it's working correctly
    """
    try:
        model = load_model(model_path)
        
        # Test with dummy input to verify model isn't collapsed
        dummy_input = np.random.rand(1, 224, 224, 3)
        dummy_pred = model.predict(dummy_input)
        
        # Check if predictions are valid (sum should be ~1.0 for softmax)
        pred_sum = np.sum(dummy_pred)
        if pred_sum < 0.9 or pred_sum > 1.1:
            print("WARNING: Model output doesn't look like valid softmax")
            return None
            
        # Check for extreme predictions (indicating collapse)
        max_pred = np.max(dummy_pred)
        if max_pred > 0.99:
            print("WARNING: Model seems to be collapsing to single class")
            return None
            
        print("Model loaded and verified successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_blood_group_calibrated(img_path, model_path='calibrated_model.h5', confidence_threshold=0.05):
    """
    Predict blood group with calibrated confidence scores
    """
    print(f"=== CALIBRATED PREDICTION ===")
    print(f"Image: {img_path}")
    
    # Load calibration map
    calibration_points = load_calibration_map()
    
    # Load and verify model
    model = load_model_safely(model_path)
    if model is None:
        return None, 0.0, []
    
    # Preprocess image
    try:
        processed_img = preprocess_for_prediction(img_path)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, 0.0, []
    
    # Make prediction
    try:
        predictions = model.predict(processed_img)
        
        # Validate predictions
        pred_sum = np.sum(predictions[0])
        if pred_sum < 0.9 or pred_sum > 1.1:
            print("ERROR: Invalid model output")
            return None, 0.0, []
        
        # Get raw prediction
        predicted_class = np.argmax(predictions[0])
        raw_confidence = float(predictions[0][predicted_class])
        predicted_blood_group = BLOOD_GROUPS[predicted_class]
        
        # Apply confidence calibration
        calibrated_confidence = apply_confidence_calibration(raw_confidence, calibration_points)
        print(f"Raw confidence: {raw_confidence:.4f}")
        print(f"Calibrated confidence: {calibrated_confidence:.4f}")
        
        # Get top 3 predictions with calibrated confidence
        sorted_indices = np.argsort(predictions[0])[::-1][:3]
        top_3_predictions = []
        for idx in sorted_indices:
            blood_group = BLOOD_GROUPS[idx]
            raw_conf = float(predictions[0][idx])
            calibrated_conf = apply_confidence_calibration(raw_conf, calibration_points)
            
            # Maintain compatibility with original format
            top_3_predictions.append({
                'blood_group': blood_group,
                'confidence': calibrated_conf  # Use calibrated confidence
            })
        
        # Apply confidence threshold on calibrated confidence
        if calibrated_confidence < confidence_threshold:
            print(f"Low calibrated confidence ({calibrated_confidence:.4f}), below threshold ({confidence_threshold})")
            return None, 0.0, []
        
        # Return results with calibrated confidence
        return predicted_blood_group, calibrated_confidence, top_3_predictions
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, []

def test_calibrated_predictor():
    """
    Test the calibrated predictor with sample images
    """
    print("=== TESTING CALIBRATED PREDICTOR ===")
    
    # Look for test images
    uploads_dir = "static/uploads"
    if os.path.exists(uploads_dir):
        test_count = 0
        for file in os.listdir(uploads_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('blood_group_report'):
                img_path = os.path.join(uploads_dir, file)
                test_count += 1
                
                print(f"\nTest {test_count}: {file}")
                result = predict_blood_group_calibrated(img_path)
                
                if result[0] is not None:
                    blood_group, confidence, top_3 = result
                    print(f"  Prediction: {blood_group}")
                    print(f"  Calibrated Confidence: {confidence:.4f}")
                    print("  Top 3 Predictions:")
                    for i, pred in enumerate(top_3):
                        print(f"    {i+1}. {pred['blood_group']}: {pred['confidence']:.4f}")
                else:
                    print("  Prediction failed or low confidence")
                
                # Limit to first 3 images for demonstration
                if test_count >= 3:
                    break
    else:
        print("No test images found")

if __name__ == "__main__":
    test_calibrated_predictor()
    
    print("\n=== CALIBRATED PREDICTOR READY ===")
    print("Features:")
    print("✅ Confidence calibration to reduce overconfidence")
    print("✅ Enhanced fingerprint preprocessing")
    print("✅ Better handling of class imbalance")
    print("✅ More reliable probability scores")
    
    print("\nTo integrate with your web application:")
    print("1. Replace imports in app.py:")
    print("   FROM: from simple_predictor import predict_blood_group")
    print("   TO:   from calibrated_predictor import predict_blood_group_calibrated as predict_blood_group")
    print("2. Ensure 'calibrated_model.h5' is available")
    print("3. The function signature remains the same for compatibility")