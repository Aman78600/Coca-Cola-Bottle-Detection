import streamlit as st
import cv2
import numpy as np
import pickle
import os
from roboflow import Roboflow

# Load model details from pickle file
def load_roboflow_model():
    try:
        with open(os.path.join('model_save', 'model_details.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        
        # Recreate the model using saved information
        rf = Roboflow(api_key=model_info['api_key'])
        project = rf.workspace().project(model_info['project_name'])
        loaded_model = project.version(model_info['version']).model
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main Streamlit app
def main():
    st.title('Coca Cola Bottle Detection')
    
    # Load the model
    loaded_model = load_roboflow_model()
    
    if loaded_model is None:
        st.error("Could not load the model. Please check your model files.")
        return

    # Image uploader
    uploaded_image = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'png', 'jpeg'],
        accept_multiple_files=False
    )

    if uploaded_image is not None:
        # Read the uploaded image
        file_bytes = uploaded_image.read()
        
        # Convert to OpenCV image
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Make a copy for drawing
        result_image = image.copy()

        # Predict using the loaded model
        result = loaded_model.predict(uploaded_image.name, confidence=40, overlap=30).json()

        # Check if predictions exist
        if result['predictions']:
            # Draw bounding boxes
            for prediction in result['predictions']:
                # Convert from [x, y, width, height] to [x1, y1, x2, y2]
                x1 = int(prediction['x'] - prediction['width'] / 2)
                y1 = int(prediction['y'] - prediction['height'] / 2)
                x2 = int(prediction['x'] + prediction['width'] / 2)
                y2 = int(prediction['y'] + prediction['height'] / 2)

                # Draw rectangle
                cv2.rectangle(
                    result_image, 
                    (x1, y1), (x2, y2), 
                    color=(255, 0, 0), 
                    thickness=2
                )
                
                # Add label
                label = f"{prediction['class']} {prediction['confidence']:.2f}"
                cv2.putText(
                    result_image, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (255, 0, 0), 
                    2
                )

            # Convert from BGR to RGB for Streamlit display
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Display the image
            st.image(result_image_rgb, caption='Detected Coca Cola Bottles')
        
        else:
            st.warning("No bottles detected in the image.")

# Run the app
if __name__ == "__main__":
    main()