import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


IMAGE_SIZE = 224

model = load_model("./car-license-plate-detection/model/model.keras")

def prepare_input(img):
    img = Image.open(img).convert("RGB")   # PIL
    
    img = np.array(img)   # 3D (h, w, c)
    org_height, org_width = img.shape[:2]  # original size

    img1 = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # resize
    img1 = (img1 / 255).astype("float32")  # scaling

    X_predict = np.expand_dims(img1, axis=0)   # X_predict = np.array([img])
    print(X_predict.shape)   # 4D (1, 224, 224, c)
    
    return (X_predict, org_height, org_width, img)    

def predict(data):
    X_predict, org_height, org_width, img = data 
    y_predict = model.predict(X_predict)   # 2D (1, 4)
    
    # bbox theo image 224x224
    bbox_predict = [int(coord * IMAGE_SIZE) for coord in y_predict[0]]

    # bbox theo image gốc
    xmin = int((bbox_predict[0] * (org_width/IMAGE_SIZE)))
    ymin = int((bbox_predict[1] * (org_height/IMAGE_SIZE)))
    xmax = int((bbox_predict[2] * (org_width/IMAGE_SIZE)))
    ymax = int((bbox_predict[3] * (org_height/IMAGE_SIZE)))

    img_copy = img.copy()
    cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return img_copy

# Streamlit UI
st.title("Car Plate Detection App 🚗📸")

img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])  # binary

if img is not None:
    data = prepare_input(img)
    
    img_org_with_box = predict(data)
    
    st.image([img, img_org_with_box], caption=["Original Image", "Detected Car Plate"], width=300)
