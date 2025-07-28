import streamlit as st
import numpy as np
import cv2
from skimage.morphology import skeletonize, closing, remove_small_objects
from skimage.filters import threshold_otsu
from skimage import io
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Airflow Vector Comparison (Clean Skeleton)")

st.markdown("""
Upload two black‑background, white‑line images (prediction & ground truth).  
This app will:
1. Binarize with Otsu threshold  
2. Remove small specks & close gaps  
3. Skeletonize the clean mask  
4. Extract local tangent angles (Sobel)  
5. Mask overlaps & compute Mean Angular Error (°)  
6. Show skeletons + angular error heatmap  
""")

pred_file = st.file_uploader("1. Upload Prediction Image", type=["png","jpg","jpeg"])
gt_file   = st.file_uploader("2. Upload Ground Truth Image", type=["png","jpg","jpeg"])

def preprocess_clean(img):
    """
    1) Convert to grayscale float in [0,1]
    2) Binarize with Otsu
    3) Remove small objects
    4) Close small gaps
    5) Skeletonize
    """
    arr = img
    # RGB/RGBA → Gray
    if arr.ndim == 3:
        # drop alpha if present
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        # already single channel
        gray = arr.astype(np.float32) / 255.0

    # Otsu threshold: white lines = True
    th = threshold_otsu(gray)
    binary = gray < th

    # Remove tiny specks
    clean = remove_small_objects(binary, min_size=100)

    # Close small gaps (3×3 square)
    closed = closing(clean, selem=np.ones((3,3), dtype=bool))

    # Skeletonize
    skel = skeletonize(closed)
    return skel.astype(np.uint8)

def tangent_angles(skel):
    """Compute local tangent angle (radians) via Sobel gradients."""
    sx = cv2.Sobel(skel.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(skel.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
    return np.arctan2(sy, sx)

def angular_error_deg(a_pred, a_gt):
    """Minimal wrapped difference, in degrees."""
    diff = np.abs(a_pred - a_gt)
    diff = np.minimum(diff, 2*np.pi - diff)
    return np.degrees(diff)

if pred_file and gt_file:
    # Load images as numpy arrays
    pred_img = io.imread(pred_file)
    gt_img   = io.imread(gt_file)

    # Resize prediction → ground truth if needed
    if pred_img.shape[:2] != gt_img.shape[:2]:
        st.warning("Resizing prediction to match ground truth shape.")
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    # Clean skeletons
    skel_pred = preprocess_clean(pred_img)
    skel_gt   = preprocess_clean(gt_img)

    # Compute local tangent maps
    ang_pred = tangent_angles(skel_pred)
    ang_gt   = tangent_angles(skel_gt)

    # Mask where both skeletons exist
    mask = (skel_pred == 1) & (skel_gt == 1)

    if not mask.any():
        st.error("No overlapping skeleton pixels found.")
    else:
        # Compute angular error map & MAE
        err_map = angular_error_deg(ang_pred, ang_gt)
        mae     = err_map[mask].mean()

        st.subheader(f"Mean Angular Error: {mae:.2f}°")

        # Display skeletons side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(skel_pred * 255,
                     caption="Prediction Skeleton (clean)",
                     use_column_width=True)
        with col2:
            st.image(skel_gt * 255,
                     caption="Ground Truth Skeleton (clean)",
                     use_column_width=True)

        # Plot error heatmap
        fig, ax = plt.subplots(figsize=(6,4))
        disp = np.ma.masked_where(~mask, err_map)
        im = ax.imshow(disp, cmap="inferno", vmin=0, vmax=90)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Angular Error (°)")
        st.pyplot(fig)
