import streamlit as st
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Airflow Vector Comparison")

st.markdown("""
Upload two black‑background, white‑line images:
1. **Prediction**  
2. **Ground Truth**  

This app will:
- Skeletonize the white lines
- Compute local tangent angles (Sobel)
- Mask to where both have lines
- Calculate Mean Angular Error (°)
- Show angular error heatmap
""")

pred_file = st.file_uploader("1. Upload Prediction Image", type=["png","jpg","jpeg"])
gt_file   = st.file_uploader("2. Upload Ground Truth Image", type=["png","jpg","jpeg"])

def preprocess(img):
    # To binary skeleton
    gray   = rgb2gray(img)
    binary = gray < 0.5           # assume white lines on black
    skel   = skeletonize(binary)
    return skel.astype(np.uint8)

def tangent_angles(skel):
    # Sobel on skeleton
    sx = cv2.Sobel(skel.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(skel.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
    angles = np.arctan2(sy, sx)   # radians in [-π, π]
    mag    = np.hypot(sx, sy)
    return angles, mag

def angular_error_deg(a_pred, a_gt):
    diff = np.abs(a_pred - a_gt)
    diff = np.minimum(diff, 2*np.pi - diff)
    return np.degrees(diff)

if pred_file and gt_file:
    # Read
    pred_img = io.imread(pred_file)
    gt_img   = io.imread(gt_file)

    # match sizes
    if pred_img.shape[:2] != gt_img.shape[:2]:
        st.warning("Resizing prediction → ground truth shape")
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    # preprocess
    skel_pred = preprocess(pred_img)
    skel_gt   = preprocess(gt_img)

    # angles & mags
    ang_pred, mag_pred = tangent_angles(skel_pred)
    ang_gt,   mag_gt   = tangent_angles(skel_gt)

    # mask: both skeletons present
    mask = (skel_pred==1) & (skel_gt==1)

    if mask.sum() == 0:
        st.error("No overlapping skeleton pixels found.")
    else:
        # compute error
        err_map = angular_error_deg(ang_pred, ang_gt)
        mae      = err_map[mask].mean()

        # Display MAE
        st.subheader(f"Mean Angular Error: {mae:.2f}°")

        # Show skeletons side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(skel_pred*255, caption="Prediction Skeleton", use_column_width=True)
        with col2:
            st.image(skel_gt*255, caption="Ground Truth Skeleton", use_column_width=True)

        # Plot error heatmap
        fig, ax = plt.subplots(figsize=(6,4))
        # mask out where no skeleton overlap
        disp = np.ma.masked_where(~mask, err_map)
        im = ax.imshow(disp, cmap="inferno", vmin=0, vmax=90)
        ax.set_title("Angular Error (°)")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Error (degrees)")
        st.pyplot(fig)
