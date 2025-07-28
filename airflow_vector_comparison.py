import streamlit as st
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage import io
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Airflow Vector Comparison")

st.markdown("""
Upload two black‑background, white‑line images (prediction & ground truth).  
This app will skeletonize, extract local tangents, mask overlaps, and compute Mean Angular Error (°).
""")

pred_file = st.file_uploader("1. Upload Prediction Image", type=["png","jpg","jpeg"])
gt_file   = st.file_uploader("2. Upload Ground Truth Image", type=["png","jpg","jpeg"])

def preprocess(img):
    """
    - Ensures a single grayscale channel in [0,1].
    - Binary threshold & skeletonize.
    """
    # --- convert to BGR uint8 if needed ---
    arr = img
    if arr.ndim == 3:
        # skimage.io.imread might give RGB or RGBA
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        # now RGB → gray
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        # already 2D
        gray = arr

    # float in [0,1]
    if gray.dtype != np.float32 and gray.dtype != np.float64:
        gray = gray.astype(np.float32) / 255.0

    # binary: white lines (<0.5 = line on dark bg)
    binary = gray < 0.5
    skel   = skeletonize(binary)
    return skel.astype(np.uint8)

def tangent_angles(skel):
    # Compute Sobel gradients on skeleton
    sx = cv2.Sobel(skel.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(skel.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
    angles = np.arctan2(sy, sx)
    return angles

def angular_error_deg(a_pred, a_gt):
    diff = np.abs(a_pred - a_gt)
    diff = np.minimum(diff, 2*np.pi - diff)
    return np.degrees(diff)

if pred_file and gt_file:
    # load via skimage to get numpy array
    pred_img = io.imread(pred_file)
    gt_img   = io.imread(gt_file)

    # resize pred → gt if mismatched
    if pred_img.shape[:2] != gt_img.shape[:2]:
        st.warning("Resizing prediction to match ground truth.")
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    # preprocess → skeletons
    skel_pred = preprocess(pred_img)
    skel_gt   = preprocess(gt_img)

    # get tangent maps
    ang_pred = tangent_angles(skel_pred)
    ang_gt   = tangent_angles(skel_gt)

    # mask where both skeletons exist
    mask = (skel_pred == 1) & (skel_gt == 1)

    if not mask.any():
        st.error("No overlapping skeleton pixels found.")
    else:
        err_map = angular_error_deg(ang_pred, ang_gt)
        mae     = err_map[mask].mean()

        st.subheader(f"Mean Angular Error: {mae:.2f}°")

        # show skeletons
        c1, c2 = st.columns(2)
        with c1: st.image(skel_pred*255, caption="Prediction Skeleton", use_column_width=True)
        with c2: st.image(skel_gt*255,   caption="Ground Truth Skeleton", use_column_width=True)

        # error heatmap
        fig, ax = plt.subplots(figsize=(6,4))
        disp = np.ma.masked_where(~mask, err_map)
        im = ax.imshow(disp, cmap="inferno", vmin=0, vmax=90)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Error (°)")
        st.pyplot(fig)
