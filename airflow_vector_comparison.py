import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import (
    skeletonize,
    binary_closing,
    remove_small_objects
)
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Airflow Vector Comparison (Clean Skeleton)")

st.markdown("""
Upload **Prediction** and **Ground Truth** images (black bg, white flow lines).  
This app will:
1. Load & convert to grayscale  
2. Binarize (Otsu)  
3. Remove small specks & close gaps  
4. Skeletonize  
5. Compute local tangent angles (Sobel)  
6. Mask overlaps & report Mean Angular Error (°)  
7. Show skeletons + angular error heatmap  
""")

pred_file = st.file_uploader("1. Prediction Image", type=["png","jpg","jpeg"])
gt_file   = st.file_uploader("2. Ground Truth Image", type=["png","jpg","jpeg"])

def preprocess_clean(arr: np.ndarray) -> np.ndarray:
    """Return a 1‑px skeleton (uint8) from a uint8 RGB/RGBA or gray array."""
    # — Convert to single‑channel uint8 gray —
    if arr.ndim == 3:
        # RGBA → RGB
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    # — Normalize to [0,1] float —
    gray_f = gray.astype(np.float32) / 255.0

    # — Otsu binarization: white lines => True —
    th = threshold_otsu(gray_f)
    binary = gray_f < th

    # — Remove specks, close tiny gaps —
    clean = remove_small_objects(binary, min_size=100)
    closed = binary_closing(clean, footprint=np.ones((3,3), bool))

    # — Skeletonize and return uint8 mask —
    skel = skeletonize(closed)
    return (skel.astype(np.uint8))

def tangent_angles(skel: np.ndarray) -> np.ndarray:
    """Compute local tangent angle map (radians) via Sobel."""
    sx = cv2.Sobel(skel.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(skel.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
    return np.arctan2(sy, sx)

def angular_error_deg(a_pred: np.ndarray, a_gt: np.ndarray) -> np.ndarray:
    """Wrapped minimal difference between two angle maps, in degrees."""
    diff = np.abs(a_pred - a_gt)
    diff = np.minimum(diff, 2*np.pi - diff)
    return np.degrees(diff)

if pred_file and gt_file:
    # — Load via PIL to handle file-like objects —
    pred_arr = np.array(Image.open(pred_file))
    gt_arr   = np.array(Image.open(gt_file))

    # — Resize prediction to match ground truth if needed —
    if pred_arr.shape[:2] != gt_arr.shape[:2]:
        st.warning("Resizing prediction → ground truth size")
        pred_arr = cv2.resize(pred_arr, (gt_arr.shape[1], gt_arr.shape[0]),
                              interpolation=cv2.INTER_AREA)

    # — Preprocess to skeletons —
    skel_pred = preprocess_clean(pred_arr)
    skel_gt   = preprocess_clean(gt_arr)

    # — Compute tangent maps —
    ang_pred = tangent_angles(skel_pred)
    ang_gt   = tangent_angles(skel_gt)

    # — Mask where both skeletons exist —
    mask = (skel_pred == 1) & (skel_gt == 1)

    if not mask.any():
        st.error("No overlapping skeleton pixels found. Check your inputs!")
    else:
        err_map = angular_error_deg(ang_pred, ang_gt)
        mae     = err_map[mask].mean()

        st.subheader(f"Mean Angular Error: {mae:.2f}°")

        # — Show skeletons —
        c1, c2 = st.columns(2)
        with c1:
            st.image(skel_pred * 255,
                     caption="Prediction Skeleton",
                     use_column_width=True)
        with c2:
            st.image(skel_gt * 255,
                     caption="Ground Truth Skeleton",
                     use_column_width=True)

        # — Error heatmap —
        fig, ax = plt.subplots(figsize=(6,4))
        disp = np.ma.masked_where(~mask, err_map)
        im = ax.imshow(disp, cmap="inferno", vmin=0, vmax=90)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Angular Error (°)")
        st.pyplot(fig)
