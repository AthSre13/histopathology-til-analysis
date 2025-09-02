import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
import datetime
import zipfile
import io

import uuid #for unique file gen

# to fix protobuf compatibility issue, dont use faster c++ imp.
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({'figure.max_open_warning': 0})

try:
    from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
    from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
    from tiatoolbox.utils.misc import imread
    from tiatoolbox.utils.visualization import overlay_prediction_contours
    TIATOOLBOX_AVAILABLE = True
except ImportError as e:
    TIATOOLBOX_AVAILABLE = False
    st.error(f"TIAToolbox import error: {e}")
    st.info("Please install TIAToolbox: `pip install tiatoolbox`")

# Configure Streamlit page
st.set_page_config(
    page_title="Histopathology TIL Analysis",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state for cleanup and workflow control
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []
if 'tissue_analysis_complete' not in st.session_state:
    st.session_state.tissue_analysis_complete = False
if 'patches_info' not in st.session_state:
    st.session_state.patches_info = None
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = None

def cleanup_temp_files():
    """Clean up all temporary files and directories"""
    for temp_path in st.session_state.temp_files:
        try:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
            elif os.path.isdir(temp_path):
                shutil.rmtree(temp_path)
        except Exception as e:
            st.warning(f"Could not remove {temp_path}: {e}")
    st.session_state.temp_files = []

def process_tissue_segmentation(image_path):
    """Process image with tissue segmentation model"""
    if not TIATOOLBOX_AVAILABLE:
        st.error("TIAToolbox is not available. Please install it first.")
        return None, None
        
    with st.spinner("Running tissue segmentation..."):
        try:
            bcc_segmentor = SemanticSegmentor(
                pretrained_model="fcn_resnet50_unet-bcss",
                num_loader_workers=2,
                batch_size=1,
            )
        except Exception as e:
            st.error(f"Error loading segmentation model: {e}")
            st.info("Try restarting the app or reinstalling TIAToolbox")
            return None, None
        
        temp_output_dir = os.path.join(
            tempfile.gettempdir(),
            f"tissue_seg_{uuid.uuid4().hex}"
        )
        st.session_state.temp_files.append(temp_output_dir)
        
        output = bcc_segmentor.predict(
            [image_path],
            save_dir=temp_output_dir,
            mode="tile",
            resolution=1.0,
            units="baseline",
            patch_input_shape=[1024, 1024],
            patch_output_shape=[512, 512],
            stride_shape=[512, 512],
            device="cuda" if torch.cuda.is_available() else "cpu",
            crash_on_exception=False,
        )
        
        return output, temp_output_dir

def extract_inflammatory_patches(output, inflammatory_threshold=0.1, patch_size=(256, 256), stride=128, top_k_patches=15):
    
    import numpy as np
    import cv2
    from imageio import imread
    
    input_file, output_path = output[0]
    
    # Load prediction data
    tile_prediction_raw = np.load(output_path + ".raw.0.npy")
    tile_prediction = np.argmax(tile_prediction_raw, axis=-1)
    original_image = imread(input_file)
    
    # Get inflammatory probability map (class_id = 2)
    inflammatory_prob = tile_prediction_raw[:, :, 2]
    
    # Border region analysis
    tumor_mask = (tile_prediction == 0).astype(np.uint8)
    stroma_mask = (tile_prediction == 1).astype(np.uint8)
    
    # Use larger dilation to capture wider border regions
    tumor_dilated = cv2.dilate(tumor_mask, np.ones((30, 30), np.uint8), iterations=1)
    stroma_dilated = cv2.dilate(stroma_mask, np.ones((30, 30), np.uint8), iterations=1)
    
    border_region = (tumor_dilated & stroma_dilated).astype(bool)
    
    # Enhance inflammation at border
    inflammatory_prob_enhanced = inflammatory_prob.copy()
    inflammatory_prob_enhanced[border_region] *= 2.0
    
    # Extract patches
    patches_info = []
    h, w = inflammatory_prob_enhanced.shape
    patch_h, patch_w = patch_size
    
    label_names_dict = {0: "Tumour", 1: "Stroma", 2: "Inflammatory", 3: "Necrosis", 4: "Others"}
    
    for y in range(0, h - patch_h + 1, stride):
        for x in range(0, w - patch_w + 1, stride):
            inflam_patch = inflammatory_prob_enhanced[y:y+patch_h, x:x+patch_w]
            pred_patch = tile_prediction[y:y+patch_h, x:x+patch_w]
            
            inflammatory_proportion = np.mean(inflam_patch > 0.5)
            mean_inflammatory_prob = np.mean(inflam_patch)
            
            # Calculate tissue composition
            tissue_composition = {}
            for class_id, class_name in label_names_dict.items():
                tissue_composition[class_name] = np.mean(pred_patch == class_id)
            
            necrosis_present = tissue_composition["Necrosis"] > 0.2  # relaxed threshold
            if inflammatory_proportion >= inflammatory_threshold and not necrosis_present:
                tumor_present = tissue_composition["Tumour"] > 0.05
                stroma_present = tissue_composition["Stroma"] > 0.05
                border_relevance = tumor_present and stroma_present
                
                til_relevance_score = (inflammatory_proportion * mean_inflammatory_prob) * \
                                      (1.5 if border_relevance else 1.0)
                
                patches_info.append({
                    'x': x, 'y': y,
                    'inflammatory_proportion': inflammatory_proportion,
                    'mean_inflammatory_prob': mean_inflammatory_prob,
                    'border_relevance': border_relevance,
                    'tumor_proportion': tissue_composition["Tumour"],
                    'stroma_proportion': tissue_composition["Stroma"],
                    'til_relevance_score': til_relevance_score
                })
    
    # --- Fallback mechanism if no patches survive filters ---
    if not patches_info:
        for y in range(0, h - patch_h + 1, stride):
            for x in range(0, w - patch_w + 1, stride):
                inflam_patch = inflammatory_prob[y:y+patch_h, x:x+patch_w]
                patches_info.append({
                    'x': x, 'y': y,
                    'inflammatory_proportion': np.mean(inflam_patch > 0.5),
                    'mean_inflammatory_prob': np.mean(inflam_patch),
                    'border_relevance': False,
                    'tumor_proportion': 0.0,
                    'stroma_proportion': 0.0,
                    'til_relevance_score': np.mean(inflam_patch)  # fallback score
                })
    
    # --- Normalize scores per slide ---
    max_score = max([p['til_relevance_score'] for p in patches_info], default=1e-6)
    for p in patches_info:
        p['til_relevance_score'] /= max_score
    
    # Sort and select top patches
    patches_info.sort(key=lambda x: x['til_relevance_score'], reverse=True)
    top_patches = patches_info[:top_k_patches]
    
    return {
        'patches': top_patches,
        'original_image': original_image,
        'prediction': tile_prediction,
        'inflammatory_prob': inflammatory_prob_enhanced,
        'total_inflammatory_patches': len(patches_info)
    }


def process_nucleus_segmentation(patch_image_path):
    if not TIATOOLBOX_AVAILABLE:
        return {}
    
    try:
        inst_segmentor = NucleusInstanceSegmentor(
            pretrained_model="hovernet_fast-monusac",
            num_loader_workers=1,
            num_postproc_workers=1,
            batch_size=1,
        )
    except Exception as e:
        st.warning(f"Could not load nucleus segmentation model: {e}")
        return {}
    
    temp_output_dir = os.path.join(
    tempfile.gettempdir(),
    f"nucleus_seg_{uuid.uuid4().hex}"
    )
    st.session_state.temp_files.append(temp_output_dir)
    
    tile_output = inst_segmentor.predict(
        [patch_image_path],
        save_dir=temp_output_dir,
        mode="tile",
        device="cuda" if torch.cuda.is_available() else "cpu",
        crash_on_exception=True,
    )
    
    tile_preds = joblib.load(f"{tile_output[0][1]}.dat")
    
    # Filter for lymphocytes and macrophages
    filtered_tile_preds = {}
    for inst_id, inst_info in tile_preds.items():
        if inst_info['type'] in [2, 3]:  # lymphocyte and macrophage
            filtered_tile_preds[inst_id] = inst_info
    
    return filtered_tile_preds

def create_visualizations(patches_info):
    # Define colors for tissue types
    tissue_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    tissue_cmap = ListedColormap(tissue_colors)
    label_names_dict = {0: "Tumour", 1: "Stroma", 2: "Inflammatory", 3: "Necrosis", 4: "Others"}
    
    original_image = patches_info['original_image']
    prediction = patches_info['prediction']
    inflammatory_prob = patches_info['inflammatory_prob']
    patches = patches_info['patches']
    
    # 1. Color-coded segmentation map
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.imshow(prediction, cmap=tissue_cmap, vmin=0, vmax=4)
    ax1.set_title("Tissue Segmentation Map", fontsize=16)
    ax1.axis('off')
    legend_handles = [Patch(color=tissue_colors[i], label=name) for i, name in label_names_dict.items()]
    ax1.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    
    # 2. Inflammatory heatmap
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    im = ax2.imshow(inflammatory_prob, cmap='Reds')
    ax2.set_title("Inflammatory Relevance Score Heatmap", fontsize=16)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, label='Inflammatory Relevance Score', shrink=0.8)
    plt.tight_layout()
    
    # 3. Original image with selected patches
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    ax3.imshow(original_image)
    ax3.set_title(f"Selected High-Inflammatory Patches (Top {len(patches)})", fontsize=16)
    
    for i, patch in enumerate(patches):
        rect = plt.Rectangle((patch['x'], patch['y']), 256, 256, linewidth=3, edgecolor='red', facecolor='none')
        ax3.add_patch(rect)
        ax3.text(patch['x'], patch['y']-10, str(i+1), color='red', 
                fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', alpha=0.8))
    ax3.axis('off')
    plt.tight_layout()
    
    return fig1, fig2, fig3

def save_figure_as_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

from PIL import Image
def manual_roi_selection(patches_info):
    """
    Interactive ROI selection using streamlit-drawable-canvas
    """
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.error("Please install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
        return None
    
    original_image = patches_info['original_image']
    
    st.subheader("Manual ROI Selection")
    st.markdown("Draw rectangles on the image to select regions of interest for nucleus analysis.")
    
    # Ensure image is in correct format for PIL
    if original_image.dtype != np.uint8:
        # Convert to uint8 if needed
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)
    
    # Create PIL image - handle both RGB and grayscale
    if len(original_image.shape) == 3:
        pil_image = Image.fromarray(original_image)
    else:
        pil_image = Image.fromarray(original_image, mode='L')
    
    # Calculate display size while maintaining aspect ratio
    max_width, max_height = 800, 600
    img_width, img_height = pil_image.size
    
    # Calculate scaling factor
    scale = min(max_width / img_width, max_height / img_height, 1.0)
    display_width = int(img_width * scale)
    display_height = int(img_height * scale)
    
    # Canvas for drawing ROIs
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.1)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=pil_image,
        update_streamlit=True,
        width=display_width,
        height=display_height,
        drawing_mode="rect",
        point_display_radius=0,
        key="roi_canvas",
    )
    
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        
        if len(objects) > 0:
            st.write(f"Selected {len(objects)} ROI(s)")
            
            # Calculate scaling factors from display to original image
            scale_x = img_width / display_width
            scale_y = img_height / display_height
            
            roi_patches = []
            for i, obj in objects.iterrows():
                if obj.get('type') == 'rect':
                    x = int(obj['left'] * scale_x)
                    y = int(obj['top'] * scale_y)
                    width = int(obj['width'] * scale_x)
                    height = int(obj['height'] * scale_y)
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, min(x, img_width - 1))
                    y = max(0, min(y, img_height - 1))
                    width = min(width, img_width - x)
                    height = min(height, img_height - y)
                    
                    roi_patches.append({
                        'x': x, 'y': y,
                        'width': width, 'height': height,
                        'roi_id': i + 1,
                        'manual_selection': True
                    })
            
            return roi_patches
    
    return None

def process_manual_roi_nucleus(original_image, roi_patches):
    """
    Process nucleus segmentation on manually selected ROIs
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    roi_temp_dir = tempfile.mkdtemp(prefix=f"roi_{timestamp}_")
    st.session_state.temp_files.append(roi_temp_dir)
    
    download_data = []
    total_lymphocytes = 0
    total_macrophages = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, roi in enumerate(roi_patches):
        progress_bar.progress((i + 1) / len(roi_patches))
        status_text.text(f"Processing ROI {i+1}/{len(roi_patches)}...")
        
        # Extract ROI image
        x, y = roi['x'], roi['y']
        width, height = roi['width'], roi['height']
        roi_img = original_image[y:y+height, x:x+width]
        
        # Save ROI temporarily
        roi_path = os.path.join(roi_temp_dir, f"roi_{i}.png")
        plt.imsave(roi_path, roi_img)
        st.session_state.temp_files.append(roi_path)
        
        try:
            # Process nucleus segmentation
            filtered_preds = process_nucleus_segmentation(roi_path)
            
            lymphocyte_count = sum(1 for inst_info in filtered_preds.values() if inst_info['type'] == 2)
            macrophage_count = sum(1 for inst_info in filtered_preds.values() if inst_info['type'] == 3)
            total_immune_count = lymphocyte_count + macrophage_count
            
            total_lymphocytes += lymphocyte_count
            total_macrophages += macrophage_count
            
            # Create visualization
            color_dict = {
                0: ("Background", (255, 255, 255)),
                2: ("Lymphocyte", (255, 0, 0)),
                3: ("Macrophage", (0, 255, 0)),
            }
            
            overlaid_predictions = overlay_prediction_contours(
                canvas=roi_img,
                inst_dict=filtered_preds,
                draw_dot=False,
                type_colours=color_dict,
                line_thickness=2,
            )
            
            # Display results
            st.write(f"#### ROI {i+1}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(roi_img, caption=f"ROI {i+1} ({width}x{height})")
            with col2:
                st.image(overlaid_predictions, 
                       caption=f"Immune Cells: {total_immune_count} (L:{lymphocyte_count}, M:{macrophage_count})")
            
            # Store for download
            download_data.append({
                'roi_id': i + 1,
                'roi_img': roi_img,
                'nucleus_overlay': overlaid_predictions,
                'lymphocyte_count': lymphocyte_count,
                'macrophage_count': macrophage_count,
                'total_immune_count': total_immune_count,
                'roi_info': roi
            })
            
            st.markdown("---")
            
        except Exception as e:
            st.error(f"Error processing ROI {i+1}: {e}")
            continue
    
    progress_bar.empty()
    status_text.text("✅ ROI analysis complete!")
    
    return download_data, total_lymphocytes, total_macrophages

def main():
    st.title("Histopathology TIL Analysis")
    st.markdown("Upload a histopathology image to analyze tissue types, inflammatory regions, and extract TIL-relevant patches.")
    
    # Sidebar for parameters
    st.sidebar.header("Analysis Parameters")
    inflammatory_threshold = st.sidebar.slider(
        "Inflammatory Threshold", 
        min_value=0.05, 
        max_value=0.5, 
        value=0.1, 
        step=0.05,
        help="Minimum proportion of inflammatory pixels in a patch"
    )
    
    top_k_patches = st.sidebar.slider(
        "Number of Top Patches", 
        min_value=5, 
        max_value=25, 
        value=15,
        help="Number of highest-scoring inflammatory patches to extract"
    )
    
    process_nucleus = st.sidebar.checkbox(
        "Process Nucleus Segmentation", 
        value=True,
        help="Run nucleus segmentation on selected patches (slower but more detailed)"
    )
    
    max_nucleus_patches = st.sidebar.slider(
        "Max Patches for Nucleus Analysis", 
        min_value=1, 
        max_value=15, 
        value=6,
        help="Number of top patches to process with nucleus segmentation"
    )
    
    uploaded_file = st.file_uploader(
        "Choose a histopathology image", 
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
        help="Upload a histopathology image for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
        temp_input_file.write(uploaded_file.getvalue())
        temp_input_file.close()
        st.session_state.temp_files.append(temp_input_file.name)
        
        # Display uploaded image
        st.subheader("Uploaded Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption="Original Histopathology Image", use_container_width=True)
        
        # STEP 1: Tissue Segmentation
        if not st.session_state.tissue_analysis_complete:
            if st.button("Run Tissue Segmentation", type="primary"):
                try:
                    st.subheader("Tissue Segmentation Analysis")
                    
                    with st.status("Processing tissue segmentation...", expanded=True) as status:
                        st.write("Loading segmentation model...")
                        output, temp_output_dir = process_tissue_segmentation(temp_input_file.name)
                        st.write("Extracting inflammatory patches...")
                        patches_info = extract_inflammatory_patches(
                            output, 
                            inflammatory_threshold=inflammatory_threshold,
                            top_k_patches=top_k_patches
                        )
                        st.write("Creating visualizations...")
                        fig1, fig2, fig3 = create_visualizations(patches_info)
                        status.update(label="Tissue analysis complete!", state="complete")
                    
                    # Store in session state
                    st.session_state.patches_info = patches_info
                    st.session_state.visualizations = (fig1, fig2, fig3)
                    st.session_state.tissue_analysis_complete = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    st.info("Please try with a different image or adjust the parameters.")
        
        # STEP 2: Show results and analysis options (only if tissue analysis is complete)
        if st.session_state.tissue_analysis_complete and st.session_state.patches_info:
            patches_info = st.session_state.patches_info
            fig1, fig2, fig3 = st.session_state.visualizations
            
            # Display results in tabs
            st.subheader("Tissue Segmentation Results")
            tab1, tab2, tab3 = st.tabs(["Segmentation Map", "Inflammatory Heatmap", "Selected Patches"])
            
            with tab1:
                st.pyplot(fig1)
                if st.button("💾 Save Segmentation Map", key="save_seg"):
                    img_bytes = save_figure_as_bytes(fig1)
                    st.download_button(
                        label="Download Segmentation Map",
                        data=img_bytes,
                        file_name="tissue_segmentation.png",
                        mime="image/png"
                    )
            
            with tab2:
                st.pyplot(fig2)
                if st.button("💾 Save Heatmap", key="save_heatmap"):
                    img_bytes = save_figure_as_bytes(fig2)
                    st.download_button(
                        label="Download Inflammatory Heatmap",
                        data=img_bytes,
                        file_name="inflammatory_heatmap.png",
                        mime="image/png"
                    )
            
            with tab3:
                st.pyplot(fig3)
                if st.button("💾 Save Patch Overview", key="save_patches"):
                    img_bytes = save_figure_as_bytes(fig3)
                    st.download_button(
                        label="Download Patch Overview",
                        data=img_bytes,
                        file_name="selected_patches.png",
                        mime="image/png"
                    )
            
            # Display patch statistics
            st.subheader("Patch Analysis Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Inflammatory Patches Found", patches_info['total_inflammatory_patches'])
            with col2:
                st.metric("Top Patches Selected", len(patches_info['patches']))
            with col3:
                border_patches = sum(1 for p in patches_info['patches'] if p['border_relevance'])
                st.metric("Border Region Patches", border_patches)
            
            # STEP 3: Analysis Mode Selection
            st.subheader("Choose Analysis Mode")
            analysis_mode = st.radio(
                "Select how to proceed with immune cell analysis:",
                options=["Automated Patch Analysis", "Manual ROI Selection"],
                help="Choose between automated patch extraction or manual region selection",
                key="analysis_mode_radio"
            )
            
            # STEP 4: Process based on selected mode
            if analysis_mode == "Manual ROI Selection":
                st.markdown("---")
                roi_patches = manual_roi_selection(patches_info)
                
                if roi_patches and len(roi_patches) > 0:
                    st.success(f"Selected {len(roi_patches)} ROI(s) for analysis")
                    
                    if st.button("Analyze Selected ROIs", type="primary", key="analyze_roi_btn"):
                        st.subheader("Manual ROI Analysis Results")
                        
                        download_data, total_lymphocytes, total_macrophages = process_manual_roi_nucleus(
                            patches_info['original_image'], roi_patches
                        )
                        
                        # Analysis Summary
                        if download_data:
                            st.subheader("📊 ROI Analysis Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Lymphocytes", total_lymphocytes)
                            with col2:
                                st.metric("Total Macrophages", total_macrophages)
                            with col3:
                                st.metric("Total ROIs Analyzed", len(download_data))
                            
                            # Create summary table
                            summary_df = pd.DataFrame([
                                {
                                    'ROI': d['roi_id'],
                                    'Size (pixels)': f"{d['roi_info']['width']}x{d['roi_info']['height']}",
                                    'Lymphocytes': d['lymphocyte_count'],
                                    'Macrophages': d['macrophage_count'],
                                    'Total': d['total_immune_count']
                                }
                                for d in download_data
                            ])
                            
                            st.write("### Detailed ROI Results")
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Download ROI results
                            if st.button("💾 Download ROI Results"):
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                    # Add summary CSV
                                    csv_buffer = io.StringIO()
                                    summary_df.to_csv(csv_buffer, index=False)
                                    zip_file.writestr("roi_analysis_summary.csv", csv_buffer.getvalue())
                                    
                                    # Add ROI images
                                    for result in download_data:
                                        roi_bytes = io.BytesIO()
                                        plt.imsave(roi_bytes, result['roi_img'], format='png')
                                        roi_bytes.seek(0)
                                        zip_file.writestr(f"roi_{result['roi_id']}_original.png", 
                                                        roi_bytes.getvalue())
                                        
                                        nucleus_bytes = io.BytesIO()
                                        plt.imsave(nucleus_bytes, result['nucleus_overlay'], format='png')
                                        nucleus_bytes.seek(0)
                                        zip_file.writestr(f"roi_{result['roi_id']}_immune_cells.png", 
                                                        nucleus_bytes.getvalue())
                                
                                st.download_button(
                                    label="Download ROI Analysis Results (ZIP)",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"roi_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip"
                                )
            
            elif analysis_mode == "Automated Patch Analysis":
                st.markdown("---")
                if process_nucleus and len(patches_info['patches']) > 0:
                    
                    if st.button("Run Automated Patch Analysis", type="primary", key="analyze_auto_btn"):
                        st.subheader("Nucleus Segmentation Analysis")
                        
                        # Create temporary directory for patch processing
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        patch_temp_dir = tempfile.mkdtemp(prefix=f"patches_{timestamp}_")
                        st.session_state.temp_files.append(patch_temp_dir)
                        
                        # Create containers for real-time display
                        progress_container = st.container()
                        results_container = st.container()
                        
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                        
                        patches_to_process = patches_info['patches'][:max_nucleus_patches]
                        total_patches = len(patches_to_process)
                        
                        # Create download data collector
                        download_data = []
                        total_lymphocytes_all = 0
                        total_macrophages_all = 0
                        
                        with results_container:
                            st.write("### Immune Cell Detection Results")
                            st.write("Results will appear here as each patch is processed...")
                            st.markdown("🟥 **Red**: Lymphocytes | 🟩 **Green**: Macrophages")
                            
                            # Process each patch and display immediately
                            for i, patch_info in enumerate(patches_to_process):
                                # Update progress
                                progress_bar.progress((i + 1) / total_patches)
                                status_text.text(f"Processing patch {i+1}/{total_patches}...")
                                
                                # Extract patch image
                                y, x = patch_info['y'], patch_info['x']
                                patch_img = patches_info['original_image'][y:y+256, x:x+256]
                                
                                # Save patch temporarily
                                patch_path = os.path.join(patch_temp_dir, f"patch_{i}.png")
                                plt.imsave(patch_path, patch_img)
                                st.session_state.temp_files.append(patch_path)
                                
                                try:
                                    # Process nucleus segmentation
                                    filtered_preds = process_nucleus_segmentation(patch_path)
                                    
                                    # Calculate separate counts for lymphocytes and macrophages
                                    lymphocyte_count = sum(1 for inst_info in filtered_preds.values() if inst_info['type'] == 2)
                                    macrophage_count = sum(1 for inst_info in filtered_preds.values() if inst_info['type'] == 3)
                                    total_immune_count = lymphocyte_count + macrophage_count
                                    
                                    # Update running totals
                                    total_lymphocytes_all += lymphocyte_count
                                    total_macrophages_all += macrophage_count
                                    
                                    # Create nucleus visualization with proper colors
                                    color_dict = {
                                        0: ("Background", (255, 255, 255)),
                                        2: ("Lymphocyte", (255, 0, 0)),      
                                        3: ("Macrophage", (0, 255, 0)),      
                                    }
                                    
                                    overlaid_predictions = overlay_prediction_contours(
                                        canvas=patch_img,
                                        inst_dict=filtered_preds,
                                        draw_dot=False,
                                        type_colours=color_dict,
                                        line_thickness=2,
                                    )
                                    
                                    # Display results immediately
                                    st.write(f"#### Patch {i+1}")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.image(patch_img, caption=f"Original Patch {i+1}")
                                        
                                    with col2:
                                        st.image(overlaid_predictions, 
                                               caption=f"Immune Cells: {total_immune_count} (L:{lymphocyte_count}, M:{macrophage_count})")
                                    
                                    # Display detailed metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("TIL Relevance Score", f"{patch_info['til_relevance_score']:.3f}")
                                    with col2:
                                        st.metric("Lymphocytes", lymphocyte_count)
                                    with col3:
                                        st.metric("Macrophages", macrophage_count)
                                    with col4:
                                        st.metric("Border Region", "Yes" if patch_info['border_relevance'] else "No")
                                    
                                    # Store for download
                                    download_data.append({
                                        'patch_id': i + 1,
                                        'patch_img': patch_img,
                                        'nucleus_overlay': overlaid_predictions,
                                        'lymphocyte_count': lymphocyte_count,
                                        'macrophage_count': macrophage_count,
                                        'total_immune_count': total_immune_count,
                                        'patch_info': patch_info
                                    })
                                    
                                    st.markdown("---")  
                                    
                                except Exception as e:
                                    st.error(f"Error processing patch {i+1}: {e}")
                                    continue
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.text("✅ Nucleus analysis complete!")
                        
                        # Analysis Summary
                        if download_data:
                            st.subheader("📊 Analysis Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Lymphocytes", total_lymphocytes_all)
                            with col2:
                                st.metric("Total Macrophages", total_macrophages_all)
                            with col3:
                                st.metric("Total Immune Cells", total_lymphocytes_all + total_macrophages_all)
                            
                            # Create detailed summary table
                            summary_df = pd.DataFrame([
                                {
                                    'Patch': d['patch_id'],
                                    'TIL Score': f"{d['patch_info']['til_relevance_score']:.3f}",
                                    'Lymphocytes': d['lymphocyte_count'],
                                    'Macrophages': d['macrophage_count'],
                                    'Total': d['total_immune_count'],
                                    'Border Region': 'Yes' if d['patch_info']['border_relevance'] else 'No'
                                }
                                for d in download_data
                            ])
                            
                            st.write("### Detailed Results Table")
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Download all results
                            st.subheader("📥 Download Results")
                            if st.button("💾 Download All Nucleus Results"):
                                # Create zip file with all nucleus results
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                    # Add summary CSV
                                    csv_buffer = io.StringIO()
                                    summary_df.to_csv(csv_buffer, index=False)
                                    zip_file.writestr("analysis_summary.csv", csv_buffer.getvalue())
                                    
                                    # Add images
                                    for result in download_data:
                                        # Save original patch
                                        patch_bytes = io.BytesIO()
                                        plt.imsave(patch_bytes, result['patch_img'], format='png')
                                        patch_bytes.seek(0)
                                        zip_file.writestr(f"patch_{result['patch_id']}_original.png", 
                                                        patch_bytes.getvalue())
                                        
                                        # Save nucleus overlay
                                        nucleus_bytes = io.BytesIO()
                                        plt.imsave(nucleus_bytes, result['nucleus_overlay'], format='png')
                                        nucleus_bytes.seek(0)
                                        zip_file.writestr(f"patch_{result['patch_id']}_immune_cells.png", 
                                                        nucleus_bytes.getvalue())
                                
                                st.download_button(
                                    label="Download Complete Analysis Results (ZIP)",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"immune_analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip"
                                )
                else:
                    st.info("Enable nucleus segmentation in the sidebar to run automated patch analysis.")
        
        # Reset button to start over
        if st.session_state.tissue_analysis_complete:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Start Over", help="Reset analysis and start with a new image"):
                    st.session_state.tissue_analysis_complete = False
                    st.session_state.patches_info = None
                    st.session_state.visualizations = None
                    cleanup_temp_files()
                    st.rerun()
        
        # Cleanup button
        if st.button("Clear Results & Cleanup"):
            cleanup_temp_files()
            st.session_state.tissue_analysis_complete = False
            st.session_state.patches_info = None
            st.session_state.visualizations = None
            st.success("Temporary files cleaned up!")
            st.rerun()

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This application performs:
1. **Tissue Segmentation** - Identifies tumor, stroma, inflammatory, necrosis regions
2. **Inflammatory Patch Extraction** - Finds regions with high inflammatory content
3. **Immune Cell Analysis** - Detects lymphocytes and macrophages in selected patches

**Model Used**: HoVerNet-Fast (MoNuSAC dataset)

**Cell Types Detected**: 
- Lymphocytes
- Macrophages 
- Built with <a href="https://github.com/TissueImageAnalytics/tiatoolbox" target="_blank">TIAToolbox</a> and Streamlit.
""",
    unsafe_allow_html=True)

# Footer cleanup
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🧹 Cleanup All Files"):
        cleanup_temp_files()
        st.success("All temporary files cleaned!")

# Auto-cleanup on app restart
if st.session_state.get('app_started', False) == False:
    cleanup_temp_files()
    st.session_state.app_started = True

if __name__ == "__main__":
    main()