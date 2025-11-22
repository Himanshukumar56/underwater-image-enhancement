import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Underwater Image Enhancement", 
    layout="wide", 
    page_icon=":ocean:"
)

# --- NAVIGATION & MAIN APP STRUCTURE ---
def main():
    st.sidebar.title("Navigation")
    # Keep the original sidebar structure
    selected_box = st.sidebar.selectbox('Choose an option:', ('Underwater Image Enhancer', 'About the App'))
    
    if selected_box == 'About the App':
        about()
    else:
        image_enhancer()

def about():
    st.title("Welcome!")
    st.markdown("### Underwater Image Enhancement Web App")
    
    st.info("This project implements a fusion-based restoration algorithm to correct color cast and improve visibility in underwater imagery.")
    
    with st.expander("📄 Abstract", expanded=True):
        st.write("""
        Underwater images find application in various fields, like marine research, inspection of
        aquatic habitat, underwater surveillance, identification of minerals, and more. However,
        underwater shots are affected a lot during the acquisition process due to the absorption
        and scattering of light. 
        
        As depth increases, longer wavelengths get absorbed by water; therefore, the images appear 
        predominantly bluish-green. This results in low contrast and color distortion. Hence, 
        underwater images need enhancement to improve quality for various applications.
        """)
    
    with st.expander("👥 Team Members"):
        st.write("""
        **Pranjali Bajpai** - 2018EEB1243  
        **Yogesh Vaidhya** - 2018EEB1277
        """)

# --- CORE IMAGE PROCESSING FUNCTIONS ---

def compensate_RB(image, flag):
    """Step 1: Red/Blue Channel Compensation"""
    imager, imageg, imageb = image.split()
    
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()
    
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    
    x, y = image.size
    
    # Normalize to 0-1
    imageR = (imageR - minR) / (maxR - minR)
    imageG = (imageG - minG) / (maxG - minG)
    imageB = (imageB - minB) / (maxB - minB)
    
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    
    # Compensation Logic
    if flag == 0:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int((imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)
                imageB[i][j] = int((imageB[i][j] + (meanG - meanB) * (1 - imageB[i][j]) * imageG[i][j]) * maxB)
        for i in range(y):
            for j in range(x):
                imageG[i][j] = int(imageG[i][j] * maxG)

    if flag == 1:
        for i in range(y):
            for j in range(x):
                imageR[i][j] = int((imageR[i][j] + (meanG - meanR) * (1 - imageR[i][j]) * imageG[i][j]) * maxR)
        for i in range(y):
            for j in range(x):
                imageB[i][j] = int(imageB[i][j] * maxB)
                imageG[i][j] = int(imageG[i][j] * maxG)
            
    compensateIm = np.zeros((y, x, 3), dtype="uint8")
    compensateIm[:, :, 0] = imageR
    compensateIm[:, :, 1] = imageG
    compensateIm[:, :, 2] = imageB
    
    return Image.fromarray(compensateIm)

def gray_world(image):
    """Step 2: Gray World White Balancing"""
    imager, imageg, imageb = image.split()
    imagegray = image.convert('L')
    
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    imageGray = np.array(imagegray, np.float64)
    
    x, y = image.size
    meanR, meanG, meanB = np.mean(imageR), np.mean(imageG), np.mean(imageB)
    meanGray = np.mean(imageGray)
    
    # Apply Gray World
    imageR = (imageR * (meanGray / meanR)).astype(np.uint8)
    imageG = (imageG * (meanGray / meanG)).astype(np.uint8)
    imageB = (imageB * (meanGray / meanB)).astype(np.uint8)
    
    whitebalancedIm = np.zeros((y, x, 3), dtype="uint8")
    whitebalancedIm[:, :, 0] = imageR
    whitebalancedIm[:, :, 1] = imageG
    whitebalancedIm[:, :, 2] = imageB
    
    return Image.fromarray(whitebalancedIm)

def sharpen(wbimage):
    """Step 3a: Unsharp Masking"""
    smoothed_image = wbimage.filter(ImageFilter.GaussianBlur)
    imager, imageg, imageb = wbimage.split()
    smoothedr, smoothedg, smoothedb = smoothed_image.split()
    
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    smoothedR = np.array(smoothedr, np.float64)
    smoothedG = np.array(smoothedg, np.float64)
    smoothedB = np.array(smoothedb, np.float64)
    
    x, y = wbimage.size
    
    # Formula: 2*Original - Gaussian
    imageR = np.clip(2 * imageR - smoothedR, 0, 255)
    imageG = np.clip(2 * imageG - smoothedG, 0, 255)
    imageB = np.clip(2 * imageB - smoothedB, 0, 255)
    
    sharpenIm = np.zeros((y, x, 3), dtype="uint8")        
    sharpenIm[:, :, 0] = imageR
    sharpenIm[:, :, 1] = imageG
    sharpenIm[:, :, 2] = imageB
    
    return Image.fromarray(sharpenIm)

def hsv_global_equalization(image):
    """Step 3b: Histogram Equalization"""
    hsvimage = image.convert('HSV')
    Hue, Saturation, Value = hsvimage.split()
    equalizedValue = ImageOps.equalize(Value, mask=None)
    
    x, y = image.size
    equalizedIm = np.zeros((y, x, 3), dtype="uint8")
    equalizedIm[:, :, 0] = np.array(Hue)
    equalizedIm[:, :, 1] = np.array(Saturation)
    equalizedIm[:, :, 2] = np.array(equalizedValue)
    
    hsvimage = Image.fromarray(equalizedIm, 'HSV')
    return hsvimage.convert('RGB')

def average_fusion(image1, image2):
    """Step 4a: Average Fusion"""
    image1_arr = np.array(image1, np.float64)
    image2_arr = np.array(image2, np.float64)
    fused_arr = np.clip((image1_arr + image2_arr) / 2, 0, 255).astype("uint8")
    return Image.fromarray(fused_arr)

def pca_fusion(image1, image2):
    """Step 4b: PCA Fusion"""
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()
    
    image1R = np.array(image1r, np.float64).flatten()
    image1G = np.array(image1g, np.float64).flatten()
    image1B = np.array(image1b, np.float64).flatten()
    image2R = np.array(image2r, np.float64).flatten()
    image2G = np.array(image2g, np.float64).flatten()
    image2B = np.array(image2b, np.float64).flatten()
    
    mean1R, mean1G, mean1B = np.mean(image1R), np.mean(image1G), np.mean(image1B)
    mean2R, mean2G, mean2B = np.mean(image2R), np.mean(image2G), np.mean(image2B)
    
    imageR = np.array((image1R - mean1R, image2R - mean2R))
    imageG = np.array((image1G - mean1G, image2G - mean2G))
    imageB = np.array((image1B - mean1B, image2B - mean2B))
    
    def get_weights(cov_matrix):
        val, vec = np.linalg.eig(cov_matrix)
        if val[0] >= val[1]:
            return vec[:, 0] / sum(vec[:, 0])
        else:
            return vec[:, 1] / sum(vec[:, 1])

    coefR = get_weights(np.cov(imageR))
    coefG = get_weights(np.cov(imageG))
    coefB = get_weights(np.cov(imageB))
    
    img1_arr = np.array(image1, np.float64)
    img2_arr = np.array(image2, np.float64)
    fusedIm = np.zeros(img1_arr.shape, dtype="uint8")
    
    fusedIm[:,:,0] = coefR[0]*img1_arr[:,:,0] + coefR[1]*img2_arr[:,:,0]
    fusedIm[:,:,1] = coefG[0]*img1_arr[:,:,1] + coefG[1]*img2_arr[:,:,1]
    fusedIm[:,:,2] = coefB[0]*img1_arr[:,:,2] + coefB[1]*img2_arr[:,:,2]
    
    return Image.fromarray(fusedIm)

def underwater_image_enhancement(image, flag):
    """
    Orchestrator function that runs the pipeline and returns all intermediate images.
    """
    # 1. Compensation
    comp_img = compensate_RB(image, flag)
    
    # 2. White Balance
    wb_img = gray_world(comp_img)
    
    # 3. Contrast & Sharpening
    contrast_img = hsv_global_equalization(wb_img)
    sharpen_img = sharpen(wb_img)
    
    # 4. Fusion
    avg_fused = average_fusion(sharpen_img, contrast_img)
    pca_fused = pca_fusion(sharpen_img, contrast_img)
    
    # Return Dictionary of all steps
    return {
        "compensated": comp_img,
        "white_balanced": wb_img,
        "contrast": contrast_img,
        "sharpened": sharpen_img,
        "average": avg_fused,
        "pca": pca_fused
    }

# --- MAIN ENHANCEMENT UI ---

def image_enhancer():
    st.title("Underwater Image Enhancement")
    st.markdown("#### Improve visibility and color using a multi-stage fusion pipeline.")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        file = st.file_uploader("Upload Image (RGB)", type=["jpg", "jpeg", "png"])

    if file is None:
        st.info("Please upload an underwater image to get started.")
        return

    image = Image.open(file)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.write("### Processing Options")
        st.caption("Click the button below to generate the enhanced image and view the internal processing steps.")
        if st.button("Enhance Image", type="primary", use_container_width=True):
            with st.spinner('Processing image pipeline...'):
                
                # Run the algorithm (Flag 1 is usually better for general underwater images)
                results = underwater_image_enhancement(image, 1)
                
                # --- SHOWING STEPS PROFESSIONALLY ---
                
                st.success("Processing Complete!")
                
                st.markdown("### Step-by-Step Transformation")
                
                # Row 1: Compensation & White Balance
                step_c1, step_c2 = st.columns(2)
                with step_c1:
                    st.markdown("**Step 1: RB Compensation**")
                    st.image(results['compensated'], caption="Corrected Color Cast", use_container_width=True)
                with step_c2:
                    st.markdown("**Step 2: Gray World WB**")
                    st.image(results['white_balanced'], caption="Restored Temperature", use_container_width=True)
                
                st.markdown("---")
                
                # Row 2: Feature Extraction
                st.markdown("**Step 3: Feature Extraction (Parallel)**")
                step_c3, step_c4 = st.columns(2)
                with step_c3:
                    st.image(results['sharpened'], caption="Input A: Sharpened Details", use_container_width=True)
                with step_c4:
                    st.image(results['contrast'], caption="Input B: Enhanced Contrast", use_container_width=True)
                
                st.markdown("---")
                
                # Row 3: Final Results
                st.markdown("### Final Output (Fusion)")
                
                final_c1, final_c2 = st.columns(2)
                with final_c1:
                    st.image(results['average'], caption="Method 1: Average Fusion", use_container_width=True)
                with final_c2:
                    st.image(results['pca'], caption="Method 2: PCA Fusion (Recommended)", use_container_width=True)

                # Download Button for PCA Result
                buffered = io.BytesIO()
                results['pca'].save(buffered, format="PNG")
                st.download_button(
                    label="Download PCA Enhanced Image",
                    data=buffered.getvalue(),
                    file_name="enhanced_output.png",
                    mime="image/png",
                    type="primary"
                )

if __name__ == "__main__":
    main()