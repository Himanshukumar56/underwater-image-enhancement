from PIL import Image, ImageFilter, ImageOps, ImageStat
import numpy as np
import os
from glob import glob
import sys

# Configuration
RAW_DIR = 'dataset/raw'
REF_DIR = 'dataset/reference'
RESULTS_DIR = 'dataset/results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# Utility: safe image open
def open_image(path):
    try:
        return Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Failed to open {path}: {e}")
        return None

# 1) histogram plotting removed for headless mode
# 2) channel_split - used for debugging, but no plotting in script

def channel_split(image, verbose=False):
    imageR, imageG, imageB = image.split()
    x, y = image.size
    Rchannel = np.zeros((y, x, 3), dtype="uint8")
    Gchannel = np.zeros((y, x, 3), dtype="uint8")
    Bchannel = np.zeros((y, x, 3), dtype="uint8")
    Rchannel[:, :, 0] = np.array(imageR)
    Gchannel[:, :, 1] = np.array(imageG)
    Bchannel[:, :, 2] = np.array(imageB)
    return Image.fromarray(Rchannel), Image.fromarray(Gchannel), Image.fromarray(Bchannel)

# Compensate R and/or B via green channel
def compensate_RB(image, flag):
    imager, imageg, imageb = image.split()
    # extrema
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()

    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    x, y = image.size

    # normalize
    denomR = (maxR - minR) if (maxR - minR) != 0 else 1
    denomG = (maxG - minG) if (maxG - minG) != 0 else 1
    denomB = (maxB - minB) if (maxB - minB) != 0 else 1

    imageR = (imageR - minR) / denomR
    imageG = (imageG - minG) / denomG
    imageB = (imageB - minB) / denomB

    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)

    if flag == 0:
        imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR
        imageB = (imageB + (meanG - meanB) * (1 - imageB) * imageG) * maxB
        imageG = (imageG * maxG)
    else:
        imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR
        imageB = imageB * maxB
        imageG = imageG * maxG

    compensateIm = np.zeros((y, x, 3), dtype="uint8")
    compensateIm[:, :, 0] = np.clip(imageR, 0, 255).astype('uint8')
    compensateIm[:, :, 1] = np.clip(imageG, 0, 255).astype('uint8')
    compensateIm[:, :, 2] = np.clip(imageB, 0, 255).astype('uint8')
    return Image.fromarray(compensateIm)

# Gray-world white balance
def gray_world(image):
    imager, imageg, imageb = image.split()
    imagegray = image.convert('L')

    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    imageGray = np.array(imagegray, np.float64)

    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    meanGray = np.mean(imageGray)

    # avoid division by zero
    meanR = meanR if meanR != 0 else 1
    meanG = meanG if meanG != 0 else 1
    meanB = meanB if meanB != 0 else 1

    imageR = (imageR * meanGray / meanR)
    imageG = (imageG * meanGray / meanG)
    imageB = (imageB * meanGray / meanB)

    x, y = image.size
    whitebalancedIm = np.zeros((y, x, 3), dtype="uint8")
    whitebalancedIm[:, :, 0] = np.clip(imageR, 0, 255).astype('uint8')
    whitebalancedIm[:, :, 1] = np.clip(imageG, 0, 255).astype('uint8')
    whitebalancedIm[:, :, 2] = np.clip(imageB, 0, 255).astype('uint8')
    return Image.fromarray(whitebalancedIm)

# Sharpen using unsharp mask style
def sharpen(wbimage, original):
    smoothed_image = wbimage.filter(ImageFilter.GaussianBlur)
    smoothedr, smoothedg, smoothedb = smoothed_image.split()
    imager, imageg, imageb = wbimage.split()

    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    smoothedR = np.array(smoothedr, np.float64)
    smoothedG = np.array(smoothedg, np.float64)
    smoothedB = np.array(smoothedb, np.float64)

    x, y = wbimage.size
    imageR = 2 * imageR - smoothedR
    imageG = 2 * imageG - smoothedG
    imageB = 2 * imageB - smoothedB

    sharpenIm = np.zeros((y, x, 3), dtype="uint8")
    sharpenIm[:, :, 0] = np.clip(imageR, 0, 255).astype('uint8')
    sharpenIm[:, :, 1] = np.clip(imageG, 0, 255).astype('uint8')
    sharpenIm[:, :, 2] = np.clip(imageB, 0, 255).astype('uint8')
    return Image.fromarray(sharpenIm)

# HSV global equalization (only on V channel)
def hsv_global_equalization(image):
    hsvimage = image.convert('HSV')
    Hue, Saturation, Value = hsvimage.split()
    equalizedValue = ImageOps.equalize(Value, mask=None)
    x, y = image.size
    equalizedIm = np.zeros((y, x, 3), dtype="uint8")
    equalizedIm[:, :, 0] = np.array(Hue)
    equalizedIm[:, :, 1] = np.array(Saturation)
    equalizedIm[:, :, 2] = np.array(equalizedValue)
    hsvimage = Image.fromarray(equalizedIm, 'HSV')
    rgbimage = hsvimage.convert('RGB')
    return rgbimage

# Average fusion
def average_fusion(image1, image2):
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)

    fusedR = ((image1R + image2R) / 2).astype('uint8')
    fusedG = ((image1G + image2G) / 2).astype('uint8')
    fusedB = ((image1B + image2B) / 2).astype('uint8')

    x, y = image1R.shape
    fusedIm = np.zeros((x, y, 3), dtype="uint8")
    fusedIm[:, :, 0] = fusedR
    fusedIm[:, :, 1] = fusedG
    fusedIm[:, :, 2] = fusedB
    return Image.fromarray(fusedIm)

# PCA fusion (kept simple and vectorized)
def pca_fusion(image1, image2):
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()
    def fuse_channel(c1, c2):
        v1 = np.array(c1, np.float64).flatten()
        v2 = np.array(c2, np.float64).flatten()
        M = np.vstack([v1, v2])
        cov = np.cov(M)
        vals, vecs = np.linalg.eig(cov)
        principal = vecs[:, np.argmax(vals)]
        weights = principal / np.sum(principal)
        fused = (weights[0] * np.array(c1, np.float64) + weights[1] * np.array(c2, np.float64))
        return np.clip(fused, 0, 255).astype('uint8')

    fusedR = fuse_channel(image1r, image2r)
    fusedG = fuse_channel(image1g, image2g)
    fusedB = fuse_channel(image1b, image2b)
    x, y = fusedR.shape
    fusedIm = np.zeros((x, y, 3), dtype="uint8")
    fusedIm[:, :, 0] = fusedR
    fusedIm[:, :, 1] = fusedG
    fusedIm[:, :, 2] = fusedB
    return Image.fromarray(fusedIm)

# PSNR helper (expects PIL Images)
def psnr(reference, fused, original):
    ref_arr = np.array(reference, dtype=np.float64)
    fused_arr = np.array(fused, dtype=np.float64)
    orig_arr = np.array(original, dtype=np.float64)
    R2 = np.max(ref_arr) ** 2
    mse_orig = np.mean((ref_arr - orig_arr) ** 2)
    mse_fused = np.mean((ref_arr - fused_arr) ** 2)
    psnr_orig = 10 * np.log10(R2 / mse_orig) if mse_orig != 0 else float('inf')
    psnr_fused = 10 * np.log10(R2 / mse_fused) if mse_fused != 0 else float('inf')
    return psnr_orig, psnr_fused

# Wrapper pipeline
def underwater_image_enhancement(image, refimage, flag, verbose=False):
    compensated = compensate_RB(image, flag)
    whitebalanced = gray_world(compensated)
    contrastenhanced = hsv_global_equalization(whitebalanced)
    sharpened = sharpen(whitebalanced, image)
    averagefused = average_fusion(sharpened, contrastenhanced)
    pcafused = pca_fusion(sharpened, contrastenhanced)
    if verbose:
        p_orig, p_fused = psnr(refimage, pcafused, image)
        print(f"PSNR original: {p_orig:.2f}, PSNR fused: {p_fused:.2f}")
    return pcafused, averagefused

# Main bulk processing
def main():
    raw_paths = sorted(glob(os.path.join(RAW_DIR, '*')))
    ref_paths = sorted(glob(os.path.join(REF_DIR, '*')))
    if not raw_paths:
        print('No raw images found in', RAW_DIR)
        return
    if not ref_paths:
        print('No reference images found in', REF_DIR)
        return
    n = min(len(raw_paths), len(ref_paths))
    print(f'Processing {n} image pairs...')
    for idx in range(n):
        raw_p = raw_paths[idx]
        ref_p = ref_paths[idx]
        im = open_image(raw_p)
        ref = open_image(ref_p)
        if im is None or ref is None:
            print(f'Skipping pair {raw_p}, {ref_p}')
            continue
        try:
            pcaf, avgf = underwater_image_enhancement(im, ref, flag=0, verbose=False)
            pcaf.save(os.path.join(RESULTS_DIR, f'pcafused{idx+1}.png'))
            avgf.save(os.path.join(RESULTS_DIR, f'averagefused{idx+1}.png'))
            if (idx + 1) % 50 == 0:
                print(f'Processed {idx+1}/{n}')
        except Exception as e:
            print(f'Error processing {raw_p} / {ref_p}: {e}')

if __name__ == '__main__':
    main()
