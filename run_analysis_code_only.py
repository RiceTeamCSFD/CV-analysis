import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import os
import matplotlib.pyplot as plt

def find_regions(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to smooth edges
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply binary thresholding for WBCs
    (T, WBCthresh) = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY_INV)

    # Morphological processing for WBCs
    WBCclosed1 = morphology.closing(WBCthresh, morphology.disk(15))
    WBCdilated1 = morphology.dilation(WBCclosed1, morphology.disk(5))
    WBCclosed2 = morphology.closing(WBCdilated1, morphology.disk(15))
    WBCdilated2 = morphology.dilation(WBCclosed2, morphology.disk(5))

    # Generate the markers as local maxima of the distance to the background for WBCs
    WBCdistance = ndi.distance_transform_edt(WBCdilated2)
    WBCcoords = peak_local_max(WBCdistance, footprint=np.ones((3, 3)), labels=WBCdilated2)
    WBCmask = np.zeros(WBCdistance.shape, dtype=bool)
    WBCmask[tuple(WBCcoords.T)] = True
    WBCmarkers, _ = ndi.label(WBCmask)

    WBClabels = watershed(-WBCdistance, WBCmarkers, mask=WBCdilated2)

    # Calculate region properties for WBCs
    possible_WBCs = measure.regionprops(WBClabels)

    # Define minimum and maximum size for filtering for WBCs
    WBC_min = 7500  # Minimum expected WBC size
    WBC_max = 50000  # Maximum expected WBC size
    WBC_min_circularity = 0.6

    # Filter regions by size for WBCs
    filtered_WBCs = np.zeros(WBClabels.shape, dtype=np.uint8)
    total_WBC = 0
    for region in possible_WBCs:
        WBC_area = region.area
        WBC_perimeter = region.perimeter
        if WBC_perimeter != 0:  # Check if WBC_perimeter is not zero
            WBC_circularity = (4 * np.pi * WBC_area) / (WBC_perimeter ** 2)
        if WBC_min < WBC_area < WBC_max and WBC_circularity > WBC_min_circularity:
            filtered_WBCs[WBClabels == region.label] = region.label

    # Apply binary thresholding for nuclei
    (T, nucThresh) = cv2.threshold(blurred, 150, 235, cv2.THRESH_BINARY_INV) 

    # Morphological processing for nuclei
    nucclosed1 = morphology.closing(nucThresh, morphology.disk(3))
    nucfilled = morphology.dilation(nucclosed1, morphology.disk(10))

    # Generate the markers as local maxima of the distance to the background for nuclei
    nucdistance = ndi.distance_transform_edt(nucfilled)
    nuccoords = peak_local_max(nucdistance, footprint=np.ones((3, 3)), labels=nucfilled)
    nucmask = np.zeros(nucdistance.shape, dtype=bool)
    nucmask[tuple(nuccoords.T)] = True
    nucmarkers, _ = ndi.label(nucmask)
    nuclabels = watershed(-nucdistance, nucmarkers, mask=nucfilled)

    # Calculate region properties for nuclei
    possible_nucs = measure.regionprops(nuclabels)

    # Define minimum and maximum size for filtering for nuclei
    nuc_min = 1000  # Minimum expected nucleus size
    nuc_max = 15000  # Maximum expected nucleus size
    nuc_min_circularity = 0.60

    # Filter regions by size for nuclei
    filtered_nucs = np.zeros(nuclabels.shape, dtype=np.uint8)
    for region in possible_nucs:
        nuc_area = region.area
        nuc_perimeter = region.perimeter
        if nuc_perimeter != 0:  # Check if WBC_perimeter is not zero
            nuc_circularity = (4 * np.pi * nuc_area) / (nuc_perimeter ** 2)
        if nuc_min < nuc_area < nuc_max and nuc_circularity > nuc_min_circularity:
            filtered_nucs[nuclabels == region.label] = region.label
    
    # Plot WBC and nucleus thresholds and watershed regions
    fig, ax = plt.subplots(2, 2, figsize=(15, 5), sharex=True, sharey=True)
    ax[0, 0].imshow(WBCdilated2, cmap=plt.cm.gray)
    ax[0, 0].axis(False)
    ax[0, 0].set_title('WBC Threshold') #Remove
    ax[0, 1].imshow(filtered_WBCs, cmap=plt.cm.nipy_spectral)
    ax[0, 1].axis(False)
    ax[0, 1].set_title('Detected WBCs')
    ax[1, 0].imshow(nucfilled, cmap=plt.cm.gray)
    ax[1, 0].axis(False)
    ax[1, 0].set_title('Nucleus Threshold')
    ax[1, 1].imshow(nuclabels, cmap=plt.cm.nipy_spectral)
    ax[0, 0].axis(False)
    ax[1, 1].set_title('Detected Nuclei')
    plt.show()

    return filtered_WBCs, filtered_nucs

def WBCcount(image):
    img = image
    WBC_labels, nuc_labels = find_regions(img)

    WBC_regions = measure.regionprops(WBC_labels)
    nuc_regions = measure.regionprops(nuc_labels)

    poly = 0
    lymph = 0

    # Iterate through each detected WBC region
    for WBC_region in WBC_regions:
        nuclei_within_WBC = 0  # Initialize count for each WBC region

        # Get bounding box of current WBC region
        minr, minc, maxr, maxc = WBC_region.bbox
        for nuc_region in nuc_regions:
            nuc_minr, nuc_minc, nuc_maxr, nuc_maxc = nuc_region.bbox
            # Check if the centroid of the nucleus is within the bounding box of the WBC region
            nuc_center = np.array([nuc_region.centroid[0], nuc_region.centroid[1]])
            if minr <= nuc_center[0] <= maxr and minc <= nuc_center[1] <= maxc:
                nuclei_within_WBC += 1

        # Classify based on nuclei count (assuming >1 nuclei = poly, 1 nucleus = lymph)
        if nuclei_within_WBC > 1:
            poly += 1
        elif nuclei_within_WBC == 1:
            lymph += 1
        # If no nuclei are found within a WBC, it may indicate a problem with segmentation or counting
        else:
            print(f"No nuclei found for WBC with label {WBC_region.label}")

    total_WBC = poly + lymph
    #print(f"Total WBC count: {total_WBC}")
    if total_WBC != 0:
        poly_percent = (poly / total_WBC) * 100
        lymph_percent = (lymph / total_WBC) * 100
    else:
        poly_percent = 0
        lymph_percent = 0
    print('Polymorph %: {}'.format(poly_percent))
    print('Lymphocyte %: {}'.format(lymph_percent))

    return poly, lymph


def gramStatus(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply binary thresholding
    _, threshInv = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY_INV)
    
    # Create a mask from the binary thresholded image
    cell_mask = np.zeros_like(threshInv)
    cell_mask[threshInv > 0] = 255

    # Convert original image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mask the HSV image using the cell mask
    masked_hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=cell_mask)

    # Flatten the masked HSV image and the mask to 1D arrays
    masked_hsv_flat = masked_hsv_image.reshape((-1, 3))
    cell_mask_flat = cell_mask.flatten()

    # Sample only the pixels within the mask
    sampled_hsv = masked_hsv_flat[cell_mask_flat > 0]

    hist = cv2.calcHist([sampled_hsv], [0], None, [90], [90, 180])

    #plt.plot(hist)
    #plt.xlabel('Pixel Hue Value')
    #plt.ylabel('Frequency')
    #plt.show()

    purple_upper = 45
    purple_lower = 5
    pink_upper = 60
    pink_lower = 45

    purple_char = np.sum(hist[purple_lower-90:purple_upper-90])
    pink_char = np.sum(hist[pink_lower-90:pink_upper-90])

    if purple_char + pink_char < 10000:
        bac_status = 'No organisms seen'
    elif purple_char > pink_char:
        bac_status = 'Gram (+)'
    elif purple_char < pink_char:
        bac_status = 'Gram (-)'

    return bac_status

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return None  # Return None if the image cannot be loaded
    
    poly, lymph = WBCcount(image)
    total = poly + lymph

    if total != 0:
        poly_percent = (poly / total) * 100
        lymph_percent = (lymph / total) * 100
    else:
        poly_percent = 0
        lymph_percent = 0

    bac_status = gramStatus(image)
    return {
        'filename': os.path.basename(image_path),
        'total_WBC': total,
        'poly_percent': poly_percent,
        'lymph_percent': lymph_percent,
        'gram_status': bac_status
    }

# Directory containing images
folder_path = "C:\\Users\\melen\\OneDrive\\Desktop\\Senior_Design\\4_26_RRneg"
image_extensions = ['.jpeg', '.jpg', '.png']

results = []  # List to hold all results

# Iterate over all files in the directory
for filename in os.listdir(folder_path):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(folder_path, filename)
        print(f'Processing {filename}')
        result = process_image(image_path)
        if result:
            results.append(result)

# Print out the total results or save them to a file
for result in results:
    print(result)

# Optional: Summarize total counts or averages
total_WBCs = sum(r['total_WBC'] for r in results)
print(f'Total WBC Count across all images: {total_WBCs}')
concentration = (total_WBCs / 0.69609575) * 39.07905965
print(f'Total WBC concentration: {concentration} cells/uL')
