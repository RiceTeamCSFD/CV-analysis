import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def find_regions(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove the scale bar for now (delete this line later)
    #gray = gray[65:, :]

    # Blur to smooth edges
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply binary thresholding fro WBCs
    (T, WBCthresh) = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)

    # Morphological processing
    WBCclosed1 = morphology.closing(WBCthresh, morphology.disk(15))
    WBCdilated1 = morphology.dilation(WBCclosed1, morphology.disk(5))
    WBCclosed2 = morphology.closing(WBCdilated1, morphology.disk(15))
    WBCdilated2 = morphology.dilation(WBCclosed2, morphology.disk(5))

    # Generate the markers as local maxima of the distance to the background
    WBCdistance = ndi.distance_transform_edt(WBCdilated2)
    WBCcoords = peak_local_max(WBCdistance, footprint=np.ones((3, 3)), labels=WBCdilated2)
    WBCmask = np.zeros(WBCdistance.shape, dtype=bool)
    WBCmask[tuple(WBCcoords.T)] = True
    WBCmarkers, _ = ndi.label(WBCmask)
    WBClabels = watershed(-WBCdistance, WBCmarkers, mask=WBCdilated2)

    # Calculate region properties
    possible_WBCs = measure.regionprops(WBClabels)

    # Define minimum and maximum size for filtering
    WBC_min = 10000  # Minimum expected WBC size
    WBC_max = 60000 # Maximum expected WBC size
    WBC_min_circularity = 0.7

    # Filter regions by size
    filtered_WBCs = np.zeros(WBClabels.shape, dtype=np.uint8)
    total_WBC = 0
    for region in possible_WBCs:
        WBC_area = region.area
        WBC_perimeter = region.perimeter
        WBC_circularity = (4 * np.pi * WBC_area) / (WBC_perimeter ** 2)

        if WBC_min < WBC_area < WBC_max and WBC_circularity > WBC_min_circularity:
            filtered_WBCs[WBClabels == region.label] = region.label

    # Apply binary thresholding for nuclei
    (T, nucThresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('nucThresh', nucThresh)

    # Morphological processing
    nucclosed1 = morphology.closing(nucThresh, morphology.disk(3))
    nucfilled = morphology.dilation(nucclosed1, morphology.disk(10))
    #nucfilled = morphology.remove_small_holes(nucdilated1, area_threshold=100)

    # Generate the markers as local maxima of the distance to the background
    nucdistance = ndi.distance_transform_edt(nucfilled)
    nuccoords = peak_local_max(nucdistance, footprint=np.ones((3, 3)), labels=nucfilled)
    nucmask = np.zeros(nucdistance.shape, dtype=bool)
    nucmask[tuple(nuccoords.T)] = True
    nucmarkers, _ = ndi.label(nucmask)
    nuclabels = watershed(-nucdistance, nucmarkers, mask=nucfilled)

    # Calculate region properties
    possible_nucs = measure.regionprops(nuclabels)

    # Define minimum and maximum size for filtering
    nuc_min = 2000  # Minimum expected nucleus size
    nuc_max = 50000 # Maximum expected nucleus size

    # Filter regions by size
    filtered_nucs = np.zeros(nuclabels.shape, dtype=np.uint8)
    for region in possible_nucs:
        nuc_area = region.area

        if nuc_min < nuc_area < nuc_max:
            filtered_nucs[nuclabels == region.label] = region.label
    
    # Print WBC and Nucleus thresholds and watershed regions
    fig, ax = plt.subplots(2, 2, figsize=(15, 5), sharex=True, sharey=True)

    ax[0, 0].imshow(WBCdilated2, cmap=plt.cm.gray)
    ax[0, 0].set_title('WBC Threshold')
    ax[0, 1].imshow(filtered_WBCs, cmap=plt.cm.nipy_spectral)
    ax[0, 1].set_title('Detected WBCs')
    ax[1, 0].imshow(nucfilled, cmap=plt.cm.gray)
    ax[1, 0].set_title('Nucleus Threshold')
    ax[1, 1].imshow(nuclabels, cmap=plt.cm.nipy_spectral)
    ax[1, 1].set_title('Detected Nuclei')
    plt.show()

    return filtered_WBCs, filtered_nucs


def WBCcount(image):
    # Create new variable to modify
    img = image
    
    WBC_labels, nuc_labels = find_regions(img)
    
    WBC_regions = measure.regionprops(WBC_labels)
    nuc_regions = measure.regionprops(nuc_labels)

    poly = 0
    lymph = 0

    for WBC_region in WBC_regions:
        # Initialize counter for nuclei within current WBC region
        nuclei_within_WBC = 0
        
        # Get bounding box of current WBC region
        minr, minc, maxr, maxc = WBC_region.bbox
        
        # Iterate over nuclei regions to check if they are within the current WBC region
        for nuc_region in nuc_regions:
            nuc_minr, nuc_minc, nuc_maxr, nuc_maxc = nuc_region.bbox
            # Check if the center of the nucleus is within the bounding box of the WBC region
            nuc_center = np.array([nuc_region.centroid[0], nuc_region.centroid[1]])
            if minr <= nuc_center[0] <= maxr and minc <= nuc_center[1] <= maxc:
                nuclei_within_WBC += 1
        
        # Increment the appropriate counter based on the type of cell (poly vs. lymph)
        if nuclei_within_WBC > 1:
            poly += 1
        elif nuclei_within_WBC == 1:
            lymph += 1
        else:
            print('Error, no nucleus detected')

    return poly, lymph


def gramStatus(image):
        
        #fileName = 'WB40x_BCpos_pic7.jpg'
        #file_image = cv2.imread(fileName)
        

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply binary thresholding
        (T, threshInv) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        
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
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        '''
        # Plot image and histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot image
        ax1.imshow(rgb)
        ax1.set_title('Histogram for ')# + fileName)
        ax1.axis('off')
        
        # Plot histogram
        hues = range(90, 180)
        ax2.plot(hues, hist)
        ax2.set_title('Histogram')
        ax2.set_xlabel('Pixel Hue Value', fontsize=16)
        ax2.set_ylabel('Frequency', fontsize=16)
        ax2.tick_params(labelsize=14)
        plt.show()
        '''
        purple_upper = 125
        purple_lower = 100
        pink_upper = 150
        pink_lower = 125

        purple_char = np.sum(hist[purple_lower-90:purple_upper-90])
        pink_char = np.sum(hist[pink_lower-90:pink_upper-90])

        if purple_char + pink_char < 10000:
            bac_status = 'No organisms seen'

        elif purple_char > pink_char:
            bac_status = 'Gram (+)'

        elif purple_char < pink_char:
            bac_status = 'Gram (-)'

        return bac_status


'''
Main code starts here
'''

# Open image
filename = '40x_gs_cenblood_pic2_wb.jpg'
image = cv2.imread(filename)

# Perform WBC count/differential
poly, lymph = WBCcount(image)
total = poly + lymph
print('WBC Count: {} cells'.format(total))

if total != 0:
    poly_percent = (poly / total) * 100
    lymph_percent = (lymph / total) * 100
else:
    poly_percent = 0
    lymph_percent = 0
print('Polymorph %: {} %'.format(poly_percent))
print('Lymphocyte %: {} %'.format(lymph_percent))

# Perform bacteria search
bac_status = gramStatus(image)
print('Gram Stain Results: {}'.format(bac_status))