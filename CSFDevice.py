import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1032, 615)
        font = QtGui.QFont()
        font.setPointSize(8)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(12, 12, 992, 551))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.startCamera = QtWidgets.QPushButton(self.widget)
        self.startCamera.setObjectName("startCamera")
        self.gridLayout.addWidget(self.startCamera, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(488, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 1, 1, 1)
        self.captureImage = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.captureImage.setFont(font)
        self.captureImage.setObjectName("captureImage")
        self.gridLayout.addWidget(self.captureImage, 0, 2, 2, 4)
        self.runAnalysis = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.runAnalysis.setFont(font)
        self.runAnalysis.setObjectName("runAnalysis")
        self.gridLayout.addWidget(self.runAnalysis, 0, 6, 2, 1)
        self.instructionLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.instructionLabel.setFont(font)
        self.instructionLabel.setObjectName("instructionLabel")
        self.gridLayout.addWidget(self.instructionLabel, 1, 0, 2, 2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 2, 2, 1, 1)
        self.imageLabel = QtWidgets.QLabel(self.widget)
        self.imageLabel.setText("")
        self.imageLabel.setPixmap(QtGui.QPixmap("../../../Downloads/placeholder.jpg"))
        self.imageLabel.setObjectName("imageLabel")
        self.gridLayout.addWidget(self.imageLabel, 3, 0, 10, 2)
        self.resultsLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.resultsLabel.setFont(font)
        self.resultsLabel.setObjectName("resultsLabel")
        self.gridLayout.addWidget(self.resultsLabel, 3, 2, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 4, 2, 1, 1)
        self.WBCcountLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.WBCcountLabel.setFont(font)
        self.WBCcountLabel.setObjectName("WBCcountLabel")
        self.gridLayout.addWidget(self.WBCcountLabel, 5, 2, 1, 3)
        spacerItem3 = QtWidgets.QSpacerItem(168, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 5, 5, 1, 2)
        self.WBCcountResult = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.WBCcountResult.setFont(font)
        self.WBCcountResult.setObjectName("WBCcountResult")
        self.gridLayout.addWidget(self.WBCcountResult, 5, 7, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 48, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem4, 6, 3, 1, 1)
        self.polymorphLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.polymorphLabel.setFont(font)
        self.polymorphLabel.setObjectName("polymorphLabel")
        self.gridLayout.addWidget(self.polymorphLabel, 7, 2, 1, 2)
        spacerItem5 = QtWidgets.QSpacerItem(198, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem5, 7, 4, 1, 3)
        self.polymorphResult = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.polymorphResult.setFont(font)
        self.polymorphResult.setObjectName("polymorphResult")
        self.gridLayout.addWidget(self.polymorphResult, 7, 7, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 48, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem6, 8, 3, 1, 1)
        self.lymphocyteLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lymphocyteLabel.setFont(font)
        self.lymphocyteLabel.setObjectName("lymphocyteLabel")
        self.gridLayout.addWidget(self.lymphocyteLabel, 9, 2, 1, 2)
        spacerItem7 = QtWidgets.QSpacerItem(198, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem7, 9, 4, 1, 3)
        self.lymphocyteResult = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lymphocyteResult.setFont(font)
        self.lymphocyteResult.setObjectName("lymphocyteResult")
        self.gridLayout.addWidget(self.lymphocyteResult, 9, 7, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 78, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem8, 10, 3, 1, 1)
        self.GramStainResultLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.GramStainResultLabel.setFont(font)
        self.GramStainResultLabel.setObjectName("GramStainResultLabel")
        self.gridLayout.addWidget(self.GramStainResultLabel, 11, 2, 1, 3)
        spacerItem9 = QtWidgets.QSpacerItem(178, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem9, 11, 5, 1, 2)
        self.GramStainResultResult = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.GramStainResultResult.setFont(font)
        self.GramStainResultResult.setObjectName("GramStainResultResult")
        self.gridLayout.addWidget(self.GramStainResultResult, 11, 7, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(20, 68, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem10, 12, 3, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1032, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.startCamera.clicked.connect(self.loadImage)
        self.captureImage.clicked.connect(self.savePhoto)
        self.runAnalysis.clicked.connect(self.analyzeImage)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.started = False
        self.runAnalysis.setEnabled(False)
        self.captureImage.setEnabled(False)


    def loadImage(self):
        # Disable start button
        self.startCamera.setEnabled(False)
        self.started = True
        self.captureImage.setEnabled(True)

        # Clear any past results
        self.WBCcountResult.setText('--')
        self.polymorphResult.setText('--')
        self.lymphocyteResult.setText('--')
        self.GramStainResultResult.setText('--')

        # Update instructions
        self.updateInstructions('Focus the image using the knob on the left. Once sharp, press "Capture Image."')

        # Connect to USB camera (CAP_DSHOW opens camera instantly, won't work on Mac)
        camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        # Initialize camera parameters
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

        while(self.started==True):
            # Capture frame
            ret, self.image = camera.read()

            # Update photo
            self.setPhoto()

            key = cv2.waitKey(1) & 0xFF
            if self.captureImage == True:
                # Release the camera
                camera.release()
                break

    
    def setPhoto(self):
        """ This function will take image input and resize it 
            only for display purpose and convert it to QImage
            to set at the label.
        """
        temp = self.image
        temp_mean = temp.astype(np.float32) * 1.0 / temp.mean(axis=(0,1))
        self.wb = (np.clip(temp_mean, 0, 1) * 255).astype(np.uint8)
        wb = self.wb

        wb = cv2.cvtColor(wb, cv2.COLOR_BGR2RGB)

        # Create scale bar (based on calculations)
        wb[50:55, 30:209, :] = 0
        wb[50:62, 30:33, :] = 0
        wb[50:62, 209:212, :] = 0
        wb[50:59, 118:121, :] = 0
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX 
        origin = (40, 45) 
        fontScale = 1.6
        color = (0, 0, 0)
        thickness = 2
        frame = cv2.putText(wb, '20 um', origin, font,  
                        fontScale, color, thickness, cv2.LINE_AA)
        
        # resize for displaying
        frame = cv2.resize(frame, (640, 480))
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimage = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(qimage))


    def savePhoto(self):
        # Set camera started to false
        self.started = False

        # Update instructions
        self.updateInstructions('Press "Run Analysis" to analyze image.')

        # Save original and whitebalanced
        whitebalanced = self.wb
        now = datetime.now()
        current_time = now.strftime("%d_%m_%Y_%H-%M-%S")
        filename = '%s.jpg' % current_time
        if not cv2.imwrite(filename, whitebalanced):
            raise Exception("Could not write image")

        # Enable camera to be started again
        self.startCamera.setEnabled(True)

        # Enable analysis to be run
        self.runAnalysis.setEnabled(True)

    
   def find_regions(self, img):
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

def WBCcount(self, image):
    img = image
    WBC_labels, nuc_labels = self.find_regions(img)

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


def gramStatus(self, image):
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
    
    def analyzeImage(self):
        # Update instructions
        self.updateInstructions('Running analysis, please wait...')

        # Disable capture image button
        self.captureImage.setEnabled(False)
        
        #image = cv2.imread('WBed40x_wbcEDTA_BCpos2_pic5.jpeg')
        image = self.wb
        
        # Perform WBC count/differential
        poly, lymph = self.WBCcount(image)
        total = poly + lymph
        self.WBCcountResult.setText('{} cells'.format(total))

        if total != 0:
            poly_percent = (poly / total) * 100
            lymph_percent = (lymph / total) * 100
        else:
            poly_percent = 0
            lymph_percent = 0
        self.polymorphResult.setText('{} %'.format(poly_percent))
        self.lymphocyteResult.setText('{} %'.format(lymph_percent))
        
        # Perform bacteria search
        bac_status = self.gramStatus(image)
        self.GramStainResultResult.setText('{}'.format(bac_status))

        # Update instructions
        self.updateInstructions('Analysis complete. Press "Start Camera" to begin another sample.')

        # Disable run analysis image button
        self.runAnalysis.setEnabled(False)


    def updateInstructions(self, instruction):
        # Update instruction label
        self.instructionLabel.setText(instruction)
        self.instructionLabel.repaint()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.startCamera.setText(_translate("MainWindow", "Start Camera"))
        self.captureImage.setText(_translate("MainWindow", "Capture Image"))
        self.runAnalysis.setText(_translate("MainWindow", "Run Analysis"))
        self.instructionLabel.setText(_translate("MainWindow", "Press \"Start Camera\" to begin viewing the live microscope image."))
        self.resultsLabel.setText(_translate("MainWindow", "Results"))
        self.WBCcountLabel.setText(_translate("MainWindow", "White Cell Count"))
        self.WBCcountResult.setText(_translate("MainWindow", "--"))
        self.polymorphLabel.setText(_translate("MainWindow", "Polymorphs"))
        self.polymorphResult.setText(_translate("MainWindow", "--"))
        self.lymphocyteLabel.setText(_translate("MainWindow", "Lymphocytes"))
        self.lymphocyteResult.setText(_translate("MainWindow", "--"))
        self.GramStainResultLabel.setText(_translate("MainWindow", "Gram Stain Result"))
        self.GramStainResultResult.setText(_translate("MainWindow", "--"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
