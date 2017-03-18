import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from lesson_functions import *
from scipy.ndimage.measurements import label
from trackers import VehTracker

# load trained classifier
with open('svc_trained.pkl', 'rb') as fid:
    svc = pickle.load(fid)
with open('feature_scalar.pkl', 'rb') as fid2:
    X_scaler = pickle.load(fid2)

# Extract features from a single image window
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# List of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Show all search windows - for debugging and report
def show_all_windows(img, xstart, ystart, xstop, ystop, scale, pix_per_cell, cell_per_block):
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    window_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            startx = xbox_left + xstart # account for starting x
            starty = ytop_draw+ystart
            endx = xbox_left+win_draw + xstart # account for starting x
            endy = ytop_draw+win_draw+ystart
            #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
            if not window_list:
                first_startx = startx
                first_starty = starty
                first_endx = endx
                first_endy = endy
            cv2.rectangle(draw_img,(startx, starty),(endx, endy),(0,0,255),3)
            window_list.append(((startx, starty), (endx, endy)))

    #draw first window in red
    cv2.rectangle(draw_img,(first_startx, first_starty),(first_endx, first_endy),(255,0,0),6)

    return draw_img, window_list

###############################################################################
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, xstart, ystart, xstop, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    window_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                startx = xbox_left + xstart # account for starting x
                starty = ytop_draw+ystart
                endx = xbox_left+win_draw + xstart # account for starting x
                endy = ytop_draw+win_draw+ystart
                window_list.append(((startx, starty), (endx, endy)))
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                cv2.rectangle(draw_img,(startx, starty),(endx, endy),(0,0,255),6)

    return draw_img, window_list

###############################################################################
## HOG and feature parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 700] # Min and max in y to search in slide_window()

# Define search area for each scale to minimize false positives
# format: winconfig[i] = (xstart, ystart, xstop, ystop, scale)
winconfig = []
winconfig.append((400,400,-1,450,0.5))
winconfig.append((350,400,-1,530,1.0))
winconfig.append((300,400,-1,580,1.5))
winconfig.append((250,400,-1,600,2.0))
winconfig.append((200,400,-1,650,2.5))
winconfig.append((150,400,-1,700,3.0))

nhistory = 7 # number of frames to accumulate hot_windows
vehtracker = VehTracker(nhistory)
def processOneFrame(img):
    frame_box_list = []
    for wincfg in winconfig:
        xstart = wincfg[0]
        ystart = wincfg[1]
        xstop = wincfg[2] if (wincfg[2] != -1) else img.shape[1]
        ystop = wincfg[3]
        scale = wincfg[4]

        # out_img1, window_list1 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        out_img1, window_list1 = find_cars(img, xstart, ystart, xstop, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        frame_box_list = frame_box_list + window_list1

    # update tracker for multi-frame processing
    vehtracker.shift_data()
    vehtracker.enter_new_data(frame_box_list)
    box_list = vehtracker.combine_data()

    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,nhistory) # change threshold with number of frames used
    # heat = apply_threshold(heat,3)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img, labels, box_list, heatmap
    # return draw_img

###############################################################################
# Test on single image
# img = mpimg.imread('./test_images/test1.jpg')

# debug and report
# for wincfg in winconfig:
#     xstart = wincfg[0]
#     ystart = wincfg[1]
#     xstop = wincfg[2] if (wincfg[2] != -1) else img.shape[1]
#     ystop = wincfg[3]
#     scale = wincfg[4]
#
#     check_img, allwinlist = show_all_windows(img, xstart, ystart, xstop, ystop, scale, pix_per_cell, cell_per_block)
#     plt.imshow(check_img)
#     outfname = 'search_windows_scale_' + str(int(10*scale)) + '.jpg'
#     plt.savefig(outfname)
#     plt.close()
#     plt.show()

# combined plot for all search windows
# fig = plt.figure(figsize=(15,10))
# check_img, allwinlist = show_all_windows(img, winconfig[0][0], winconfig[0][1], img.shape[1], winconfig[0][3], winconfig[0][4], pix_per_cell, cell_per_block)
# plt.subplot(321)
# plt.imshow(check_img)
# plt.title('scale = 0.5')
# check_img, allwinlist = show_all_windows(img, winconfig[1][0], winconfig[1][1], img.shape[1], winconfig[1][3], winconfig[1][4], pix_per_cell, cell_per_block)
# plt.subplot(322)
# plt.imshow(check_img)
# plt.title('scale = 1.0')
# check_img, allwinlist = show_all_windows(img, winconfig[2][0], winconfig[2][1], img.shape[1], winconfig[2][3], winconfig[2][4], pix_per_cell, cell_per_block)
# plt.subplot(323)
# plt.imshow(check_img)
# plt.title('scale = 1.5')
# check_img, allwinlist = show_all_windows(img, winconfig[3][0], winconfig[3][1], img.shape[1], winconfig[3][3], winconfig[3][4], pix_per_cell, cell_per_block)
# plt.subplot(324)
# plt.imshow(check_img)
# plt.title('scale = 2.0')
# check_img, allwinlist = show_all_windows(img, winconfig[4][0], winconfig[4][1], img.shape[1], winconfig[4][3], winconfig[4][4], pix_per_cell, cell_per_block)
# plt.subplot(325)
# plt.imshow(check_img)
# plt.title('scale = 2.5')
# check_img, allwinlist = show_all_windows(img, winconfig[5][0], winconfig[5][1], img.shape[1], winconfig[5][3], winconfig[5][4], pix_per_cell, cell_per_block)
# plt.subplot(326)
# plt.imshow(check_img)
# plt.title('scale = 3.0')
# plt.savefig('search_windows_for_all_scales.jpg')
# plt.close()

#### single image plot
# out_img = processOneFrame(img)
# plt.imshow(out_img)
# plt.show()
# plt.imshow(heatmap,cmap='hot')
# plt.show()

#### heatmap plots ####
# draw_img, heatmap = processOneFrame(img)
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(draw_img)
# plt.title('Car Positions')
# plt.subplot(122)
# plt.imshow(heatmap, cmap='hot')
# plt.title('Heat Map')
# fig.tight_layout()
# plt.show()
# outfname = 'carpos_heatmap_' + str(idx) + '.jpg'
# plt.savefig(outfname)
# plt.close()


############################################################

# Part 1 - Test Images
# images = glob.glob('./test_images/*.jpg')
# # images = glob.glob('./testframes/*.jpg')
# for idx, fname in enumerate(images):
#     # load image
#     img = mpimg.imread(fname)

    ##### test single images ######
    # result = processOneFrame(img)
    # plt.figure(figsize=(15,10))
    # plt.imshow(result)
    # outfname = 'result_' + str(idx) + '.jpg'
    # plt.savefig(outfname)
    # plt.close()

    ##### multiple hot windows ####
    # out_img, labels, box_list, heatmap = processOneFrame(img)
    # plt.figure(figsize=(15,10))
    # plt.imshow(draw_boxes(img, box_list))
    # outfname = 'hotwindows_' + str(idx) + '.jpg'
    # plt.savefig(outfname)
    # plt.close()

    ##### heatmaps #####
    # out_img, labels, box_list, heatmap = processOneFrame(img)
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(draw_boxes(img, box_list))
    # plt.title('Car Positions')
    # plt.subplot(122)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # fig.tight_layout()
    # outfname = 'carpos_heatmap_' + str(idx) + '.jpg'
    # plt.savefig(outfname)
    # plt.close()

    #### accumulated heatmap after `nhistory` frames
    # out_img, labels, box_list, heatmap = processOneFrame(img)
    # plt.imshow(heatmap, cmap='gray')
    # outfname = 'final_heatmap_' + str(idx) + '.jpg'
    # plt.savefig(outfname)
    # plt.close()
############################################################

# Part 2 - Video File
from moviepy.editor import VideoFileClip
#
result_output = 'result_test_video_frames' + str(nhistory) + '.mp4'
clip1 = VideoFileClip("test_video.mp4")
# clip1.write_images_sequence("testframe%03d.jpg")
#
# result_output = 'result_project_video_frames' + str(nhistory) + '.mp4'
# clip1 = VideoFileClip("project_video.mp4")
#
white_clip = clip1.fl_image(processOneFrame) # NOTE: this function expects color images!!
white_clip.write_videofile(result_output, audio=False)

###############################################################################
