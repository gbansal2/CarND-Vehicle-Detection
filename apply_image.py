import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
from helper_functions import *
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

class Tracker():
    def __init__(self):
        self.heat = None
        self.framecount = 0

tobj = Tracker()

svc = joblib.load('svc_model.pkl')

with open('params.pkl', 'rb') as f: 
    [orient, pix_per_cell, cell_per_block, spatial_size,
        hist_bins, X_scaler, color_space] = pickle.load(f)


# Define a single function that can extract features using hog sub-sampling and make predictions

ystarts = (350,408,444)
ystops = (484,600,700)
scales = ((1.0,382,510),(1.5,408,600),(2.0,444,700))

image = mpimg.imread('test_images/test1.jpg')
tobj.heat = np.zeros((image.shape[0],image.shape[1],10),dtype=np.float)


def process_image(img):
    [hot_boxes, all_boxes] = find_cars(img, ystarts, ystops, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)

    # Add heat to each box in box list
    # Maintain heat for previous 9 frames and update the current
    for i in range(0,8):
        tobj.heat[:,:,i] = tobj.heat[:,:,i+1]
    #tobj.heat[:,:,0] = tobj.heat[:,:,1]
    #tobj.heat[:,:,1] = tobj.heat[:,:,2]
    #tobj.heat[:,:,2] = np.zeros_like(tobj.heat[:,:,2],dtype=np.float)
    #tobj.heat[:,:,2] = add_heat(tobj.heat[:,:,2],hot_boxes)
    tobj.heat[:,:,9] = np.zeros_like(tobj.heat[:,:,9],dtype=np.float)
    tobj.heat[:,:,9] = add_heat(tobj.heat[:,:,9],hot_boxes)

    # dstack the last two and current heats
    current_heat = np.average(np.dstack((tobj.heat[:,:,0],tobj.heat[:,:,1],
                             tobj.heat[:,:,2],
                             tobj.heat[:,:,3],
                             tobj.heat[:,:,4],
                             tobj.heat[:,:,5],
                             tobj.heat[:,:,6],
                             tobj.heat[:,:,7],
                             tobj.heat[:,:,8],
                             tobj.heat[:,:,9])),axis=2)

    #print(current_heat.shape)

    #thresh = tobj.framecount % 10 + 2
    thresh = 1
        
    # Apply threshold to help remove false positives
    #tobj.heat = apply_threshold(tobj.heat, thresh)
    current_heat = apply_threshold(current_heat, thresh)

    # Visualize the heatmap when displaying    
    #heatmap = np.clip(tobj.heat, 0, 255)
    heatmap = np.clip(current_heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    out_img = draw_labeled_bboxes(np.copy(img), labels)
    #allboxes_img = draw_boxes(img, all_boxes)

    #plt.imshow(out_img)
    #plt.show()
    
    #plt.imshow(allboxes_img)
    #plt.show()

    #plt.imshow(heatmap, cmap='hot')
    #plt.show()

    return out_img

#Apply images
#images = glob.glob('test_images/test*.jpg')
#
#for fname in images:
#    #read each image
#    img = mpimg.imread(fname)
#    out_img = process_image(img)
#    #plt.imshow(out_img)
#    #plt.show()
##print(tobj.heat.shape)


##Apply video
white_output = 'output_videos/project_video_tracking.mp4'
clip1 = VideoFileClip("project_video.mp4").subclip(5,15)
#clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) 
white_clip.write_videofile(white_output, audio=False)
