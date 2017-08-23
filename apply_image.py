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

svc = joblib.load('svc_model.pkl')

with open('params.pkl', 'rb') as f: 
    [orient, pix_per_cell, cell_per_block, spatial_size,
        hist_bins, X_scaler, color_space] = pickle.load(f)


# Define a single function that can extract features using hog sub-sampling and make predictions

ystarts = (350,408,444)
ystops = (484,600,700)
scales = ((1.0,382,510),(1.5,408,600),(2.0,444,700))

def process_image(img):
    [hot_boxes, all_boxes] = find_cars(img, ystarts, ystops, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)

    # Add heat to each box in box list
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat,hot_boxes)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

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

#images = glob.glob('test_images/test*.jpg')
#for fname in images:
#    #read each image
#    img = mpimg.imread(fname)
#    out_img = process_image(img)
#    plt.imshow(out_img)
#    plt.show()

#Apply video
white_output = 'output_videos/project_video_tracking.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) 
white_clip.write_videofile(white_output, audio=False)
