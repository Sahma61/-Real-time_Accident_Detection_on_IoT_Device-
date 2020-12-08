import cv2
import os
def getFrame(sec, path, out_path):
    vidcap = cv2.VideoCapture(path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        st_path = os.path.join(out_path, "image"+str(count)+".jpg")
        cv2.imwrite(st_path, image)     # save frame as JPG file
    return hasFrames

temp_dir = '/home/sahma61/Downloads/Proj_final/test_input'
output_dir = '/home/sahma61/Downloads/Proj_final/output_frames'

for video_file in os.listdir(temp_dir):
    path = os.path.join(temp_dir, video_file)
    print(path)
    output_dir_folder = os.path.join(output_dir, video_file)
    os.mkdir(output_dir_folder)
    sec = 0
    frameRate = 1.0 #//it will capture image in each 0.5 second
    count=1
    success = getFrame(sec, path, output_dir_folder)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec, path, output_dir_folder)
