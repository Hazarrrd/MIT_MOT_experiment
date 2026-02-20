import numpy as np
import cv2 as cv
import glob
#ffmpeg -f v4l2 -framerate 5 -video_size 1280x720 -pix_fmt YUYV -i /dev/video0 -vf "fps=1" -q:v 1 /home/janek/psychologia/MIT_MOT_experiment/chessboard_photos/chessboard_%03d.jpg
##ffmpeg -f v4l2 -framerate 10 -video_size 1280x720 -pix_fmt yuv422p -i /dev/video0 -vf "fps=1" -q:v 1 /home/janek/psychologia/MIT_MOT_experiment/chessboard_photos/chessboard_%03d.jpg
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('/home/janek/psychologia/MIT_MOT_experiment/chessboard_photos/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8,6), None)
    #print(corners)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (8,6), corners2, ret)
       # cv.imshow('img', img)
       # cv.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(f"ret: {ret}")
print(f"mtx: {mtx}")
print(f"dist: {dist}")  
print(rvecs)
print(tvecs)

img = cv.imread('/home/janek/psychologia/MIT_MOT_experiment/chessboard_photos/test_026.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print(roi)
print(newcameramtx)

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )

cv.destroyAllWindows()