#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:49:30 2021

Object Tracker

Calculates velocity and acceleration of athlete running 40 Yard Dash


@author: Dr Connor Jones
         con1z@hotmail.com
         Available for Freelance Work
"""


##### BG removal and tracking combined #####
import numpy as np
import cv2
import matplotlib.pyplot as plt
from averageFrame import averageFrame
from ManualDistanceCalibration import WeightedCurve







##### User Inputs #####
file_path = '40Yard_Trimmed2.mp4'
#Measurement between yard lines (input any number of measurements)
# measurements = [143,141,139,150,148,150,141,141,139,135]
Units = 1           #For metres set to 1, for yards set to 2
#####






# startFrame = 0     #Frame to start object detection

cap = cv2.VideoCapture(file_path)
result = averageFrame(cap)
cap = cv2.VideoCapture(file_path)
first_iter = True
first_iter2 = True
# result = cv2.imread('averaged_frame.jpg')
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

FPS = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(3))  # float `width`
height = int(cap.get(4))  # float `height`


frameTotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
centroid = np.zeros(frameTotal)
TOP = np.zeros(frameTotal)
BOTTOM = np.zeros(frameTotal)

LEFTcoords = [0] * frameTotal
RIGHTcoords = [0] * frameTotal
TOPcoords = [0] * frameTotal
BOTTOMcoords = [0] * frameTotal

# LEFTcoords = np.zeros(frameTotal)
# RIGHTcoords = np.zeros(frameTotal)
# TOPcoords = np.zeros(frameTotal)
# BOTTOMcoords = np.zeros(frameTotal)

bboxRec = [0] * frameTotal      #Initiate list

while True:
    ret, frameCOL = cap.read()
    frameno = cap.get(cv2.CAP_PROP_POS_FRAMES)
    # if frameno < startFrame:
    #     continue
    if frameCOL is None:
        break
    
    frame = cv2.cvtColor(frameCOL, cv2.COLOR_BGR2GRAY)


    if first_iter:
        avg = np.float32(frame)
        first_iter = False


    bg_rem = frame - result
    
    # bg_rem = cv2.convertScaleAbs(frame - result)
    clean = bg_rem < 200
    # clean = bg_rem
    bg_rem_clean = clean * bg_rem
    # blur = cv2.GaussianBlur(bg_rem_clean,(5,5),0)
    blur = bg_rem_clean
    ret2,bg_rem_clean = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret2,bg_rem_clean = cv2.threshold(blur,10,255,cv2.THRESH_BINARY)
    
    #####Erodes and Dilates the image
    # bg_rem_clean = cv2.erode(bg_rem_clean, None, iterations=1)
    # bg_rem_clean = cv2.dilate(bg_rem_clean, None, iterations=1)
    
    #####Select ROI
    x1 = 300
    x2 = 1000
    y1 = 0
    y2 = 1080
    ROI = bg_rem_clean[y1:y2, x1:x2]
    cv2.imshow("ROI", ROI)
    
    contours, hierarchy = cv2.findContours(ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    

    extLeft = [0] * len(contours)
    extRight = [0] * len(contours)
    extTop = [0] * len(contours)
    extBot = [0] * len(contours)

    i = 0
    
    popup = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1:
            # popup.append(i)       #Used to remove contours
            pass
        extLeft[i] = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight[i] = tuple(cnt[cnt[:, :, 0].argmax()][0])
        extTop[i] = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extBot[i] = tuple(cnt[cnt[:, :, 1].argmax()][0])
        i = i + 1
        
    for i in popup[::-1]:
        contours.pop(i)
        
    extLeftX = np.zeros(len(contours))
    extLeftY = np.zeros(len(contours))
    
    extRightX = np.zeros(len(contours))
    extRightY = np.zeros(len(contours))
    
    extTopX = np.zeros(len(contours))
    extTopY = np.zeros(len(contours))
    
    extBotX = np.zeros(len(contours))
    extBotY = np.zeros(len(contours))
        
    for i in range(0,len(contours)):
        extLeftX[i] = extLeft[i][0]
        extLeftY[i] = extLeft[i][1]
        
        extRightX[i] = extLeft[i][0]
        extRightY[i] = extLeft[i][1]
        
        extTopX[i] = extLeft[i][0]
        extTopY[i] = extLeft[i][1]
        
        extBotX[i] = extLeft[i][0]
        extBotY[i] = extLeft[i][1]
        
    extLeftXY = np.stack((extLeftX, extLeftY), axis=1)
    index = np.argmin(extLeftXY, axis=0)[0]
    extLeftMOST = extLeftXY[index,:]
    
    extRightXY = np.stack((extRightX, extRightY), axis=1)
    index = np.argmax(extRightXY, axis=0)[0]
    extRightMOST = extRightXY[index,:]
    
    extTopXY = np.stack((extTopX, extTopY), axis=1)
    index = np.argmax(extTopXY, axis=0)[1]
    extTopMOST = extTopXY[index,:]
    
    extBotXY = np.stack((extBotX, extBotY), axis=1)
    index = np.argmin(extBotXY, axis=0)[1]
    extBotMOST = extBotXY[index,:]

    #####Finds extreme points for largest contour
    # cnt = max(contours, key=cv2.contourArea)
    # extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
    # extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
    # extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    # extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
    
    # int_array = float_array.astype(int)
    
    #Accomodate for full image, not just ROI
    extLeftFULL = tuple([extLeftMOST.astype(int)[0]+x1,extLeftMOST.astype(int)[1]+y1])
    extRightFULL = tuple([extRightMOST.astype(int)[0]+x1,extRightMOST.astype(int)[1]+y1])
    extTopFULL = tuple([extTopMOST.astype(int)[0]+x1,extTopMOST.astype(int)[1]+y1])
    extBotFULL = tuple([extBotMOST.astype(int)[0]+x1,extBotMOST.astype(int)[1]+y1])
    
    
    # cv2.imshow("bg_rem_clean", bg_rem_clean)
    # cv2.waitKey(1)
    
    
    binary = bg_rem_clean/255
    # cv2.imshow("bg_rem_clean", binary)
    R = frameCOL[:,:,0] * binary
    G = frameCOL[:,:,1] * binary
    B = frameCOL[:,:,2] * binary
    RGB = np.dstack((R,G,B))
    bg_rem_clean = np.uint8(RGB)
    # cv2.imshow("bg_rem_clean", bg_rem_clean)
    

    
    # cv2.drawContours(bg_rem_clean, [cnt], -1, (0, 255, 255), 2)
    cv2.circle(bg_rem_clean, extLeftFULL, 8, (0, 0, 255), -1)
    cv2.circle(bg_rem_clean, extRightFULL, 8, (0, 255, 0), -1)
    cv2.circle(bg_rem_clean, extTopFULL, 8, (255, 0, 0), -1)
    cv2.circle(bg_rem_clean, extBotFULL, 8, (255, 255, 0), -1)
    # show the output image
    cv2.imshow("Image", bg_rem_clean)
    
    LEFTcoords[int(frameno)] = extLeftFULL
    RIGHTcoords[int(frameno)] = extRightFULL
    TOPcoords[int(frameno)] = extTopFULL
    BOTTOMcoords[int(frameno)] = extBotFULL
    
    TOP[int(frameno)] = extTopFULL[1]
    BOTTOM[int(frameno)] = extBotFULL[1]
    
    
    ##### Tracker
    
    while first_iter2:
        tracker = cv2.TrackerKCF_create()
        
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        tracker_type = tracker_types[1]
    
     
        if tracker_type == 'BOOSTING':

            tracker = cv2.TrackerBoosting_create()

        if tracker_type == 'MIL':

            tracker = cv2.TrackerMIL_create()

        if tracker_type == 'KCF':

            tracker = cv2.TrackerKCF_create()

        if tracker_type == 'TLD':

            tracker = cv2.TrackerTLD_create()

        if tracker_type == 'MEDIANFLOW':

            tracker = cv2.TrackerMedianFlow_create()

        if tracker_type == 'GOTURN':

            tracker = cv2.TrackerGOTURN_create()

        if tracker_type == 'MOSSE':

            tracker = cv2.TrackerMOSSE_create()

        if tracker_type == "CSRT":

            tracker = cv2.TrackerCSRT_create()
    
        # Define an initial bounding box
        bbox = (515, 232, 61, 75)
        # Uncomment the line below to select a different bounding box
        # bbox = cv2.selectROI(bg_rem_clean, False)
        # Initialize tracker with first frame and bounding box
        ret = tracker.init(bg_rem_clean, bbox)
        first_iter2 = False
    
    
    
    # Start timer
    timer = cv2.getTickCount()
    # Update tracker
    ret, bbox = tracker.update(bg_rem_clean)
    bboxRec[int(frameno)-1] = bbox
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    # Draw bounding box
    if ret:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        # cv2.rectangle(bg_rem_clean, p1, p2, (255,0,0), 2, 1)
        centroid[int(frameno)] = np.average([p1[0],p2[0]])
    else :
        # Tracking failure
        # cv2.putText(bg_rem_clean, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        centroid[int(frameno)] = np.nan

    # # Display tracker type on frame
    # cv2.putText(bg_rem_clean, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    # cv2.putText(bg_rem_clean, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", bg_rem_clean)
    cv2.waitKey(1)
      
        
        

#cv2.imshow("result", result)
# cv2.imwrite("averaged_frame.jpg", result)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()













##### Calculate velocity and acceleration

# from calibrationFactor import calibrationFactor


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except (ValueError):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



##### Calculating Speed NEW

TOPtrim = np.trim_zeros(TOP)
BOTTOMtrim = np.trim_zeros(BOTTOM)

for i in range(1,len(TOPtrim)):
    if TOPtrim[i] < TOPtrim[i-1]:
        TOPtrim[i] = TOPtrim[i-1]

for i in range(0,5):
    TOPtrim[i] = TOPtrim[i] + 27    #Eliminates bent over position for first 4 yards
                                    #Can now treat athlete at same height throughout
# TOPtrim[range(0,3)] = [301,302,303]

# Yardage = np.linspace(0,37.5,len(TOPtrim))
Yardage = np.linspace(0,40,len(TOPtrim))
plt.plot(Yardage,TOPtrim)
plt.plot(Yardage,BOTTOMtrim)

TOPclean = savitzky_golay(TOPtrim, 51, 2)
BOTTOMclean = savitzky_golay(BOTTOMtrim, 51, 2)

# #####Predicts TOPclean for missing values based on clean polynomial
# frames = np.linspace(0,len(TOPclean)-1,len(TOPclean)) #Use instead of yardage on x-axis
# a, b, c, d = np.polyfit(frames,TOPclean, 3)
# x_out = np.linspace(114, 120, 120-114+1)   # choose 20 points, 10 in, 10 outside original range
# y_pred = np.polyval([a, b, c, d], x_out)    # y_pred refers to predicted values of y
# TOPpred = np.concatenate((TOPclean, y_pred), axis=0)
# #####Predicts BOTTOMclean for missing values based on clean polynomial
# # frames = np.linspace(0,len(BOTTOMclean)-1,len(BOTTOMclean)) #Use instead of yardage on x-axis
# a, b, c, d = np.polyfit(frames,BOTTOMclean, 3)
# x_out = np.linspace(114, 120, 120-114+1)   # choose 20 points, 10 in, 10 outside original range
# y_pred = np.polyval([a, b, c, d], x_out)    # y_pred refers to predicted values of y
# BOTpred = np.concatenate((BOTTOMclean, y_pred), axis=0)

plt.plot(Yardage,TOPclean)
plt.plot(Yardage,BOTTOMclean)

# Yardage = np.linspace(0,40,len(TOPpred))    #Forces distances back into 40yards
# TOPpred = savitzky_golay(TOPpred, 51, 2)
# plt.plot(Yardage,TOPpred)
# BOTpred = savitzky_golay(BOTpred, 51, 2)
# plt.plot(Yardage,BOTpred)

TOPpred = TOPclean
BOTpred = BOTTOMclean

initialHeight = round(TOPpred[0] - BOTpred[0])
endHeight = round(TOPpred[-1] - BOTpred[-1])


ratio = 40 / (endHeight - initialHeight)

dist = (TOPpred - BOTpred) * ratio

##### Forces back between 0 and 40 yards
distSUB = dist - dist[0]
distSUB[distSUB<0] = 0
distSUB = np.sqrt(distSUB)      #Take sqrt for inverse distance law 1/r^2
distClean = (distSUB / distSUB[-1]) * 40


wCurve = WeightedCurve()

# distW = distClean * wCurve
# distW = (distClean + (2*wCurve)) / 3
distW = wCurve

plt.plot(distW)
plt.plot(distClean)

distClean = distW

plt.close('all')

###############

# CFmetres,CFyards = calibrationFactor(measurements)

# centroid_clean = centroid[~(centroid==0)]
# yhat = savitzky_golay(centroid_clean, 51, 3) # window size 51, polynomial order 3

x_s = (np.array(range(0,len(distClean))))/FPS
y_m = distClean * 0.9144    #Convert from yards to metres
y_y = distClean



if Units == 1:
    
    plt.plot(x_s,(y_y*1.09361))
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (metres)')
    plt.savefig('Displacement.png')
    
    plt.figure(2)
    plt.plot(x_s,abs(y_y*1.09361))
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (yards)')
    plt.savefig('Distance.png')
    
elif Units == 2:
    plt.plot(x_s,y_y)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (yards)')
    plt.savefig('Displacement.png')
    
    plt.figure(2)
    plt.plot(x_s,abs(y_y))
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (yards)')
    plt.savefig('Distance.png')  







##### Calculates Velocity


if Units == 1:
    #####In meters
    vel = np.diff(y_m)/(1/FPS)
    
    # vhat = savitzky_golay(vel, 51, 3)
    vhat = vel
    plt.figure(3)
    plt.plot(x_s[:-1],vhat,'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.savefig('Velocity.png')
    
    # vhat = savitzky_golay(vel, 51, 3)
    vhat = vel
    plt.figure(4)
    plt.plot(x_s[:-1],abs(vhat),'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.savefig('Speed.png')
    
    vhat[np.isnan(vhat)] = 0
    vel_ms = np.max(abs(vhat))      #Max velocity (m/s)
    vel_mph = round((vel_ms * 2.23694),1)              #Max velocity (mph)
    print('Maximum Speed:',vel_mph,'mph')

elif Units == 2:
    #####In yards
    vel = np.diff(y_y)/(1/FPS)
    
    # vhat = savitzky_golay(vel, 51, 3)
    vhat = vel
    plt.figure(3)
    plt.plot(x_s[:-1],vhat,'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (y/s)')
    plt.savefig('Velocity.png')
    
    # vhat = savitzky_golay(vel, 51, 3)
    vhat = vel
    plt.figure(4)
    plt.plot(x_s[:-1],abs(vhat),'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (y/s)')
    plt.savefig('Speed.png')
    
    vhat[np.isnan(vhat)] = 0
    vel_ys = round(np.max(abs(vhat)),1)      #Max velocity (m/s)
    #vel_mph = round((vel_ys),1)              #Max velocity (mph)
    print('Maximum Speed:',vel_ys,'y/s')




##### Calculates acceleration
# a = (v - u) / t
vtrim = vhat[~(vhat==0)]
a = np.zeros(len(vtrim))
for ii in range(1,len(vtrim)):
    a[ii] = (vtrim[ii] - vtrim[ii-1]) / (1/FPS)     #Acceleration between each frame (m/s/s)



ahat = a
# ahat = savitzky_golay(a, 51, 4)
plt.figure(5)
plt.plot(x_s[:len(ahat[1:-1])],ahat[1:-1],'g')
plt.xlabel('Time (s)')
if Units == 1:
    plt.ylabel('Acceleration (m/s/s)')
    acc_mss = round(np.max(abs(ahat)),1)    #Max acceleration in m/s/s
    print('Maximum Acceleration:',acc_mss,'m/s/s')
elif Units == 2:
    plt.ylabel('Acceleration (y/s/s)')
    acc_mss = round(np.max(abs(ahat)),1)    #Max acceleration in m/s/s
    print('Maximum Acceleration:',acc_mss,'y/s/s')
plt.savefig('Acceleration.png')

plt.figure(6)
plt.plot(x_s[:len(ahat[1:-1])],ahat[1:-1],'g')
plt.xlabel('Time (s)')
if Units == 1:
    plt.ylabel('Absolute Acceleration (m/s/s)')
    acc_mss = round(np.max(abs(ahat)),1)    #Max acceleration in m/s/s
    # print('Maximum Acceleration:',acc_mss,'m/s/s')
elif Units == 2:
    plt.ylabel('Absolute Acceleration (y/s/s)')
    acc_mss = round(np.max(abs(ahat)),1)    #Max acceleration in m/s/s
    # print('Maximum Acceleration:',acc_mss,'y/s/s')
plt.savefig('Absolute Acceleration.png')



##### Create Video #####


#####To make the video look prettier on the end frames #####
a = 0
b = 0    
for i in range(111,116):
    a = a + 25
    X = TOPcoords[111][0] + a
    b = b + 20
    Y = TOPcoords[111][1] + b
    TOPcoords[i] = (X,Y)
    
a = 0
b = 0    
for i in range(113,116):
    a = a + 60
    X = BOTTOMcoords[112][0] + a
    b = b - 5
    Y = BOTTOMcoords[112][1] + b
    BOTTOMcoords[i] = (X,Y)
    
X = int((BOTTOMcoords[106][0] + BOTTOMcoords[108][0]) / 2)
Y = int((BOTTOMcoords[106][1] + BOTTOMcoords[108][1]) / 2)
BOTTOMcoords[107] = (X,Y)


vhat = np.append(vhat,vhat[-1])
vhat = np.append(vhat,vhat[-1])

ahat = np.append(ahat,vhat[-1])
ahat = np.append(ahat,vhat[-1])

##########

cap = cv2.VideoCapture(file_path)


missv = frameTotal - len(vhat)
vfull = np.pad(vhat, (0, missv), 'constant')
vfull = np.insert(vfull,0,0)

missa = frameTotal - len(ahat)
afull = np.pad(ahat, (0, missa), 'constant')
afull = np.insert(afull,0,0)

# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (width,height))
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m', 'p','4','v'), FPS, (width,height))
 




while True:
    ret, frame = cap.read()
    
    frameno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  

    
    
    bbox = bboxRec[frameno-1]
    
    # Draw bounding box
    if ret:
        cv2.circle(frame, TOPcoords[frameno], 8, (255, 0, 0), -1)
        cv2.circle(frame, BOTTOMcoords[frameno], 8, (255, 255, 0), -1)
        # # Tracking success
        # p1 = (int(bbox[0]), int(bbox[1]))
        # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        # # centroid[int(frameno)] = np.average([p1[0],p2[0]])
    else :
        # # Tracking failure
        # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        # # centroid[frameno] = np.nan
        pass
    
    if Units == 1:
        cv2.putText(frame, "Velocity " + str("{:.1f}".format(abs(vfull[frameno-1]* 2.23694))) + "mph", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0),5);

        cv2.putText(frame, "Acceleration " + str("{:.1f}".format(abs(afull[frameno-1]))) + "m/s/s", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5);
    
    elif Units == 2:
        
        cv2.putText(frame, "Velocity " + str("{:.1f}".format(abs(vfull[frameno-1]))) + "y/s", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0),5);

        cv2.putText(frame, "Acceleration " + str("{:.1f}".format(abs(afull[frameno-1]))) + "y/s/s", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5);
        
    cv2.waitKey(1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):       #Press q to break
        break
    
    if frame is None:
        break
    
    out.write(frame)
    
    cv2.imshow('frame',frame)
    

cap.release()
out.release()
cv2.destroyAllWindows()
plt.close('all')


