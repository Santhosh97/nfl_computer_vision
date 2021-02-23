#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def averageFrame(cap):
    
    first_iter = True
    result = None
    
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        
        if first_iter: # first iteration of the while loop
            avg = np.float32(frame)
            first_iter = False
            
        cv2.accumulateWeighted(frame, avg, 0.005)
    
        result = cv2.convertScaleAbs(avg)
    
    # cv2.imshow("Result", result)

    # cv2.imwrite("averaged_frame.jpg", result)
    
    # cv2.waitKey(0)

    # # When everything done, release the capture
    cap.release()
    # cv2.destroyAllWindows()
    
    return result