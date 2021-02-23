#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def WeightedCurve():
    
    dist = [0,5,10,15,20,25,30,35,40]
    frame = [0,25,38,54,66,76,87,100,112]
    
    CF = 4.68/frame[-1]    #4.68 seconds
    
    time = np.array(frame)*CF
    
    
    plt.plot(time,dist)
    
    
    a, b, c, d = np.polyfit(frame,dist, 3)
    x_out = np.linspace(0, 113, 113)
    ypred = np.polyval([a, b, c, d], x_out)    # y_pred refers to predicted values of y
    # pred = np.concatenate((TOPclean, y_pred), axis=0)
    time = np.linspace(0, 5, 113)
    plt.plot(time,ypred)
    
    # wCurve = ypred/ypred[-1]
    wCurve = ypred
    
    return wCurve