import numpy as np
import math
import Functions as f
import cv2

f.generate_phantom(8)
Phantom = cv2.imread("Phantom.png",0)
T1 = f.generate_array(Phantom,8,1)
T2 = f.generate_array(Phantom,8,2)


def T2Prep (Phantom, TR, TE, t1_array ,t2_array):
    
    RF1 = np.array([[1,0,0],[0,math.cos(math.pi/2),math.sin(math.pi/2)],[0,-math.sin(math.pi/2),math.cos(math.pi/2)]])     # Rotate around x-axis with angle 90
    RF2 = np.array([[1,0,0],[0,math.cos(-math.pi/2),math.sin(-math.pi/2)],[0,-math.sin(-math.pi/2),math.cos(-math.pi/2)]])             # Rotate around x-axis with angle 180
    Height,Width = np.shape(Phantom)
    T=50
    for i in range (Height):
        for j in range (Width):
            
            decayrecovery = np.array([[np.exp(-T/t2_array[i,j]),0,0],[0,np.exp(-T/t2_array[i,j]),0],[0,0,1-np.exp(-T/t1_array[i,j])]]) + np.array([0, 0, 1-np.exp(-T/t1_array[i,j])])

            Phantom[i, j, :] = np.dot(RF1, Phantom[i, j, :])  # Phantom after rotation around 90 around x-axis
            
            Phantom[i, j, :] = np.dot(decayrecovery, Phantom[i, j, :])  # Phantom after decay at x-y plane (on y-axis)

            Phantom[i, j, :] = np.dot(RF2, Phantom[i, j, :])  # Phantom after rotation around 90 then 180 around x-axis
            

    return Phantom
