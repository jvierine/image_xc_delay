#!/usr/bin/env python3

import numpy as n
import cv2
import matplotlib.pyplot as plt
import stuffr

# blink_100ms
#area0=[[500,502],[50,250]]
#area1=[[500,502],[1165,1600]]

# blink_010ms
#area0=[[500,502],[250,580]]
#area1=[[500,502],[1365,1650]]

# blink_002ms
area0=[[688,689],[440,441]]
area1=[[307,308],[1458,1459]]



#cap = cv2.VideoCapture("blink_100ms.mov")
#cap = cv2.VideoCapture("blink_010ms.mov")
cap = cv2.VideoCapture("blink_002ms.mov")
frames_per_sec=cap.get(cv2.CAP_PROP_FPS)
n_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)

#print(cv2.GetCaptureProperty(cap,cv2.CV_CAP_PROP_FPS))
#print("fps %1.2f"%(cap.get(cv2.CV_CAP_PROP_FPS)))
I0=[]
I1=[]
fi=0
plot_frames=False
while(cap.isOpened()):
    print("%d/%d"%(fi,n_frames))
    fi+=1
    re,frame = cap.read()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        I0.append(n.sum(gray[area0[0][0]:area0[0][1],area0[1][0]:area0[1][1] ]))
        I1.append(n.sum(gray[area1[0][0]:area1[0][1],area1[1][0]:area1[1][1] ]))
    except:
        print("err")
        cap.release()
        pass

    if plot_frames:
        plt.pcolormesh(gray)
        plt.colorbar()
        plt.show()
        
I0=n.array(I0)
I1=n.array(I1)


n_frames=len(I0)
plt.plot(I0)
plt.plot(I1)
plt.show()


# incoherent averaging of cross-spectra
XC=stuffr.decimate(n.fft.fftshift(n.fft.fft(I0)*n.conj(n.fft.fft(I1))),dec=10)

# frequencies
freqs=stuffr.decimate(n.fft.fftshift(n.fft.fftfreq(n_frames,d=1.0/frames_per_sec)),dec=10)
plt.plot(freqs,n.abs(XC))
plt.title("Cross-specral power")
plt.show()

plt.plot(freqs,n.angle(XC),".")
# time delay
tau = -2e-3
plt.plot(freqs,n.mod(2*n.pi*freqs*tau+n.pi,2*n.pi)-n.pi)
plt.title("Cross-phase")
plt.show()
