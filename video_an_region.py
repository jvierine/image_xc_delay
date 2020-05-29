#!/usr/bin/env python3

import numpy as n
import cv2
import matplotlib.pyplot as plt
import stuffr
import h5py


def fit_delay(freq,xc):
    n_meas = len(freq)
    A=n.zeros([n_meas,1])
    A[:,0] = 2.0*n.pi*freq
    xhat=n.linalg.lstsq(A,n.angle(xc))[0]
    return(xhat[0])


def delay_region(fname="blink_001ms.mp4",
                 ofname="tau_0.h5",
                 max_freq=10.0,
                 min_freq=1.0,
                 area0 = [400,500],
                 plot_frames=False,
                 n_frames = 10000,
                 area1 = [[300, 500],[1600, 1800]]):

    # use this frequency range to derive time delay
    cap = cv2.VideoCapture(fname)
    frames_per_sec=cap.get(cv2.CAP_PROP_FPS)
    n_total_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)

    I0=[]
    I1=[]
    fi=0
    
    while(cap.isOpened()):
        if n_frames > 0 and fi > n_frames:
            cap.release()
            break
        print("%d/%d (%d)"%(fi,n_frames,n_total_frames))
        fi+=1
        re,frame = cap.read()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(gray.shape)
            I0.append(gray[area0[0],area0[1]])
            I1.append(gray[area1[0][0]:area1[0][1],area1[1][0]:area1[1][1] ])
        except:
            print("err")
            cap.release()
            break
            pass

        if plot_frames:
            plt.pcolormesh(gray)
            plt.colorbar()
            plt.show()
            
    I0=n.array(I0)
    I1=n.array(I1)
    I0=I0-n.mean(I0)
    I1=I1-n.mean(I1)
    n_frames=len(I0)
    delays = n.zeros([I1.shape[1],I1.shape[2]])

    for i in range(delays.shape[0]):
        for j in range(delays.shape[1]):
            # incoherent averaging of cross-spectra
            XC=stuffr.decimate(n.fft.fftshift(n.fft.fft(I0)*n.conj(n.fft.fft(I1[:,i,j]))),dec=10)
            # frequencies
            freqs=stuffr.decimate(n.fft.fftshift(n.fft.fftfreq(n_frames,d=1.0/frames_per_sec)),dec=10)
            fidx=n.where( (n.abs(freqs) < max_freq) & (n.abs(freqs)> min_freq) )[0]
            tau_meas=fit_delay(freqs[fidx],XC[fidx])
            delays[i,j]=tau_meas

    delta_y = n.arange(delays.shape[0])
    delta_x = n.arange(delays.shape[1])
    ho=h5py.File(ofname,"w")
    ho["tau"]=1e-3
    ho["delta_tau_s"]=delays
    pdx=delta_x+area1[1][0]-area0[1]
    pdy=delta_y+area1[0][0]-area0[0]
    ho["dx_pixel"]=pdx
    ho["dy_pixel"]=pdy
    ho["n_frames"]=n_frames
    ho.close()
    plt.pcolormesh(pdx,pdy,delays*1e3)
    plt.xlabel("Image row difference (pixels)")
    plt.ylabel("Image column difference (pixels)")

    plt.title("Delay as a function of pixel location (ms)")
    plt.colorbar()
    plt.show()

delay_region(fname="blink_001ms.mp4", ofname="tau_1ms_0.h5", area0 = [470,600], n_frames = 10000, area1 = [[370, 570],[1660, 1860]], plot_frames=False)
#delay_region(fname="blink_001ms.mp4", ofname="tau_1ms.h5", area0 = [400,500], n_frames = 1000, area1 = [[300, 500],[1600, 1800]], plot_frames=False)
#delay_region(fname="blink_001ms.mp4", ofname="tau_1ms.h5", area0 = [400,500], n_frames = 1000, area1 = [[300, 500],[1600, 1800]])
#delay_region(fname="blink_001ms.mp4", ofname="tau_1ms.h5", area0 = [400,500], n_frames = 1000, area1 = [[300, 500],[1600, 1800]])
