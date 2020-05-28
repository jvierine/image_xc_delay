#!/usr/bin/env python3

import numpy as n
import cv2
import matplotlib.pyplot as plt
import stuffr
import scipy.signal as ss

def fit_phase(freq,phase):
    n_meas=len(freq)
    A=n.zeros([n_meas,1])
    A[:,0]=2*n.pi*freq
    tau=n.linalg.lstsq(A,phase)[0]

    model=n.dot(A,tau)
    var_est=n.mean(n.abs(model-phase)**2.0)
    post_var=var_est*n.linalg.inv(n.dot(n.transpose(A),A))[0,0]
    est_std=n.sqrt(post_var)
    return(tau,est_std,model)

def estimate_delay(fname="blink_002ms.mov",
                   max_freq=8.0,
                   min_freq=0.5,
                   dec=10,
                   area0=[688,[440,441]],
                   fft_len=512,
                   plot_frames=False,
                   area1=[307,[1458,1459]]):

    
    
    # blink_002ms
    
    #cap = cv2.VideoCapture("blink_100ms.mov")
    #cap = cv2.VideoCapture("blink_010ms.mov")
    cap = cv2.VideoCapture(fname)
    frames_per_sec=cap.get(cv2.CAP_PROP_FPS)
    n_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    #print(cv2.GetCaptureProperty(cap,cv2.CV_CAP_PROP_FPS))
    #print("fps %1.2f"%(cap.get(cv2.CV_CAP_PROP_FPS)))
    I0=[]
    I1=[]
    fi=0

    while(cap.isOpened()):
        print("%d/%d"%(fi,n_frames))
        fi+=1
        re,frame = cap.read()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(gray.shape)
            I0.append(n.sum(gray[area0[0],area0[1][0]:area0[1][1] ]))
            I1.append(n.sum(gray[area1[0],area1[1][0]:area1[1][1] ]))
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

    XCA=n.zeros(fft_len,dtype=n.complex128)
    n_frames=len(I0)
    n_windows=int(n.floor(n_frames/fft_len))
    freqs=n.fft.fftshift(n.fft.fftfreq(fft_len,d=1/frames_per_sec))
    for i in range(n_windows):
        XCA+=n.fft.fftshift(n.fft.fft(I0[(i*fft_len):((i+1)*fft_len)])*n.conj(n.fft.fft(I1[(i*fft_len):((i+1)*fft_len)])))
    
    t=1e3*n.arange(n_frames)/frames_per_sec
    plt.plot(t,I0,label="$I_1(t)$")
    plt.plot(t,I1,label="$I_1(t)$")
    plt.legend()
    plt.ylabel("Image intensity")
    plt.xlabel("Time (ms)")
    plt.show()
    
    plt.subplot(121)
    plt.plot(freqs,10.0*n.log10(n.abs(XCA)**2.0))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Cross-spectral power (dB)")
    plt.title("Cross-specral power")
    
    plt.subplot(122)
    
    zi=n.argmin(n.abs(freqs))
    phase=n.unwrap(n.angle(XCA))
    phase=phase-phase[zi]
    
    fidx=n.where( (n.abs(freqs)<max_freq)  & (n.abs(freqs)>min_freq) )[0]
    tau,tau_std,model=fit_phase(freqs[fidx],phase[fidx])
    print("tau %1.2f+/-%1.2f"%(tau*1e6,tau_std*1e6))
    
    plt.plot(freqs,phase,".",label="All measurements")
    plt.plot(freqs[fidx],phase[fidx],".",label="Fitted measurements")
    plt.ylabel("Phase (rad)")
    plt.xlabel("Frequency (Hz)")
    # time delay
    #tau = 100e-3
    plt.plot(freqs,2*n.pi*freqs*tau,label="Best fit")
    plt.plot(freqs,2*n.pi*freqs*(tau+tau_std),color="gray")
    plt.plot(freqs,2*n.pi*freqs*(tau-tau_std),color="gray")
    plt.title("Cross-phase")
    plt.legend()
    plt.show()

    plt.plot(freqs[fidx],phase[fidx]-model,".")
    plt.show()
    return(tau,tau_std)


estimate_delay(fname="blink_5ms_juha.mp4",max_freq=6.0,min_freq=0.5,dec=10,area0=[441,[506,853]],area1=[441,[1636,2137]])
#estimate_delay(fname="blink_002ms.mov",max_freq=8.0,min_freq=0.5,dec=10,area0=[688,[440,441]],area1=[307,[1458,1459]])

#estimate_delay(fname="blink_100ms.mov",max_freq=8.0,min_freq=0.5,dec=10,area0=[500,[100,101]],area1=[500,[1500,1501]])

#estimate_delay(fname="blink_010ms.mov",max_freq=8.0,min_freq=0.5,dec=10,area0=[500,[250,500]],area1=[500,[1365,1650]])
