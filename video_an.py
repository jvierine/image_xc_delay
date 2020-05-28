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
    post_var=var_est*n.linalg.inv(n.dot(n.transpose(A),A))
    est_std=n.sqrt(post_var)
    return(tau,est_std,model)
    
max_freq=15.0
dec=10
# blink_100ms
area0=[[500,501],[100,101]]
area1=[[500,501],[1500,1501]]

# blink_010ms
#area0=[[500,502],[250,580]]
#area1=[[500,502],[1365,1650]]

# blink_002ms
#area0=[[688,689],[440,441]]
#area1=[[307,308],[1458,1459]]



cap = cv2.VideoCapture("blink_100ms.mov")
#cap = cv2.VideoCapture("blink_010ms.mov")
#cap = cv2.VideoCapture("blink_002ms.mov")
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
t=1e3*n.arange(n_frames)/frames_per_sec
plt.plot(t,I0,label="$I_1(t)$")
plt.plot(t,I1,label="$I_1(t)$")
plt.legend()
plt.ylabel("Image intensity")
plt.xlabel("Time (ms)")
plt.show()


# incoherent averaging of cross-spectra
w=1.0#ss.hann(len(I0))
XC=stuffr.decimate(n.fft.fftshift(n.fft.fft(w*I0)*n.conj(n.fft.fft(w*I1))),dec=dec)

# frequencies
freqs=stuffr.decimate(n.fft.fftshift(n.fft.fftfreq(n_frames,d=1.0/frames_per_sec)),dec=dec)
plt.subplot(121)
plt.plot(freqs,10.0*n.log10(n.abs(XC)**2.0))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Cross-spectral power (dB)")
plt.title("Cross-specral power")

plt.subplot(122)

zi=n.argmin(n.abs(freqs))
phase=n.unwrap(n.angle(XC))
phase=phase-phase[zi]

fidx=n.where(n.abs(freqs)<max_freq)[0]
tau,tau_std,model=fit_phase(freqs[fidx],phase[fidx])
print("tau %1.2f+/-%1.2f"%(tau*1e6,tau_std*1e6))

plt.plot(freqs,phase,".")
plt.ylabel("Phase (rad)")
plt.xlabel("Frequency (Hz)")
# time delay
tau = 100e-3
plt.plot(freqs,2*n.pi*freqs*tau)
plt.title("Cross-phase")
plt.show()

plt.plot(freqs[fidx],phase[fidx]-model,".")
plt.show()
