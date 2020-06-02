#!/usr/bin/env python3

import numpy as n
import cv2
import matplotlib.pyplot as plt
import stuffr
import scipy.signal as ss

def fit_phase(freq,phase,pwr,var_fit=0.0059**2.0,use_var_fit=False):
    n_meas=len(freq)
    A=n.zeros([n_meas,1])
    A[:,0]=2*n.pi*freq
    tau=n.linalg.lstsq(A,phase)[0]

    model=n.dot(A,tau)

    # phase error std is proportional to (pwr)
#    plt.plot((model-phase)/pwr**0.25,".")
 #   plt.show()
    var_est=n.mean(n.abs(model-phase)**2.0)
    if use_var_fit:
        vat_fit=var_est
        
    post_var=var_fit*n.linalg.inv(n.dot(n.transpose(A),A))[0,0]
    est_std=n.sqrt(post_var)
    return(tau,est_std,model,var_est)

def estimate_delay(fname="blink_002ms.mov",
                   max_freq=8.0,
                   min_freq=2.0,
                   dec=10,
                   area0=[688,[440,441]],
                   fft_len=512,
                   plot_frames=False,
                   n_frames=0,
                   unwrap=False,
                   weight=False,
                   area1=[307,[1458,1459]]):

    
    
    # blink_002ms
    
    #cap = cv2.VideoCapture("blink_100ms.mov")
    #cap = cv2.VideoCapture("blink_010ms.mov")
    cap = cv2.VideoCapture(fname)
    frames_per_sec=cap.get(cv2.CAP_PROP_FPS)
    n_total_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if n_frames == 0:
        n_frames=n_total_frames
    print(n_total_frames)
    print(frames_per_sec)
    #print(cv2.GetCaptureProperty(cap,cv2.CV_CAP_PROP_FPS))
    #print("fps %1.2f"%(cap.get(cv2.CV_CAP_PROP_FPS)))
    I0=[]
    I1=[]
    fi=0

    while(cap.isOpened()):
        print("%d/%d"%(fi,n_frames))
        if fi > n_frames:
            cap.release()
            break
        fi+=1
        
#        print(frame.shape)
        try:
            re,frame = cap.read()
            # green color led, green color channel
            gray=frame[:,:,1]
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
    I0=I0-n.mean(I0)
    I1=n.array(I1)
    I1=I1-n.mean(I1)

#    
#    XCA=n.zeros(fft_len,dtype=n.complex128)
    n_frames=len(I0)
#    n_windows=int(n.floor(n_frames/fft_len))
    

    XC=n.fft.fftshift(n.fft.fft(I0)*n.conj(n.fft.fft(I1)))
    
    XCA=stuffr.decimate(XC,dec=dec)

    if weight:
        # weight by power    
        pw=n.abs(XC)**2.0
        ws=stuffr.decimate(pw,dec=dec)
        freqs=stuffr.decimate(pw*n.fft.fftshift(n.fft.fftfreq(n_frames,d=1/frames_per_sec)),dec=dec)/ws
    else:
        freqs=stuffr.decimate(n.fft.fftshift(n.fft.fftfreq(n_frames,d=1/frames_per_sec)),dec=dec)

#    for i in range(n_windows):
 #       XCA+=n.fft.fftshift(n.fft.fft(I0[(i*fft_len):((i+1)*fft_len)])*n.conj(n.fft.fft(I1[(i*fft_len):((i+1)*fft_len)])))
    
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
    
    if unwrap:
        phase=n.unwrap(n.angle(XCA))
        phase=phase-n.mean(phase)
        plt.plot(freqs,phase,".")
        plt.show()
    else:
        phase=n.angle(XCA)
#    phase=phase-phase[zi]
#    phase=n.angle(XCA)
    
    fidx=n.where( (n.abs(freqs)<max_freq)  & (freqs>min_freq) )[0]
    tau,tau_std,model,var_est=fit_phase(freqs[fidx],phase[fidx],n.abs(XCA[fidx]))
    print("tau %1.2f+/-%1.2f measurement error std estimate %f"%(tau*1e6,tau_std*1e6,n.sqrt(var_est)))
    
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


# same blinker
#estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_b.mp4",max_freq=8.0,min_freq=2,dec=10,area0=[374,[510,511]],area1=[374,[600,601]],plot_frames=True,n_frames=1000)


# same light
estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_eb.mp4",max_freq=10.0,min_freq=0,dec=40,area0=[266,[560,561]],area1=[266,[565,566]],plot_frames=False,n_frames=0,weight=True,unwrap=False)
estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_eb.mp4",max_freq=10.0,min_freq=0,dec=40,area0=[266,[560,561]],area1=[266,[565,566]],plot_frames=False,n_frames=300,weight=True,unwrap=False)
estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_eb.mp4",max_freq=10.0,min_freq=0,dec=40,area0=[266,[560,561]],area1=[266,[565,566]],plot_frames=False,n_frames=1000,weight=True,unwrap=False)
estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_eb.mp4",max_freq=10.0,min_freq=0,dec=40,area0=[266,[560,561]],area1=[266,[565,566]],plot_frames=False,n_frames=3000,weight=True,unwrap=False)
estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_eb.mp4",max_freq=10.0,min_freq=0,dec=40,area0=[266,[560,561]],area1=[266,[565,566]],plot_frames=False,n_frames=10000,weight=True,unwrap=False)



# different frame
estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_eb.mp4",max_freq=10.0,min_freq=0,dec=40,area0=[266,[560,561]],area1=[266,[1800,1801]],plot_frames=False,n_frames=300,weight=True,unwrap=False)
estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_eb.mp4",max_freq=10.0,min_freq=0,dec=40,area0=[266,[560,561]],area1=[266,[1800,1801]],plot_frames=False,n_frames=1000,weight=True,unwrap=False)
estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_eb.mp4",max_freq=10.0,min_freq=0,dec=40,area0=[266,[560,561]],area1=[266,[1800,1801]],plot_frames=False,n_frames=3000,weight=True,unwrap=False)
estimate_delay(fname="/data0/blink_cam_cal/blink_1ms_0deg_eb.mp4",max_freq=10.0,min_freq=0,dec=40,area0=[266,[560,561]],area1=[266,[1800,1801]],plot_frames=False,n_frames=10000,weight=True,unwrap=False)

#estimate_delay(fname="/data0/blink_cam_cal/blink_2ms_0deg_b.mp4",max_freq=8.0,min_freq=2,dec=10,area0=[374,[510,511]],area1=[374,[1930,1931]],plot_frames=False,n_frames=10000)

#estimate_delay(fname="/data0/blink_cam_cal/blink_2ms_0deg_b.mp4",max_freq=8.0,min_freq=2,dec=10,area0=[374,[510,511]],area1=[374,[1930,1931]],plot_frames=False,n_frames=10000)


#estimate_delay(fname="blink_002ms.mov",max_freq=6.0,min_freq=0.5,dec=10,area0=[688,[440,441]],area1=[307,[1458,1459]],n_frames=1000)

#estimate_delay(fname="blink_100ms.mov",max_freq=8.0,min_freq=0.5,dec=10,area0=[500,[100,101]],area1=[500,[1500,1501]],n_frames=1000)

#estimate_delay(fname="blink_010ms.mov",max_freq=6.0,min_freq=0.5,dec=20,area0=[500,[250,500]],area1=[500,[1365,1650]],n_frames=0)
