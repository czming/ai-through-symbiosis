import numpy as np 
import cv2 
import argparse
from mlsocket import MLSocket


#notes: 
# * the dji action 2 only allows 1280x720 resolution wiht 30 fps when used as a webcam. when recording on its own, we can do 4k x 60fps.
# this code works on ubuntu.
# run python cv2_live_ubuntu.py and it automatically records as webcam hooked up to laptop. Then when done, press a and the video will save.

HOST = "127.0.0.1"
PORT = 48293

if __name__ == '__main__':

    socket = MLSocket()

    socket.connect((HOST, PORT))
        
    # This will return video from the first webcam on your computer. 
    cap = cv2.VideoCapture(0)   

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    fps =  cap.get(cv2.CAP_PROP_FPS)
    print("video in " + str(width) + " x " + str(height) + " resolution at " + str(fps) + " fps.")

    # Define the codec and create VideoWriter object 
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('picklist_41_mock.mp4', fourcc, fps, (width, height)) 
    
    img = np.zeros((640, 480))

    ret, _ = cap.read()

    if ret:
        # loop runs if capturing has been initialized.  
        while(True): 
            # reads frames from a camera  
            # ret checks return at each frame 
            ret, frame = cap.read()  
                    
            # output the frame 
            # out.write(frame)  
            # socket.send(frame)
            
            # The original input frame is shown in the window  
            # cv2.imshow('Press a to quit', img) 
        
            cv2.imshow('press a to quit', cv2.resize(frame, (426, 240)))
            # The window showing the operated video stream  
            # cv2.imshow('frame', hsv) 

            if frame.size > 0:
                socket.send(frame)
        
            
            # Wait for 'a' key to stop the program  
            if cv2.waitKey(1) & 0xFF == ord('a'): 
                break
        

    print("video closed")
    # Close the window / Release webcam 
    cap.release() 
    socket.send(np.zeros((640, 480, 3)))
    
    # After we release our webcam, we also release the output 
    # out.release()  
    
    # De-allocate any associated memory usage
    cv2.destroyAllWindows() 