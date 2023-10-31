import cv2 

if __name__ == '__main__':
    #this is needed for windows - not sure about linux/mac systems.
    #VideoCapture(i), i = 0, 1, 2, ... might work. Depends on the specific computer you use 
    cam = cv2.VideoCapture(cv2.CAP_DSHOW)

    if not cam.isOpened():
        print("error opening camera")
        exit()
    
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()
        # if frame is read correctly ret is True
        if not ret:
            print("error in retrieving frame")
            break
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame
        cv2.imshow('frame', img)
        # file.write(img)

        if cv2.waitKey(1) == ord('q'):
            break

cam.release()
# file.release()
cv2.destroyAllWindows()