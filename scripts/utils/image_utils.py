import cv2

#TODO: put the following three functions in their own file
# https://stackoverflow.com/a/58126805
def resize_img(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

'''
Get a specific frame from a capture. Returns the image itself.
'''
def get_frame(cap, frame = 0):
    curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    ret, img = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, curr - 1)
    if ret:
        return img 
    else:
        return RuntimeError("could not read frame number " + int(frame) + " from capture.")

'''
Display an image in a computer-friendly way (resize to normal screen dims)
'''
def show_frame(image, dims = [1280, 720], name = "",):
    image = resize_img(image, dims[0], dims[1])

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    