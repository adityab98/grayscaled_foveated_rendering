import numpy as np
import cv2
#import ffmpeg
#import sys
#import skvideo.io
import math

def distance(x1,y1,x2,y2):
    return math.sqrt(abs(y2-y1)**2+abs(x2-x1)**2)

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = img.shape[0] # Get the dimensions
    height = img.shape[1]
    x=width//2
    y=height//2
    radius=min(width,height)//2
    for i in range(0,width):
        for j in range(0, height):
            #if(ellipse(i,j,x,y,a,b)):
            if(distance(i,j,x,y)<=radius):
                continue
            else:
                img[i,j]=gray[i,j]                
    #cv2.imshow('image',img)
    #cv2.waitKey(1)
    return img

def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d

def pyramid(im, sigma=1, prNum=6):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]
    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)
    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)
    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im
    return pyramids

def foveate_img(im, fixs):
    sigma=0.248
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape
    p = 7.5
    k = 3
    alpha = 2.5
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)
    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma
    omega[omega>1] = 1
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))
    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]
        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]
    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])
    im_fov = im_fov.astype(np.uint8)
    return im_fov

if __name__ == "__main__":
    """vid_writer=skvideo.io.FFmpegWriter(output,outputdict={
        '-vcodec':'libx264','-b':'1290000'
    })"""
    #for i in range(2,8):
    output='c.avi'
    file='other/f1.avi'
    #print(output+file)
    cap = cv2.VideoCapture(file)
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    codec=cv2.VideoWriter_fourcc('M','J','P','G')
    #api=cv2.CAP_ANY
    api=cv2.CAP_FFMPEG
    vid_writer=cv2.VideoWriter(output,api,codec, 25, (int(width),int(height)))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        xc, yc = int(frame.shape[1]/2), int(frame.shape[0]/2)
        print("Foveating", end=' ')
        frame=foveate_img(frame,[(xc, yc)])
        print("Grayscaling")
        frame=grayscale(frame)
        #vid_writer.writeFrame(frame)
        vid_writer.write(frame)
    #cv2.destroyAllWindows()
    cap.release()
    #vid_writer.close()    
    vid_writer.release()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    