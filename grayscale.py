import cv2

img = cv2.imread('img.jpg');        # read color/rgb image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # convert rgb image to gray

width = img.shape[0] # Get the dimensions
height = img.shape[1]

x=width//2
y=height//2
z=min(width,height)//3
minx=x-z
maxx=x+z
miny=y-z
maxy=y+z

for i in range(0,minx):
    for j in range(0,height):
        img[i,j]=gray[i,j]
        
for i in range(maxx,width):
    for j in range(0,height):
        img[i,j]=gray[i,j]

for i in range(0,width):
    for j in range(0,miny):
        img[i,j]=gray[i,j]

for i in range(0,width):
    for j in range(maxy,height):
        img[i,j]=gray[i,j]
            

"""
ind = np.int(gray.shape[1]/2)       # get width/2 value of the image for indexing
img[:,0:ind,0] = gray[:,0:ind]      # make blue component value equal to gray image
img[:,0:ind,1] = gray[:,0:ind]      # make green component value equal to gray image
img[:,0:ind,2] = gray[:,0:ind]      # make red component value equal to gray image
"""
#cv2.imshow('Result',img)        # show image result
cv2.imwrite("output.jpg",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()