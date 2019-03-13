import argparse
import numpy as np
import os, sys
from numpy import linalg as LA
from numpy import linalg as la
from matplotlib import pyplot as plt
import math
from PIL import Image
import random

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
#-------------------------------------------------------------------------------

def binary(A):
    for i in range(0,len(A)):
        for j in range(0,len(A[0])):
            if (A[i,j]>100):
                A[i,j]=1
            else:
                A[i,j]=0
    return A

def Superimposing(ctr,image,src):
    #print(ctr)
    pts_dst = np.array(ctr,dtype=float)
    pts_src = np.array([[0,0],[511, 0],[511, 511],[0,511]],dtype=float)

    h, status = cv2.findHomography(pts_src, pts_dst)

    #print(image.shape[1],image.shape[0])

    temp = cv2.warpPerspective(src, h,(image.shape[1],image.shape[0]));
    cv2.fillConvexPoly(image, pts_dst.astype(int), 0, 16);

    image = image + temp;

    return image,h

def perspective_view(image):
    #sequence tl,tr,br,bl
    dst1 = np.array([
        [0,720],
        [0, 0],
        [1280,0],
        [1280,720]], dtype = "float32")

    source=np.array([
        [0+150,720-80],
        [0+570, 450],
        [1280-570,450],
        [1280-50,720-80]], dtype = "float32")


    M1,status = cv2.findHomography(source, dst1)

    warp1 = cv2.warpPerspective(image, M1, (1280,720))
    #warp2=cv2.medianBlur(warp1,3)
    blur=cv2.GaussianBlur(warp1.copy(),(5,5),0)
    smooth=cv2.addWeighted(blur,1.5,warp1,-0.5,0)
    #bin=binary(warp1)
    return smooth

def inverseperceptive(image):
    dst1 = np.array([
        [0,720],
        [0, 0],
        [1280,0],
        [1280,720]], dtype = "float32")

    source=np.array([
        [0+150,720-80],
        [0+570, 450],
        [1280-570,450],
        [1280-50,720-80]], dtype = "float32")



    M1,status = cv2.findHomography(dst1,source,)
    warp1 = cv2.warpPerspective(image.copy(), M1, (1280,720))
    warp2=cv2.medianBlur(warp1,3)

    return warp2

def homography_calc(src,dest):
    c1 = tag_des[0]
    c2 = tag_des[1]
    c3 = tag_des[2]
    c4 = tag_des[3]

    w1 = src[0]
    w2 = src[1]
    w3 = src[2]
    w4 = src[3]

    A=np.array([[w1[0],w1[1],1,0,0,0,-c1[0]*w1[0],-c1[0]*w1[1],-c1[0]],
                [0,0,0,w1[0], w1[1],1,-c1[1]*w1[0],-c1[1]*w1[1],-c1[1]],
                [w2[0],w2[1],1,0,0,0,-c2[0]*w2[0],-c2[0]*w2[1],-c2[0]],
                [0,0,0,w2[0], w2[1],1,-c2[1]*w2[0],-c2[1]*w2[1],-c2[1]],
                [w3[0],w3[1],1,0,0,0,-c3[0]*w3[0],-c3[0]*w3[1],-c3[0]],
                [0,0,0,w3[0], w3[1],1,-c3[1]*w3[0],-c3[1]*w3[1],-c3[1]],
                [w4[0],w4[1],1,0,0,0,-c4[0]*w4[0],-c4[0]*w4[1],-c4[0]],
                [0,0,0,w4[0], w4[1],1,-c4[1]*w4[0],-c4[1]*w4[1],-c4[1]]])

    #Performing SVD
    u, s, vt = la.svd(A)

            # normalizing by last element of v
            #v =np.transpose(v_col)
    v = vt[8:,]/vt[8][8]

    req_v = np.reshape(v,(3,3))

    return req_v

def Undistort(image):
    dist = np.mat([ -2.42565104e-01 , -4.77893070e-02 , -1.31388084e-03 , -8.79107779e-05,
        2.20573263e-02])

    K = np.mat([[  1.15422732e+03 , 0.00000000e+00  , 6.71627794e+02],
     [  0.00000000e+00 ,  1.14818221e+03  , 3.86046312e+02],
     [  0.00000000e+00 ,  0.00000000e+00  , 1.00000000e+00]])

    destination = cv2.undistort(image, K, dist, None, K)

    return destination

def ExtractROI(image,length,width):
    black=np.zeros((360,1280))

    # Crop image
    imageROI = image[length:720,0:1280]
    black=np.concatenate((black, imageROI), axis=0)
    return black

def Yellowlane(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower = np.array([20, 100, 100])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    transform_y=perspective_view(mask)
    blur_y=cv2.GaussianBlur(transform_y.copy(),(5,5),0)
    return blur_y

def Whitelane(image):
    #gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(image,(kernel_size, kernel_size),0)
    mask = cv2.inRange(blur_gray, 180, 255)
    #edges = cv2.Canny(gray,100,200)
    return mask

def Hough_lines(image,Original):
    width,height=image.shape

    edges = cv2.Canny(image, 100, 200)
    rho = np.ceil(np.sqrt(width*width+height*height))
    theta = np.pi/60
    threshold = 6
    min_line_length = 30
    max_line_gap = 25
    line_image = np.copy(Original) * 0
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    #lines_edges = cv2.addWeighted(lines, 0.8, line_image, 1, 0)`````
    return line_image

def Histogram(image):
    leftlane=[]
    rightlane=[]
    for j in range(10):
        c=72*(10-j)
        d=72*(9-j)
        for i in range(20):
            a=64*(i)
            b=64*(i+1)
            img=image[d:c,a:b]
            hist = cv2.calcHist([img],[0],None,[3],[0.5,1.5])
            if(hist[1]>1 and i<10):
                leftlane=np.append(leftlane,[(c+d)/2,(a+b)/2])
            if(hist[1]>4 and 20-i<10):
                rightlane=np.append(rightlane,[(c+d)/2,(a+b)/2])
    rangeleft=int(leftlane.shape[0]/2)
    left=np.reshape(leftlane,[rangeleft,2])
    try:
        if(rightlane==[]):
            raise ValueError

    except ValueError:
        rightlane=np.append(rightlane,leftlane[0])
        rightlane=np.append(rightlane,leftlane[1]+800)
    rangeright=int(rightlane.shape[0]/2)
    right=np.reshape(rightlane,[rangeright,2])

    return image,left,right

def plotlines(image,Original,left,right,polyfit_old,count):
    left_fit = np.polyfit(left[:,0], left[:,1], 2)
    right_fit = np.polyfit(right[:,0], right[:,1], 2)

    buffer_m=1000
    buffer_c=1000
    if (count==0):
        polyfit_old=[left_fit[1],left_fit[2]]

    if ((left_fit[1]-polyfit_old[0])>buffer_m):
        left_fit[1]=polyfit_old[0]+buffer_m
        print(1)
    elif ((left_fit[1]-polyfit_old[0])<-buffer_m):
        left_fit[1]=polyfit_old[0]-buffer_m
        print(2)
    else:
        left_fit[1]=left_fit[1]
        print("kuch nahi")

    if ((left_fit[2]-polyfit_old[1])>buffer_c):
        left_fit[2]=polyfit_old[1]+buffer_c
        print(1)
    elif ((left_fit[2]-polyfit_old[1])<-buffer_c):
        left_fit[2]=polyfit_old[1]-buffer_c
        print(2)
    else:
        left_fit[2]=left_fit[2]
        print("kuch nahi")


    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = left_fit[0]*ploty*ploty + left_fit[1]*ploty + left_fit[2]
    right_fitx =right_fit[0]*ploty*ploty + right_fit[1]*ploty + right_fit[2]
    center_line=(left_fit[0]*ploty*ploty+right_fit[0]*ploty*ploty)/2 + (left_fit[1]*ploty+right_fit[1]*ploty)/2 + (left_fit[2]+right_fit[2])/2
    black=Original*0
    for i in range(ploty.shape[0]):
        y=int(ploty[i])
        x=int(left_fitx[i])
        x_r=int(right_fitx[i])
        x_c=int(center_line[i])
        koko=(x_r-x)
        for i in range(koko):
            if(x+i>1280-1 or x+i<0 ):
                break
            black[y,x+i]=[0,0,51]

    for j in range(10):
        if(x_c+j-15>1280-1 or x_c+j-15<0 ):
            break
        indexf=70*(j+1)-10
        indexi=70*(j)+10
        pointf=(int(center_line[indexf]),int(ploty[indexf]))
        pointi=(int(center_line[indexi]),int(ploty[indexi]))
        cv2.arrowedLine(black, pointf, pointi, (0,153,76), 15,4,0,0.5)
    slope=(math.atan((left_fitx[720-50]-left_fitx[50])/620))*180/np.pi
    print(slope)
    polyfit_old=left_fit
    return black,slope,polyfit_old

def Text(image,slope):
    slp=round(slope,2)
    info="Angle of turning is: "+str(np.abs(slp))+' degrees'
    cv2.putText(image,info,(350,100),cv2.FONT_HERSHEY_SIMPLEX,1,(51,0,0),3,cv2.LINE_AA)
    if(slp>0):
        cv2.putText(image,'Turn LEFT',(540,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,51,102),3,cv2.LINE_AA)
    elif(slp<0):
        cv2.putText(image,'Turn RIGHT',(540,200),cv2.FONT_HERSHEY_SIMPLEX,1,(51,102,0),3,cv2.LINE_AA)
    else:
        cv2.putText(image,'Go STRAIGHT',(540,200),cv2.FONT_HERSHEY_SIMPLEX,1,(51,0,102),3,cv2.LINE_AA)


    return image
#-------------------------------------------------------------------------------
def Imageprocessor(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    img_array=[]
    while (success):
        if (count==0):
            success, image = vidObj.read()
        width,height,layers=image.shape
        size = (width,height)
        image=Undistort(image)
        undist_img=image.copy()
        gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transform_w=perspective_view(gray)
        image_w=Whitelane(transform_w)
        image_y=Yellowlane(undist_img)
        final_transform=image_w+image_y
        final_transform_original=final_transform.copy()
        bin=binary(final_transform)
        Histo,leftlane,rightlane=Histogram(bin)
        if (count==0):
            back=[0,0]
        tr_with_lines,slope,back_n=plotlines(final_transform_original,undist_img,leftlane,rightlane,back,count)
        back=back_n

        global_lines=inverseperceptive(tr_with_lines)
        gray_merge=cv2.bitwise_or(undist_img,global_lines)
        result=cv2.addWeighted(undist_img,1,global_lines,1,0)
        pakka=Text(result,slope)

        count += 1
        print(count)
        cv2.imwrite('%d.jpg' %count,pakka)
        #img_array.append(pakka)
        success, image = vidObj.read()

    return img_array,size

#video file
def video(img_array,size):
    video=cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16.0,size)
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()
# main
if __name__ == '__main__':

    # Calling the function
    Image,size=Imageprocessor('project_video.mp4')
    video(Image,size)
