# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:20:01 2019

@author: AKASH
"""

import cv2
import numpy as np
import imutils
from numpy.linalg import inv
from numpy.linalg import norm

lena = cv2.imread('Lena.png')
lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)


def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # print(np.argmax(diff))
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

def homo(p1, p2):  # To generate the Homography
    a = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        a.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        a.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    a = np.asarray(a)
    u, s, vh = np.linalg.svd(a)
    l = vh[-1, :] / vh[-1, -1]
    h = np.reshape(l, (3, 3))
    return h

def calc_mat(h):
    K=np.array([[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]]).T
    b_new = inv(K)*h
    b1 = b_new[:,0].reshape(3,1)
    b2 = b_new[:,1].reshape(3,1)
    r3 = np.cross(b_new[0,:],b_new[1,:])
    b3 = b_new[:,2].reshape(3,1)
    lambda_val = 2/(norm((inv(K)).dot(b1))+norm((inv(K)).dot(b2)))
    r1 = lambda_val*b1
    r2 = lambda_val*b2
    r3 = lambda_val*lambda_val*r3.reshape(3,1)
    t = lambda_val*b3
    r = np.concatenate((r1,r2,r3),axis = 1)
    return r,t,K

def homogenous_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    max_dim = 160

    # camera coordiante points
    dst = np.array([
        [0, 0],
        [max_dim - 1, 0],
        [max_dim - 1, max_dim - 1],
        [0, max_dim - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    m1 = homo(rect, dst)
    warped_img = cv2.warpPerspective(image, m1, (max_dim, max_dim))
    r,t,K = calc_mat(m1)
    m2 = homo(dst, rect)  # For Lena on Tag

    return warped_img,r,t,K,rect

# Take the video
cap = cv2.VideoCapture('Tag2.mp4')


while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # smooth
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[2:5]
    axis = np.float32([[0,0,0], [0,159,0], [159,159,0], [159,0,0],[0,0,-159],[0,159,-159],[159,159,-159],[159,0,-159] ])
# loop over the contours
    for c in cnts:
        # approximate the contour
        #area = cv2.contourArea(c)
        #print(area)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            corners = approx
            break
        elif len(approx) != 4:
            corners = None

    tag_count = 0
    #print(corners[:, 0].shape)
    if corners is not None:
        tag_count =  tag_count + 1
        cv2.drawContours(frame, [corners], -1, (0, 255, 0), 1)
        cv2.imshow("Outline", frame)
        
        warped,r,t,K,rect = homogenous_transform(frame, corners[:, 0])
        cv2.imshow("Warped", warped)
        
        imgpts,jac = cv2.projectPoints(axis,r,t,K,np.zeros((1,8)))
        img = draw(frame,imgpts)

        cv2.imshow('Points',img)

    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
