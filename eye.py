import matplotlib.pyplot as plt
from skimage import io, color
import cv2
import numpy as np
import skimage
from skimage import feature

img = cv2.imread("eye2.jpg")
#cv2.imshow("obraz wejsciowy",img)
img_height=img.shape[0]
img_width=img.shape[1]

#rozjasnia obraz
def brightness(img):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sumH=0
    sumV=0
    sumS=0
    for w in range(img_width):
        for h in range(img_height):
            sumV+=imgHSV[h,w][2]
            sumS+=imgHSV[h,w][1]
            sumH+=imgHSV[h,w][0]
    
    meanH=sumH/img_width/img_height
    meanS=sumS/img_width/img_height
    meanV=sumV/img_width/img_height
    print('h',meanH)
    print('s',meanS)
    print('v', meanV)
    
    if meanV<150:
        alfa=150/meanV
        for w in range(img_width):
            for h in range(img_height):
                imgHSV[h,w][2]=min(imgHSV[h,w][2]*alfa,255)
                
    if meanS<200:
        alfa=200/meanS
        for w in range(img_width):
            for h in range(img_height):
                imgHSV[h,w][1]=min(imgHSV[h,w][1]*alfa,255)
    
    img2=cv2.cvtColor(imgHSV,cv2.COLOR_HSV2BGR)
    cv2.imshow('jasniejszy', img2)
    cv2.waitKey(0)
    return img2

#zaczelam robi, ale na razie nie wychodzi: rozmycie tła, zeby nie bylo ostrych krawedzi, ale trzeba z tym jeszcze zdecydowanie cos zrobic
def blur_background(img):
    sumR=0
    sumG=0
    sumB=0
    for w in range(img_width):
      for h in range(img_height):
          sumR+=img[h,w][2]
          sumG+=img[h,w][1]
          sumB+=img[h,w][0]
    
    size=img_height*img_width
    meanR=sumR/size
    meanG=sumG/size
    meanB=sumB/size
    
    print('r', meanR)
    print('g', meanG)
    print('b', meanB)
    
    for w in range(img_width):
        for h in range(img_height):      
            if img[h,w][2]<100 and img[h,w][1]<100 and img[h,w][0]<100:
                img[h,w]=[meanB*4,meanG,meanR*0.7]
    return img
    
#bardziej nasyca kolory            
def color_filter(img):
    sumR=0
    sumG=0
    sumB=0
    for w in range(img_width):
      for h in range(img_height):
          sumR+=img[h,w][2]
          sumG+=img[h,w][1]
          sumB+=img[h,w][0]
    
    size=img_height*img_width
    meanR=sumR/size
    meanG=sumG/size
    meanB=sumB/size
    print('r', meanR)
    print('g', meanG)
    print('b', meanB)
    
    for w in range(img_width):
        for h in range(img_height):            
            
        #zwieksz kolor czerwony czerwonym pikselom
            if img[h,w][2]>220:
                img[h,w][2]=255
            else:
                if img[h,w][2]>meanR:
                    img[h,w][2]=img[h,w][2]*25/22
      
        #zwieksz zielony-glownie pomaranczowym
            if img[h,w][1]>abs(128-meanG):
                if img[h,w][1]<abs(128-meanG)*2:
                    img[h,w][1]*=1.25
                else:
                    if img[h,w][1]<160:
                        img[h,w][1]*=1.5
                    else:
                        img[h,w][1]=255
                        
        #zwieksz niebieski
            if img[h,w][0]>meanB:
                if img[h,w][0]<meanB*2:
                    img[h,w][0]*=1.5
                else:
                    if img[h,w][0]<127:
                        img[h,w][0]*=2
                    else:
                        img[h,w][0]=255
    
                 
    #img=blur_background(img)
    cv2.imshow('red',img)
    cv2.waitKey(0)
    return img

kernel_sharpening = np.array([[-0.5,-0.5,-0.5], 
                              [-0.5, 5,-0.5],
                              [-0.5,-0.5,-0.5]])
    
img=brightness(img)
#wyostrza krawedzie, ale wtedy znajduje az za duzo krawedzi
sharpened = cv2.filter2D(img, -1, kernel_sharpening)
cv2.imshow('sh',sharpened)
cv2.waitKey(0)

img=color_filter(img)

sharpened = cv2.filter2D(img, -1, kernel_sharpening)
cv2.imshow('s',sharpened)
cv2.waitKey(0)

#imgGray=cv2.cvtColor(sharpened,cv2.COLOR_BGR2GRAY)
#zapisuje zdjecie i je czytam przez io.imread, bo jakies problemy z typem danych mialam
cv2.imwrite( "Image.jpg", img );

img=io.imread("Image.jpg")
img = color.rgb2gray(img)
#im mniejsza sigma, tym wiecej krawedzi wykrywa
canny = feature.canny(img, sigma=0.9)

plt.gray()
fig = plt.figure(figsize=(20,12))
ax=fig.add_subplot(2, 2, 1)
ax.axis('off')
ax.imshow(canny)
plt.show()
plt.close()


cv2.destroyAllWindows()