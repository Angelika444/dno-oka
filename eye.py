import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology
import cv2
import numpy as np
from skimage import feature
from skimage.morphology import square

image = cv2.imread("eye7.jpg")
#cv2.imshow("obraz wejsciowy",img)
img_height=image.shape[0]
img_width=image.shape[1]
no_eye=[]


def mask(img):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_red=np.array([0,180,150])
    upper_red=np.array([10,255,255])
    mask1=cv2.inRange(imgHSV, lower_red,upper_red) #wybiera piksele, które są czerwone
    
    lower_red2=np.array([160,180,150])
    upper_red2=np.array([179,255,255])
    mask2=cv2.inRange(imgHSV, lower_red2,upper_red2)#wybiera piksele, które są czerwone
    
    mask=cv2.add(mask1,mask2)
    imgRed = cv2.bitwise_and(img, img, mask = mask)
    
    #cv2.imshow('onlyred',imgRed)
    #cv2.waitKey(0)
    
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
    '''print('h',meanH)
    print('s',meanS)
    print('v', meanV)'''
    
    if meanV<150:
        alfa=150/meanV
        for w in range(img_width):
            for h in range(img_height):
                imgHSV[h,w][2]=min(int(imgHSV[h,w][2]*alfa),255)
                
    if meanS<200:
        alfa=200/meanS
        for w in range(img_width):
            for h in range(img_height):
                imgHSV[h,w][1]=min(int(imgHSV[h,w][1]*alfa),255)
    
    img2=cv2.cvtColor(imgHSV,cv2.COLOR_HSV2BGR)
    #cv2.imshow('jasniejszy', img2)
    #cv2.waitKey(0)
    return img2

#zaczelam robi, ale na razie nie wychodzi: rozmycie tła, zeby nie bylo ostrych krawedzi, ale trzeba z tym jeszcze zdecydowanie cos zrobic
def blur_background(img):
    sumR=0
    sumG=0
    sumB=0
    summ=0
    for w in range(img_width):
      for h in range(img_height):
          #sumR+=img[h,w][2]
          #sumG+=img[h,w][1]
          #sumB+=img[h,w][0]
          summ+=img[h,w]
    
    size=img_height*img_width
    meanR=sumR/size
    meanG=sumG/size
    meanB=sumB/size
    meann=summ/size
    
    '''print('r', meanR)
    print('g', meanG)
    print('b', meanB)'''
    
    for w in range(img_width):
        for h in range(img_height):  
            if img[h,w]<0.15:
                img[h,w]=meann
            #if img[h,w][2]<100 and img[h,w][1]<100 and img[h,w][0]<100:
                #img[h,w]=[int(meanB*4),int(meanG),int(meanR*0.7)]
                #img[h,w]=[0,0,0]
    return img
    
def in_circle(circles,w,h,less=0):
    in_c=False
    if circles is not None:
        if (circles[0][0][0]-w)**2+(circles[0][0][1]-h)**2<(circles[0][0][2]-less)**2:
            in_c=True
    return in_c
def color_filter(img,circles):
              
    for w in range(img_width):
        for h in range(img_height):            
            if(in_circle(circles,w,h)):
                #zwieksz lekko kolor czerwony
                img[h,w][2]=min(255,int(img[h,w][2]*1.05))
                #zwieksz barwe zielona, proporcjonalnie do jej nasycenia
                img[h,w][1]=min(int(img[h,w][1]*1.5),255)
                            
                 
    #img=blur_background(img)
    #cv2.imshow('green_filter',img)
    #cv2.waitKey(0)
    return img

def detect_circle(img):
     img2=img.copy()
     imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     min_length=min(img.shape[0],img.shape[1])
     max_length=max(img.shape[0],img.shape[1])
     minradius=int(0.5*min_length)
     #alg. HoughCircles znajduje koło na zdjęciu, na wejciu img w skali szarosci, circles-np.array z inf. o pikselach
     circles = cv2.HoughCircles(imgGray,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=minradius,maxRadius=int(max_length*0.5))

     if circles is not None:
         circles = np.uint16(np.around(circles))#zaokragla war. do war. calkowitych
         #cv2.imshow('detected circles',img)
         x=int(circles[0][0][0])
         y=int(circles[0][0][1])
         r=int(circles[0][0][2])
         if x!=0 and y!=0 and r!=0:
            
            for i in circles[0,:]:
                cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),1) # rysuje koło
            #cv2.imshow('circle',img2)
            #cv2.waitKey(0)
     return circles
                
def sharpening(img):
    sh=-0.5
    kernel_sharpening = np.array([[sh,sh,sh], 
                                  [sh, abs(sh*8)+1,sh],
                                  [sh,sh,sh]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    #cv2.imshow('sh',sharpened)
    #cv2.waitKey(0)
    return sharpened

def compare(img,template,circles,kk,image2):
    
    tp=0
    tn=0
    fp=0
    fn=0
    for w in range(img_width):
        for h in range(img_height):            
            if(in_circle(circles,w,h,2)):
                if h>limit_down and h<limit_up:
                    if img[h][w]:
                        if template[h][w]>0.5:
                            tp+=1
                            image2[h,w]=[0,0,255]
                        else:
                            fp+=1
                            image2[h,w]=[0,255,0]
                    else:
                        if template[h][w]>0.5:
                            fn+=1
                            image2[h,w]=[255,0,0]
    
                        else:
                            tn+=1
    print(tp, fp, fn, tn, (tp+tn)/(tp+tn+fp+fn))
    if kk>0:
        ax=fig.add_subplot(3, 2, kk)
        ax.axis('off')
        ax.imshow(image2)
    #else:
        #cv2.imshow('comparing',image2)
        #cv2.waitKey(0)

#cv2.imshow('basic',img)
#cv2.waitKey(0)
#rozjasniamy obraz i nasycamy barwę
image=brightness(image)
#wyostrza krawedzie, ale wtedy znajduje az za duzo krawedzi
#sharpened=sharpening(img)
#wykrywamy zarys oka
circles=detect_circle(image)

#cv2.imwrite( "Image.jpg", img );
#głownie zwiększa barwę zieloną na obrazie, zwiekszajac kontrast pomiedzy czerwonym a pomaranczowym
image=color_filter(image,circles)

#sharpened = sharpening(img)
#img2=mask(sharpened)

#imgGray=cv2.cvtColor(sharpened,cv2.COLOR_BGR2GRAY)
#zapisuje zdjecie i je czytam przez io.imread, bo jakies problemy z typem danych mialam
cv2.imwrite( "Image2.jpg", image );


img_template=cv2.imread("eye7_1.jpg")
img_template = color.rgb2gray(img_template)
kernel = np.ones((2,2),np.uint8)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=blur_background(gray)

high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
high_thresh=max(high_thresh,130)
lowThresh = 0.5*high_thresh
#print(lowThresh,high_thresh)
edges = cv2.Canny(image,lowThresh,high_thresh)
#cv2.imshow('edges',edges)
#cv2.waitKey(0)

closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,kernel)
#cv2.imshow('closing',closing)
#cv2.waitKey(0)

#kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(closing,kernel,iterations = 1)
#cv2.imshow('erosion',erosion)
'''
print('tp fp tn fn:')
print('opencv close')
compare(closing,img_template,circles,0,image.copy())
print('opencv close+erode')
compare(erosion,img_template,circles,0,image.copy())
#cv2.waitKey(0)'''


def detect_background_line(img,image2):
    max_count1=0
    max_count2=0
    row1=0
    row2=0
    for h in range(img_height):
        count=0
        for w in range(img_width):
            if img[h,w]:
                count+=1
        if count>max_count1:
            max_count1=count
            row1=h
        elif count>max_count2:
            max_count2=count
            row2=h
    image2[row1,:]=[255,0,0]
    image2[row2,:]=[0,255,0]
    ax=fig.add_subplot(3, 2, 4)
    ax.axis('off')
    ax.imshow(image2)
    for w in range(img_width):
        no_eye.append((row1,w))
        no_eye.append((row2,w))
    limit_up=max(row1,row2)
    limit_down=min(row1,row2)
    return limit_down, limit_up
            
'''cv2.imwrite('edges-50-150.jpg',edges)
minLineLength=100
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

a,b,c = lines.shape
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite('houghlines5.jpg',gray)'''

'''
img = color.rgb2gray(img)
#im mniejsza sigma, tym wiecej krawedzi wykrywa
canny = feature.canny(img, sigma=0.7)
cv2.imshow('canny',canny)
cv2.waitKey(0)'''

'''ret,thresh = cv2.threshold(canny,127,255,cv2.THRESH_BINARY)
cv2.imshow('tresh',thresh)
cv2.waitKey(0)'''

#img=io.imread("Image.jpg")
#img = color.rgb2gray(img)

plt.gray()
fig = plt.figure(figsize=(20,12))
image=io.imread("Image2.jpg")
img2 = color.rgb2gray(image)
img2=blur_background(img2)

#im mniejsza sigma, tym wiecej krawedzi wykrywa
#canny = feature.canny(img, sigma=0.7)
canny2 = feature.canny(img2, sigma=0.7)
limit_down,limit_up=detect_background_line(canny2,image.copy())
mor=morphology.binary_closing(canny2)
mor2=morphology.binary_erosion(mor,square(2))
#canny2=morphology.binary_erosion(canny2)



ax=fig.add_subplot(3, 2, 1)
ax.axis('off')
ax.imshow(canny2)
ax=fig.add_subplot(3, 2, 2)
ax.axis('off')
ax.imshow(mor)
ax=fig.add_subplot(3, 2, 3)
ax.axis('off')
ax.imshow(mor2)

print('skimage close')
compare(mor,img_template,circles,5,image.copy())
print('skimage close+erode')
compare(mor2,img_template,circles,6,image.copy())

plt.show()
plt.close()
#cv2.imshow('basica',img_template)
#cv2.waitKey(0)

cv2.destroyAllWindows()