''' 

    محمد النبوي سليمان غازي
Mohammed El-Nabawy soliman ghaze

'''
# add reqired libraries
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image, ImageEnhance 
from skimage.morphology import skeletonize
from tkinter import filedialog
import cv2 as cv
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy import ndimage
from skimage.util import random_noise
from skimage import img_as_float
from skimage import io, color, morphology
import matplotlib.pyplot as plt



root = Tk()
root.title('image processing') #the window title
root.geometry('1060x800') #the window size


#resize the image function and call the image class
def resize(img):
    global image #global to can be used by save function
    image = img 
    img = img.resize((250, 230)) 
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    return(img)


#display the input image function in the panel label
def display(img):
    img = resize(img) #call resize function to resize the input image before display
    panel.config(image = img) 
    # set the image as img
    panel.image=img
    panel.place(x=0, y=0)


#display the image function after adding noise  in the noise label        
def display_noise(img):
    img = Image.fromarray(img) #convert the img to image formate to can be display
    img = resize(img) #call resize function to resize the noisy image before display
    noise_label.config(image = img)
    # set the image as img
    noise_label.image=img
    noise_label.place(x=0, y=0)


#display the image function after perform image processing operation in the resule label 
def display_result(img):
    img = resize(img) #call resize function to resize the result image before display
    result.config(image = img)
    # set the image as img
    result.image=img
    result.place(x=0, y=0)


#convert image to RGB
def  RGB():
    # img_open the input image
    img = img_open.convert('RGB') #open the image from the image path and then convert it to RGB
    display(img) #call display function


#convert image to Gray
def grayimg():
    # img_open the input image
    img = img_open.convert('LA') #open the image from the image path and then convert it to Gray
    display(img)


#ADD solt and pepper noise to the input image
def salt_pepper():
    # Add salt-and-pepper noise to the image.
    # img_read the input image
    noise_img = random_noise(img_read, mode='s&p',amount=.3) #amount refer to amount of noise in the image and it between [0 : 1]
    img2 = np.array(255*noise_img, dtype = 'uint8')    
    display_noise(img2) 

    
#ADD gaussian noise to the input image
def Gaussiannoise():
    # Add gaussian noise to the image.
    # img_read the input image
    noise_img = random_noise(img_read, mode='gaussian') #have no amount attribute
    img2 = np.array(255*noise_img, dtype = 'uint8')    
    display_noise(img2)


#ADD poisson noise to the input image
def Poissonnoise():
    # Add poisson noise to the image.
    # img_read the input image
    noise_img = random_noise(img_read, mode='poisson') #have no amount attribute like gaussian
    img2 = np.array(255*noise_img, dtype = 'uint8') 
    display_noise(img2)


# insrt image the device
def insert_image():
    #global to be used by other function
    global img_read, img_open, img_read_gray, img_path 
    #the image path
    img_path=filedialog.askopenfilename(initialdir='/', title='Choose Image',  filetypes=(('image files', '.png'), ('image files', '.jpg'), ('image files', '.jpeg')))
   
    img_read = cv.imread(img_path) # read the image from the path
    img_read_gray = cv.imread(img_path, 0)
    #rearrange the image color to display
    b,g,r = cv.split(img_read) 
    img_read = cv.merge((r,g,b))
    
    img_open = Image.fromarray(img_read) #convert the img_read to image formate to can be display
    display(img_open)



#the rule for display the image (RGB or Gray image)
def convert():
    value=var.get()
    if value == 1:
        RGB()
        
    elif value == 2:
        grayimg()
        
          
#the rule for add noise to the image (salt and pepper or Gaussian noise or Poisson noise)
def noise():
    value=var1.get()
    if value ==1: 
        salt_pepper()
        
    elif value == 2:
        Gaussiannoise()
        
    elif value == 3:
        Poissonnoise()

    
# increase or decrease the brightness of the image
def Brightness_adjustment():
    # Where the img_open is the input image 
    enhancer = ImageEnhance.Brightness(img_open) # bright the image
    enhanced_im = enhancer.enhance(1.8)
    
    display_result(enhanced_im)


# increase or decrease the contrast of the image
def Contrast_adjustmen():
    # Where img_open is the input image 
    enhancer = ImageEnhance.Contrast(img_open) #contrast the image
    enhanced_im = enhancer.enhance(1.8)
    
    display_result(enhanced_im)


# creat a histogram graph for pixels values of the input image
def Histogram():
    
    #Where img_read_gray is gray input image
    gray_hist = cv.calcHist([img_read_gray], [0], None, [256], [0,256])
	# the figure that will contain the plot
    fig = Figure(figsize = (3.5, 3))

	# adding the subplot
    plot1 = fig.add_subplot()

	# plotting the graph
    plot1.plot(gray_hist)

	# creating the Tkinter canvas
	# containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master = labelframe_20)
	# placing the canvas on the Tkinter window
    canvas.get_tk_widget().place(x=10, y=0)


# reassign the pixels values to creat Histogram graph for pixels values after equalization
def Histogram_Equalization():
    #Where img_read_gray is gray input image
    equ = cv.equalizeHist(img_read_gray)
    
    gray_hist = cv.calcHist([equ], [0], None, [256], [0,256])
	# the figure that will contain the plot
    fig = Figure(figsize = (3.5, 3))

	# adding the subplot
    plot1 = fig.add_subplot()

	# plotting the graph
    plot1.plot(gray_hist)

	# creating the Tkinter canvas
	# containing the Matplotlib figure
    global canvas
    canvas = FigureCanvasTkAgg(fig, master = labelframe_20)
    canvas.draw()

	# placing the canvas on the Tkinter window
    canvas.get_tk_widget().place(x=0, y=0)


#low pass filter creat a bluring image
def Low_pass_filte():
    #Where img_read is input image
    blur = cv.GaussianBlur(img_read,(5,5),0) # GaussianBlure is type of low path filter
    
    blur = Image.fromarray(blur)
    
    display_result(blur)


#high pass filter creat an edges for input image  
def High_pass_filter():
    #Where img_read is input image
    # (5x5) kernal which used by the input image 
    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 2, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, -1, -1, -1, -1]])
   
    k5  = ndimage.convolve(img_read_gray, kernel_5x5) #apply the kernal to the image
    blurred = cv.GaussianBlur(img_read_gray, (11, 11), 0) #blure yhe input image
    g_hpf = img_read_gray - blurred
    g_hpf = Image.fromarray(g_hpf)
    
    display_result(g_hpf) #desplay the result image


#appling median filter on the image
def Median_filter():
    #Where img_read is input image    
    median = cv.medianBlur(img_read,5) #apply median filter
    median = Image.fromarray(median)
    
    display_result(median) #display the result
    

#appling median filter on the image
def Avereging_filter():
    #Where img_read is input image        
    blur = cv.blur(img_read,(5,5)) #aopply avereging filter(blur)
    blur = Image.fromarray(blur)

    display_result(blur) #display the result


# rule to choose the type of filter 
def Edge_detection_filter():
    value=var2.get()
    
    if value == 1:
        Laplacian_filter()        
    elif value == 2:
        
        Vert_Prewitt()
    elif value == 3:
        Zero_cross()
        
    elif value == 4:
        Gaussian_filter()
        
    elif value == 5:
        Horiz_Prewitt()
        
    elif value == 6:
        Thicken()
        
    elif value == 7:
        vert_sobel()
        
    elif value == 8:
        Lap_Of_Gau_log()
        
    elif value == 9:
        Skelton()
        
    elif value == 10:
        Horiz_sobel()
        
    elif value == 11:
        Canny_method()
        
    elif value == 12:
        thinning()


#appling laplacian filter on the image
def Laplacian_filter():
    #Where img_read_gray is input image 
    lap = cv.Laplacian(img_read_gray, cv.CV_64F) #applay laplaciann filter
    lap = np.uint8(np.absolute(lap))
    
    lap = Image.fromarray(lap)
    display_result(lap) #display the result image


#appling vertical prewiit filter on the image
def Vert_Prewitt():
    #Where img_read_gray is input image
    img_gaussian = cv.GaussianBlur(img_read_gray,(3,3),0)
    
    kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) # vertical prewiit kernal
    img_prewittx = cv.filter2D(img_gaussian, -1, kernelx) #applay the kernal on the image
    
    img_prewittx = Image.fromarray(img_prewittx)
    display_result(img_prewittx) #display the result image


#appling zero cross filter on the image    
def Zero_cross():
    #Where img_read_gray is input image
    # Apply Gaussian Blur
    blur = cv.GaussianBlur(img_read_gray,(3,3),0) #blur the image first     
    laplacian = cv.Laplacian(blur,cv.CV_64F) #and then  apply laplacian filter
    laplacian1 = laplacian/laplacian.max() #devide by max
    
    laplacian1 = Image.fromarray(laplacian1)
    display_result(laplacian1) #display the result image


#appling gaussian filter on the image
def Gaussian_filter():
    #Where img_read_gray is input image
    blur = cv.GaussianBlur(img_read,(5,5),0) # Gaussian kernal
    blur = Image.fromarray(blur)
    display_result(blur) #display the result image


#appling horizontal prewiit filter on the image
def Horiz_Prewitt():
    #Where img_read_gray is input image
    img_gaussian = cv.GaussianBlur(img_read_gray,(3,3),0)  
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) # horizontal prewiit kernal
    img_prewitty = cv.filter2D(img_gaussian, -1, kernely) #applay the kernal on the image
    
    img_prewitty = Image.fromarray(img_prewitty)
    display_result(img_prewitty) #display the result image


#appling filter to thicken the image borders
def Thicken():
    #Where img_path is path of the image in the device 
    image = img_as_float(color.rgb2gray(io.imread(img_path)))
    image_binary = image < 0.5
    out_skeletonize = morphology.skeletonize(image_binary)
    #out_thin = morphology.thin(image_binary)
            
    out_skeletonize = Image.fromarray(out_skeletonize)
    display_result(out_skeletonize) #display the result image


#appling vertical sobel filter on the image
def vert_sobel():
    #Where img_read_gray is input image
    sobelx = cv.Sobel(img_read_gray, cv.CV_64F, 1, 0)
    
    sobelx = Image.fromarray(sobelx)
    display_result(sobelx) #display the result image


#appling laplace of gaussian filter on the image
def Lap_Of_Gau_log(sigma=1., kappa=0.75, pad=False):
    #Where img_read_gray is input image
    assert len(img_read_gray.shape) == 2
    img = cv.GaussianBlur(img_read_gray, (0, 0), sigma) if 0. < sigma else img_read_gray
    img = cv.Laplacian(img, cv.CV_64F)
    rows, cols = img.shape[:2]
    min_map = np.minimum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    pos_img = 0 < img[1:rows-1, 1:cols-1]
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    zero_cross = neg_min + pos_max
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.
    log_img = values.astype(np.uint8)
    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)

    log_img = Image.fromarray(log_img)
    display_result(log_img) #display the result image


#appling filter to skelton the image
def Skelton():
    #Where img_read_gray is input image
    ret, bw_img = cv.threshold(img_read,127,255,cv.THRESH_BINARY)

    skeleton = skeletonize(bw_img)
    
    skeleton = Image.fromarray(skeleton)
    display_result(skeleton) #display the result image


#appling horizontal sobel filter on the image
def Horiz_sobel():
    #Where img_read_gray is input image
    sobely = cv.Sobel(img_read_gray, cv.CV_64F, 0, 1)
    
    sobely = Image.fromarray(sobely)
    display_result(sobely) #display the result image


#appling cannyt filter on the image
def Canny_method():
    #Where img_read_gray is input image
    canny = cv.Canny(img_read_gray, 150, 175)
    
    canny = Image.fromarray(canny)
    display_result(canny) #display the result image


#appling filter to thinning the image
def thinning():
    (thresh, im_bw) = cv.threshold(img_read_gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    thresh = 127
    im_bw = cv.threshold(img_read_gray, thresh, 255, cv.THRESH_BINARY)[1]
    #out_skeletonize = morphology.skeletonize(image_binary)
    im_bw = morphology.thin(im_bw)
            
    im_bw = Image.fromarray(im_bw)
    display_result(im_bw) #display the result image


#detect lines on the image
def Line_Detection():
    #Where img_read is input image
    #Where img_read_gray is gray input image 
    img=img_read
    edges = cv.Canny(img_read_gray, 75, 150) # apllay canny filter first
    # Apply Hough transform on the canny result image.
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    for line in lines:
       x1, y1, x2, y2 = line[0]
       cv.line(img, (x1, y1), (x2, y2), (0, 0, 128), 1)
   
    img = Image.fromarray(img)
    display_result(img) #display the result image



def Circles_Detection():
    #Where img_path is the path of the input image
    #Where img_read_gray is gray input image 
    # Blur using 3 * 3 kernel.
    img = cv.imread(img_path)
      
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      
    gray_blurred = cv.blur(gray, (3, 3))
      
    # Apply Hough transform on the blurred image.
    detected_circles = cv.HoughCircles(gray_blurred, 
                       cv.HOUGH_GRADIENT, 1, 20, param1 = 50,
                   param2 = 30, minRadius = 1, maxRadius = 40)
      
    # Draw circles that are detected.
    if detected_circles is not None:
      
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
      
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
      
            # Draw the circumference of the circle.
            cv.circle(img, (a, b), r, (0, 255, 0), 2)
      
            # Draw a small circle (of radius 1) to show the center.
            cv.circle(img, (a, b), 1, (0, 0, 255), 3)
            
    img = Image.fromarray(img)
    display_result(img) #display the result image


#Dilate the input image
def Dilation():
    #use rectangle kernal 
    kernel = np.ones((5,5), np.uint8)
    #Where img_read_gray is gray input image 
    img_dilation = cv.dilate(img_read_gray, kernel, iterations=1) #perform dilate operation
    
    img_dilation = Image.fromarray(img_dilation)
    display_result(img_dilation) #display the result image


#erode the input image
def Erosion():
    #use rectangle kernal
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5,5), np.uint8)
    #Where img_read_gray is gray input image 
    img_erosion = cv.erode(img_read_gray, kernel, iterations=1) #perform erosion operation
    
    img_erosion = Image.fromarray(img_erosion)
    display_result(img_erosion) #display the result image
    

#perform close operation on the input image
def Close():
    #use rectangle kernal
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5,5), np.uint8)
    #Where img_read_gray is gray input image 
    img_dilation = cv.dilate(img_read_gray, kernel, iterations=1) #perform dilate operation first
    Close = cv.erode(img_dilation, kernel, iterations=1) # and then perform erosion operation on the result from the previous dilate operation
    
    Close = Image.fromarray(Close)
    display_result(Close) #display the result image


#perform open operation on the input image
def Open():
    #use rectangle kernal
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5,5), np.uint8)
    #Where img_read_gray is gray input image 
    img_erosion = cv.erode(img_read_gray, kernel, iterations=1) #perform erosion operation first
    Open = cv.dilate(img_erosion, kernel, iterations=1) # and then perform dilate operation on the result from the previous erosion operation
    
    Open = Image.fromarray(Open)
    display_result(Open) #display the result image



def kernal():
    pass


# functioon to save the result image on the device
def Save_Result_image():
    filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg") # extention of the result img
    if not filename:
        return
    image.save(filename)


#close the program
def Exit():
    root.destroy()





###### The GUI part
## main label frame 
labelframe_1 = LabelFrame(root, text='CSC8558/ programming assignment [by: Mohammed El-Nabawy Soliman Ghaze]')
labelframe_1.pack(fill='both', expand='yes')



## first lable frame and it's content
labelframe_11 = LabelFrame(labelframe_1, text='load image', width=160, height=85)
labelframe_11.place(x=15, y=5)
ttk.Button(labelframe_11, text='open', width=15, command=insert_image).place(x=25, y=15)



## second lable frame and it's content
labelframe_12 = LabelFrame(labelframe_1, text='convert', width=160, height=85)
labelframe_12.place(x=190, y=5)
var  = IntVar()
ttk.Radiobutton(labelframe_12, text = "Default color", variable = var, value = 1, command=convert).place(x=15,y=5)
ttk.Radiobutton(labelframe_12, text = "Gray color", variable = var, value = 2,command=convert).place(x=15, y=25)



## third lable frame and it's content
labelframe_13 = LabelFrame(labelframe_1, text='add noise', width=160, height=85)
labelframe_13.place(x=365, y=5)
var1 = IntVar()
ttk.Radiobutton(labelframe_13, text = "Salt & pepper noise", variable = var1, value = 1, command=noise).place(x=10,y=0)
ttk.Radiobutton(labelframe_13, text = "Gaussian noise", variable = var1, value = 2,command=noise).place(x=10, y=20)
ttk.Radiobutton(labelframe_13, text = "Poisson noise", variable = var1, value = 3,command=noise).place(x=10, y=40)



##label frame and it's content
labelframe_14 = LabelFrame(labelframe_1, text='''Point transform Op's''', width=510, height=155)
labelframe_14.place(x=15, y=100)
ttk.Button(labelframe_14, text='Brightness adjustment',width=25, command = Brightness_adjustment).place(x=10, y=10)
ttk.Button(labelframe_14, text='Contrast adjustment', width=25, command = Contrast_adjustmen).place(x=110, y=40)
ttk.Button(labelframe_14, text='Histogram', width=25, command = Histogram).place(x=220, y=70)
ttk.Button(labelframe_14, text='Histogram Equalization',width=25, command=Histogram_Equalization).place(x=330, y=100)



##label frame and it's content
labelframe_15 = LabelFrame(labelframe_1, text='''Local transform Op's''', width=510, height=180)
labelframe_15.place(x=15, y=265)
ttk.Button(labelframe_15, text='Low pass filter',width=25, command = Low_pass_filte).place(x=10, y=10)
ttk.Button(labelframe_15, text='High pass filter', width=25, command = High_pass_filter).place(x=10, y=45)
ttk.Button(labelframe_15, text='Median filter (gray image)', width=25, command = Median_filter).place(x=10, y=80)
ttk.Button(labelframe_15, text='Avereging filter',width=25, command=Avereging_filter).place(x=10, y=115)



#####label frame insid local label frame
labelframe_15_1 = LabelFrame(labelframe_15, text='Edge detection filter', width=300, height=150)
labelframe_15_1.place(x=190, y=0)
var2 = IntVar()
ttk.Radiobutton(labelframe_15_1, text = "Laplacian filter", variable = var2, value = 1, command=Edge_detection_filter).place(x=15,y=5)
ttk.Radiobutton(labelframe_15_1, text = "Vert.Prewitt", variable = var2, value = 2,command=Edge_detection_filter).place(x=15, y=25)
ttk.Radiobutton(labelframe_15_1, text = "Zero cross", variable = var2, value = 3,command=Edge_detection_filter).place(x=15, y=45)
ttk.Radiobutton(labelframe_15_1, text = "Gaussian filter", variable = var2, value = 4, command=Edge_detection_filter).place(x=15,y=65)
ttk.Radiobutton(labelframe_15_1, text = "Horiz.Prewitt", variable = var2, value = 5,command=Edge_detection_filter).place(x=15,y=85)
ttk.Radiobutton(labelframe_15_1, text = "Thicken", variable = var2, value = 6,command=Edge_detection_filter).place(x=15,y=105)
ttk.Radiobutton(labelframe_15_1, text = "vert.sobel", variable = var2, value = 7, command=Edge_detection_filter).place(x=135,y=5)
ttk.Radiobutton(labelframe_15_1, text = "Lap Of Gau(log)", variable = var2, value = 8,command=Edge_detection_filter).place(x=135,y=25)
ttk.Radiobutton(labelframe_15_1, text = "Skelton", variable = var2, value = 9,command=Edge_detection_filter).place(x=135,y=45)
ttk.Radiobutton(labelframe_15_1, text = "Horiz.sobel", variable = var2, value = 10, command=Edge_detection_filter).place(x=135,y=65)
ttk.Radiobutton(labelframe_15_1, text = "Canny method", variable = var2, value = 11,command=Edge_detection_filter).place(x=135,y=85)
ttk.Radiobutton(labelframe_15_1, text = "thinning", variable = var2, value = 12,command=Edge_detection_filter).place(x=135,y=105)



##label frame and it's content
labelframe_16 = LabelFrame(labelframe_1, text='''Global transform Op's''', width=260, height=160)
labelframe_16.place(x=15, y=460)
ttk.Button(labelframe_16, text='Line Detection using Hough Transform', width=35, command = Line_Detection).place(x=15, y=30)
ttk.Button(labelframe_16, text='Circles Detection using Hough Transform', width=35, command=Circles_Detection).place(x=15, y=80)



##label frame for morphological operation
labelframe_17 = LabelFrame(labelframe_1, text='''Morphological Op's''', width=245, height=160)
labelframe_17.place(x=280, y=460)
ttk.Button(labelframe_17, text='Dilation',width=15, command = Dilation).place(x=30, y=10)
ttk.Button(labelframe_17, text='Erosion', width=15, command = Erosion).place(x=30, y=40)
ttk.Button(labelframe_17, text='Close', width=15, command = Close).place(x=30, y=70)
ttk.Button(labelframe_17, text='Open',width=15, command=Open).place(x=30, y=100)



'''
l=ttk.Label(labelframe_17, text='Choose type of kernal').place(x=117, y=40)
n = StringVar()
kernal = ttk.Combobox(labelframe_17, width = 15, textvariable = n)
kernal['values'] = ('Arbitrary', 
                    'diamond',
                    'disk',
                    'line',
                    'octagon',
                    'pair',
                    'periodic',
                    'rectangle',
                    'square'
                    )
kernal.place(x=117, y=60)
kernal.current(0)
'''



#save the the result image
ttk.Button(root, text='Save Result image',width=25, command = Save_Result_image).place(x=70, y=650)

#Exit the program
ttk.Button(root, text='Exit', width=25, command = Exit).place(x=320, y=650)




labelframe_17 = LabelFrame(labelframe_1, text='''Original image''', width=260, height=260)
labelframe_17.place(x=530, y=70)

labelframe_18 = LabelFrame(labelframe_1, text='''after adding noise''', width=260, height=260)
labelframe_18.place(x=795, y=70)

labelframe_19 = LabelFrame(labelframe_1, text='''Result''', width=260, height=260)
labelframe_19.place(x=530, y=360)

labelframe_20 = LabelFrame(labelframe_1, text='''Histograme''', width=260, height=260)
labelframe_20.place(x=795, y=360)



panel = Label(labelframe_17)
noise_label = Label(labelframe_18)
result = Label(labelframe_19)


root.mainloop() 