
# do not edit this code. 
# A new code is at home Ubuntu machine (new predict_online_stroke_data functionality tested in the keras_deploy directory)
# -- tail at the ( would cause the symbole to be recognized as 6. Even though they are very similar, 
#     it looks like learning has issue. things to do
#     1. balance the data -- make the number of symbole to be 1000 for all (the large number of (_1_1 may caused ML to learn its features clearly so taht any deviation may cause ML to think the symbole is more close to 6_1_1. 
#     4. nornalize the values of feature so that values are around 100 (currently the count features are order of 10)
#     2. remove the noise of the tail of the stroke that changes abruptly and causes ( to be 6 
#     3. add a feature the number in vertical transition
#     5. increase the number of cells per layers
#.    6. make the H larger (40 -> 100 maybe?)

import numpy as np
import re
import math
import csv
import sys
from collections import Counter

erp = np.array( [ 61.,   95.,   60.,   94.,   61.,   92.,   66.,   89.,
    74.,   86.,   84.,   82.,   99.,   80.,  110.,   80.,  121.,  80.,
    130.,   80.,  137.,   83.,  141.,   84.,  145.,   88., 148.,   95.,  
    148.,  104.,  148.,  116.,  148.,  128.,  146.,  142.,  140.,  152.,  
    134.,  161.,  126.,  169.,  119.,  176.,  111.,  183.,  101.,  189.,   
    98.,  191.,   93.,  193.,   92.,  194.,   91.,  195.] )

# #https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
a = np.array([ [  0.  ,   0.  ],[  0.3 ,   0.  ],[  1.25,  0.1 ],
               [  2.1 ,  0.9 ],[  2.85,  2.3 ],[  3.8 ,  3.95],
               [  5.  ,  5.75],[  6.4 ,  7.8 ],[  8.05,  9.9 ],
               [  9.9 , 11.6 ],[ 12.05, 12.85],[ 14.25, 13.7 ],
               [ 16.5 , 13.8 ],[ 17.25, 13.35],[ 18.3 , 12.2 ],
               [ 19.8 , 10.5 ],[ 20.55,  8.15],[ 21.95,  6.1 ],
               [ 22.35,  3.95],[ 23.1 ,  1.9 ]])

#the logic does not care fo the comma, commas are for human readability, the function extract all numbers and reshape them
#in terms of num_col, which is the number of column of the two dimensional array
def str_to_np_array(trace_regular_string, num_col, verbose=False):
    refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", trace_regular_string)
    map_float = np.array( list( map(float, refindout)))
    strokes = np.reshape( map_float , (-1, num_col))
    if(verbose):
        print(trace_regular_string)
        print(refindout)
        print(map_float)
        print(strokes)
    return(strokes)

#https://stackoverflow.com/questions/6879596/why-is-the-python-csv-reader-ignoring-double-quoted-fields
#
def read_stroke_file(data_file, verbose=False):
    strk = []
    key = []
    length= []
    num = []
    with open(data_file, 'r') as f:
        #reader = csv.reader(f, delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
        reader = csv.reader(f, skipinitialspace=True) # delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
        l = next(reader, None)  #skipping header
#        trace_regular_idx = l.index('trace_regular') 
#do not use trace --- it has sometimes three element per point
        trace_regular_idx = l.index('trace_regular')
        symbol_final_idx = l.index('symbol_final')
        #print(l)
        for row in reader:
            #print(row)
            #print(row[0])
            #trace_regular =  row[13] #prior to Oct 3, when I modified regulization, I shifted 5 metric in add_trace_bb R function
            trace_regular =  row[trace_regular_idx]
            #print(trace_regular)
            key_ =  row[symbol_final_idx]
            strokes = str_to_np_array(trace_regular, 2)
            strk.append(strokes)
            key.append(key_)
            length.append(len(strokes))
            num.append(row[0]) #sequence
    return(strk, key, length, num)

#abc_train_2011_12_13_new_regul_clean_x_unbalanced.csv")

def stroke_zero_base ( strokes ):
    zero_org_strk = np.zeros(strokes.shape)
    scaled_strk = np.zeros(strokes.shape)
    x = strokes[:,0]
    y = strokes[:,1]
    x_max = max(x)
    x_min = min(x)
    y_max = max(y)
    y_min = min(y)
    x_len = x_max - x_min
    y_len = y_max - y_min
    xy_ratio = float(y_len / x_len)
    if(False):
        print("x_len %5.2f y_len %5.2f xy_ratio %5.2f" % (x_len, y_len, xy_ratio))
    zero_org_strk[:,0]  = strokes[:,0] - x_min #x origin = 0
    zero_org_strk[:,1]  = strokes[:,1] - y_min #y origin = 0
    scaled_strk[:,0] = (zero_org_strk[:,0] / y_len ) * 100
    scaled_strk[:,1] = (zero_org_strk[:,1] / y_len ) * 100
    return x_len, y_len, xy_ratio, zero_org_strk, scaled_strk

 

#Fix that if two points are identical, it should add some noise so that it won't interpolate the same points 
#over and over again. 

def interpolate(stroke, dynamic=True, verbose=False):
    if(verbose):
        print( "enter the interpolate func")
        print( "enter the interpolate func: dynamic interporation is implemented 2018-01-21")
    a = []
    local_staying_flag = 0
    for s in range(1,len(stroke)):
        x=stroke[s,0]
        y=stroke[s,1]
        prev_x = stroke[(s-1),0]
        prev_y = stroke[(s-1),1]
        if( x== prev_x and y == prev_y ):
            #sys.exit()
            prev_x = prev_x + 0.001 #this is to prevent crash in curvature calculation
            prev_y = prev_y + 0.001 #this is to prevent crash in curvature calculation
        dist = np.sqrt(  np.square(x-prev_x) + np.square(y-prev_y))
        if( False ) :
            print( dist )
        if( dist < 1.0):
            local_staying_flag = local_staying_flag + 1
        else:
            local_staying_flag = 0
        if( local_staying_flag > 5 and False):
            print( dist )
            print( " ********************************* 5 strokes staying at one place -- local ")
            #sys.exit()             
        fill = 0
        if dynamic:
            if dist >= 3 and dist < 10:
                fill = 3
            elif dist >= 10 and dist < 20: # dynamic filling for large gaps
            #the unit of dist is H=100 scale. 
                fill = 5
            elif dist >= 20 and dist < 30: #
                fill = 8
            elif dist >= 30 :
                fill = 10
            else: #if dist <=2:
                fill = 0
        for k in range(fill+1): #this should not be zero because at least one should place original point
            if( x != prev_x):
                slope = (y-prev_y) / (x-prev_x)
                    #print( "prev_x %3.2f, prev_y %3.2f; x %3.2f, y %3.2f, slope %3.2f" %(prev_x, prev_y, x, y, slope))
            #for k in range(fill+1):
                tmp_x = prev_x + k*(x-prev_x)/(fill+1)
                tmp_y = slope * (tmp_x - prev_x) + prev_y
            #        a.append(tmp_x)
            #        a.append(tmp_y)
            #        print( "   tmp_x %3.2f, tmp_y %3.2f" %(tmp_x, tmp_y))
            else:
            #for k in range(fill+1):
                tmp_x = x
                tmp_y = prev_y + k*(y-prev_y)/(fill+1)
            #print( "   tmp_x %3.2f, tmp_y %3.2f" %(tmp_x, tmp_y))
            a.append(tmp_x)
            a.append(tmp_y)
            if( k == 0): 
                a.append(s) #original points
            else: 
                a.append(-1) #filled points
    a.append(stroke[-1][0]) #add the last element
    a.append(stroke[-1][1]) #add the last element
    a.append(len(stroke))
    b = np.reshape(np.array(a), (-1,3))
    return(b)

def get_distance (stroke, verbose=False):
    a = []; a.append(np.nan);
    for s in range(1,len(stroke)):    
        x=stroke[s,0]
        y=stroke[s,1]
        prev_x = stroke[(s-1),0]
        prev_y = stroke[(s-1),1]
        dist = np.sqrt(  np.square(x-prev_x) + np.square(y-prev_y))
        a.append(dist)
    a = np.reshape(a, (-1,1))
    if(verbose):
        print(a.shape)
    return(a)

def stroke_to_img( scaled_strk, frame_tall, xy_ratio, shift_dots = 3, verbose=False):
    if(verbose):
        print("enter the func: stroke_to_img")
    location = []
    frame_wide =  int(math.ceil(frame_tall/xy_ratio)) + 6 #, 12)
    img = np.zeros( (frame_tall, frame_wide) )
    if(verbose):
        print( "frame tall %d, wide %d, xy_ratio %3.2f ---> 100_tall %d, 100_wide %d" % ( frame_tall, frame_wide ,xy_ratio , 100, int((frame_wide*100/frame_tall)) ))
    length = len(scaled_strk)
    H = .8*frame_tall  #shrink actual image to 80% of the image size. 40*.8 = 32 
    prev_x = prev_y = 0
    for s in range(length):
        img_x = int(math.floor(scaled_strk[s,0] * H/100)) + shift_dots
        img_y = int(math.floor(scaled_strk[s,1] * H/100)) + shift_dots + 1
        #print("i %d, x %d, y %d "%(s, img_x, img_y)) # the bug is that the image should be shrinked and shifted to prevent negative index and out of box index
        #here y and x order is there because of ease of printing in the screen.
        img[ img_y, img_x] = 1
        location.append( [img_x, img_y ]) #location is y and x for human readability
        #x_dis = (img_x - prev_x )
        #y_dis = (img_y - prev_y )
        #if(s > 0 and (abs(x_dis) > 1 or abs(y_dis) > 1)): #a large jump indicator
                #print( "x_dis %d, y_dis %d" %(x_dis, y_dis, ))
        #prev_x = img_x
        #prev_y = img_y
    #new_strk = np.concatenate((scaled_strk, location), axis=1) #remember location in image is is y and x not x and y
    #return(img, new_strk)
    return(img, location)   

def print_array(img):
    print([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9])
    for s in range(img.shape[0]):
        x = "%s " % img[s, :]   #crazy string format inserting \n in the middle, --> removed at the next line
        print( re.sub('\n', '', x))


def img_padding(img):
    img2 = np.zeros( (img.shape[0] + 2, img.shape[1] + 2))
    #img = np.zeros( (img_size, math.ceil(img_size/xy_ratio)) )
    #img_pad = matrix(0,img_height+2,img_width+2) #padding
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
                img2[i+1,j+1] = img[i,j]
    return(img2)


def img_smoothing(img, width=3):
    gauss = np.array([[0.0625, 0.125, 0.0625],
    [0.1250, 0.250, 0.1250],
    [0.0625, 0.125, 0.0625]])
    pad = img_padding(img)
    smooth_img = np.zeros(  (img.shape[0], img.shape[1] ))
    #width = 3 #filter width
    #img_pad = matrix(0,img_height+2,img_width+2) #padding
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            temp = pad[row:(row + width), col:(col + width)]
            smooth_img[row,col] = np.sum( np.multiply(gauss, temp))
    return(smooth_img)


#shape gives nrow and ncol, so 40,5 means 40 rows and 5 columns. so y and x is the sequence in 

def img_filtering(img, filter, width=3):
    pad = img_padding(img)
    smooth_img = np.zeros(  (img.shape[0], img.shape[1] ))
    #width = 3 #filter width
    #img_pad = matrix(0,img_height+2,img_width+2) #padding
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            temp = pad[row:(row + width), col:(col + width)]
            smooth_img[row,col] = np.sum( np.multiply(filter, temp))
    return(smooth_img)


# FKI features
# 1. count 1
# 2. mean
# 3. variance
# 4. upper contour
# 5. lower contour
# 6. upper contour direction (<0 ascend, >0 descending)
# 7. lower contour direction (<0 ascend, >0 descending)
# 8. 0/1 transitions
# 9. count 1 between upper and lower contour. --- THIS IS THE SAME AS FEATURE 1. WHAT'S WRONG WITH THIS? KEEP IT FOR NOW
def offline_FKI(binary_img):
    H = binary_img.shape[0]
    W = binary_img.shape[1]
    FKI = np.zeros( (9, W))
    FKI[0,:] = np.sum( binary_img , axis = 0) # Number of black pixels in the column:  which is count
    #np.multiply( np.array(range(H)), binary_img[:,0])/H
    range_x = np.array(range(H))
    for j in range(W):
        if(FKI[0, j] != 0):
            nonzeros = np.multiply( range_x, binary_img[:,j])
            nonzeros = nonzeros[nonzeros!=0] #https://stackoverflow.com/questions/46383987/remove-zeros-from-nparray-efficiently
            FKI[1, j] = np.mean(nonzeros) #which is sum(nonzeros)/FKI[0,j] # Center of gravity of the column
            #Second order moment of the column:
            #FKI[2, j] = sum(np.square(np.multiply( range_x, binary_img[:,j])))/np.square(FKI[0,j]) # this does not look correct for the variance caculation 
            FKI[2, j] = np.sqrt( np.var(nonzeros) ) # Second order moment of the column:
            FKI[3, j] = np.flatnonzero(binary_img[:,j])[0]  # Position of the upper contour in the column:
            FKI[4, j] = np.flatnonzero(binary_img[:,j])[-1] # Position of the lower contour in the column:
            FKI[7, j] = np.sum(abs( np.roll(binary_img[:,j], 1) - binary_img[:,j] )) # Number of black-white transitions in the column:
            FKI[8, j] = np.sum(binary_img[ int(FKI[3,j]): int(FKI[4,j]), j])+1 # Number of black pixels between the upper and lower contours
    for j in range(W):
        if(FKI[0, j] != 0):
            if (j == 0):
                FKI[5,j] = (FKI[3,(j+1)] - FKI[3,j]) # Orientation of the upper contour in the column  <0 means going up, >0 means going down
                FKI[6,j] = (FKI[4,(j+1)] - FKI[4,j]) # Orientation of the lower contour in the column
            elif(j == (W-1)):
                FKI[5,j] = (FKI[3,W] - FKI[3,(W-1)])
                FKI[6,j] = (FKI[4,W] - FKI[4,(W-1)])
            else:
                FKI[5,j] = (FKI[3,(j+1)] - FKI[3,(j-1)])/2
                FKI[6,j] = (FKI[4,(j+1)] - FKI[4,(j-1)])/2
    return(FKI)





#img = tight_img_map(scaled_strk, 40, xy_ratio)
#print_array(img)
#http://mccormickml.com/2013/02/26/image-derivative/
#http://mccormickml.com/2013/02/26/image-derivative/
#http://mccormickml.com/2013/02/26/image-derivative/

#for filtering feed 255*img_after_smooth images ---> first derivatives are -128 to 128 values
#x first derivative
x_derivate_mask =     np.array([[-1.0, 0.0, 1.0],
                                [-1.0, 0.0, 1.0],
                                [-1.0, 0.0, 1.0]]) / 3

#y first derivative
y_derivate_mask =     np.array([[ -1.0, -1.0, -1.0],
                                [ 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0]]) / 3

#I do not know the range of the second derivative values
#x second derivative
x_laplacian_mask =     np.array([[0.0, 0.0, 0.0],
                                [1.0, -2.0, 1.0],
                                [0.0, 0.0, 0.0]])  

#y second derivative
y_laplacian_mask =     np.array([[ 0.0, 1.0, 0.0],
                                [ 0.0, -2.0, 0.0],
                                [0.0,  1.0, 0.0]])  



# strk_read, key_read = read_stroke_file("/Users/youngpark/Documents/handwritten-mathematical-expressions/Data_Stroke/abc_train_2011_12_13.csv")
# x_len, y_len, xy_ratio, zero_org_strk, scaled_strk = stroke_zero_base(strk_read[0])
# filled_stroke = interpolate(scaled_strk, fill=2)
# img_before_smooth, new_strok = stroke_to_img(filled_stroke, 40, xy_ratio, margin=3)
# img_after_smooth = img_smoothing(img_before_smooth)
# binary_img = np.ceil(img_after_smooth)
# FKI = offline_FKI(binary_img)
# print_array( binary_img )
# print_array( FKI )



#curvature may be also important feature
#a nice animation for the curvature
#https://mathematica.stackexchange.com/questions/95425/can-i-get-the-curvature-at-any-point-of-a-random-curve
#https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy

#After I chaged interpoation method, the curvature got changed--- less fill in for short distanced pairs
#this imporved tail high curvature effects in the q_1_1 (the first data in the train data)

#noisy curve causes curvature to be jumpy and shaky
#https://stackoverflow.com/questions/50604298/numerical-calculation-of-curvature
def get_curvature(filled_strk, distance, verbose=False):
    a = filled_strk
    if(False):
        print(" filled n scaled strk %s" % (a))
        print_array(a)
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])
    #velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    #ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    #tangent = np.array([1/ds_dt] * 2).transpose() * velocity
    #d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    denom = (dx_dt * dx_dt + dy_dt * dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    #singluar point of x,y both got zero slopes
    #when user strokes lingering in a small area, all sorts of strange behaviors are occuring
        # two points are having the same values
        # two points are symmetric and having zero slopes in x and y direction
        # what else? 
    if( np.count_nonzero(denom) != len(denom) and False): #detect zeros in demoninator -- both x' and y' are zeros
        for idx,val in enumerate(denom):
            if(val == 0): #x' and y' are both zeros
                print(" denominator contains zero (dx_dt * dx_dt + dy_dt * dy_dt) ")# % (dx_dt * dx_dt + dy_dt * dy_dt))
                print(" idx %s " % (idx))# % (dx_dt * dx_dt + dy_dt * dy_dt))
                print( "dx_dt %s dy_dt %s d2x_dt2 %s d2y_dt2 %s " % (dx_dt[idx], dy_dt[idx], d2x_dt2[idx], d2y_dt2[idx]))
                print(" curvature %s " % (curvature[idx]))
        print(a.shape)
        print(distance.shape)
        print(curvature.shape) 
        #sys.exit("-=-=->>>>>>>> get_curvature: denominator is zero ")             
    if(verbose):
        with np.printoptions(precision=3, suppress=True):
    #print(x)np.set_printoptions(precision=4)       
            print( np.concatenate((a, distance, np.reshape(curvature, (-1,1))), axis=1) )
        #print(np.concatenate((a, np.reshape(curvature, (-1,1))), axis=1))
    if(verbose):
        print(" \n\n denominator dx_dt * dx_dt + dy_dt * dy_dt %s " % (dx_dt * dx_dt + dy_dt * dy_dt))
    return (curvature) 

#https://stackoverflow.com/questions/42882842/whats-the-difference-between-n-and-n-1-in-numpy

# strk_read, key_read = read_stroke_file("/Users/youngpark/Documents/handwritten-mathematical-expressions/Data_Stroke/abc_train_2011_12_13.csv")
# x_len, y_len, xy_ratio, zero_org_strk, scaled_strk = stroke_zero_base(strk_read[0])
# print( get_curvature(scaled_strk))



def get_binary_img (img_after_smooth, digitized_strok, verbose = False):
    binary_img = np.ceil(img_after_smooth)
    binary_img_display = np.ceil(img_after_smooth)*1 # *7 to display 7 for nonzero values
    if(verbose):
        print( "enter the get_binary_img func")        
        print( img_after_smooth.shape)
    for idx, i in enumerate(digitized_strok):
        x = int(i[3])
        y = int(i[4])
#        print( i, i[3], i[4] )
        if(i[2] != -1 ): #original strok
            binary_img_display[y,x] = 9 #(idx%5)+1
        if( i[2] == -1 ): #fillers
            if( idx < 20):
                binary_img_display[y,x] = 2 #(idx%5)+1
            elif( idx < 30 and idx >= 20 ):
                binary_img_display[y,x] = 3 #(idx%5)+1
            elif( idx < 40 and idx >= 30 ):
                binary_img_display[y,x] = 4 #(idx%5)+1
        if(idx == 0): #first
            binary_img_display[y,x] = 8        #if(idx == (len(digitized_strok)-1)):
        #    binary_img_display[ int(i[3])][int(i[4])] = 9
    return(binary_img, binary_img_display)

#http://www.ugrad.math.ubc.ca/coursedoc/math100/notes/apps/second-deriv.html
#We say that a graph is concave down when it is curved in this way. When the second derivative is negative, 
#it implies that the graph is concave down since the slope of tangent lines must be decreasing. Here is 
#another example of a graph which is concave down.
#
#[Toselli  2007]
#http://users.dsic.upv.es/~ajuan/research/2004/Juan04_08c.pdf
#Then each cell is characterized
#by the following features: normalized grey level, horizontal
#grey-level derivative and vertical grey-level derivative. 
#y'' and x'' are all negative because all concave down 

def online_derivatives(img_after_smooth, filled_stroke, distance, verbose=False):
    img_after_smooth_255 = img_after_smooth*255    #print( img_after_smooth_255[13:16, 12:15])    #print( binary_img_display[13:16, 12:15])
    yd = img_filtering(img_after_smooth_255, y_derivate_mask)
    xd = img_filtering(img_after_smooth_255, x_derivate_mask)
    ydd = img_filtering(img_after_smooth_255, y_laplacian_mask)
    xdd = img_filtering(img_after_smooth_255, x_laplacian_mask)
    curv = get_curvature(filled_stroke, distance, verbose)
    return (xd, yd, xdd, ydd, curv)
#
def extract_features( s_nparray , H=40 , verbose = False):#, display_image = False):
    x_len, y_len, xy_ratio, zero_org_strk, scaled_strk = stroke_zero_base(s_nparray)
    filled_stroke = interpolate(scaled_strk, dynamic=True)
    distance = get_distance(filled_stroke)
    img_before_smooth, location = stroke_to_img(filled_stroke, H, xy_ratio, shift_dots=3)
    digitized_strok = np.concatenate((filled_stroke, location), axis=1)
    img_after_smooth = img_smoothing(img_before_smooth)
    binary_img, binary_img_display = get_binary_img(img_after_smooth, digitized_strok)
    FKI = offline_FKI(binary_img)
    x_der, y_der, x_der_der, y_der_der, curv = online_derivatives(img_after_smooth, filled_stroke, distance, verbose)
#    if(display_image):
    if(False):
        print_array( binary_img_display )
        print_array(digitized_strok)
        print( "SHAPES: digitized_strok %s, binary_img %s, FKI %s, y' %s, x' %s, y'' %s x'' %s curv %s" 
        %(digitized_strok.shape, binary_img.shape, FKI.shape, y_der.shape, x_der.shape, y_der_der.shape, x_der_der.shape, curv.shape))
    #output formatting
    features = ""
    for idx,val in enumerate(digitized_strok):
        x = val[0]; y = val[1]; filler = int(val[2]); img_y = int(val[4]); img_x = int(val[3]); #y=3 and x=4 because this is image location 
        #for i in range(9):
        y_d = y_der[img_y, img_x] 
        x_d = x_der[img_y, img_x] 
        y_dd = y_der_der[img_y, img_x] 
        x_dd = x_der_der[img_y, img_x]  #
        curv_xy = curv[idx] #= (dx1*dy2 - dx2*dy1)/((dx1^2 + dy1^2)^(3/2)); erp029
        if(False): #this does not work, it does not capture curvature of the line but more of that of 3D dome of the color intensity
            if( x_d ==0 and y_d == 0):
                curv_xy = 0
            else:
                curv_xy = np.abs((x_dd*y_d - x_d*y_dd)/ ((x_d*x_d + y_d*y_d)**1.5)) #.  d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        if(val[2] != -1 and False): #original strok
            print("%04d  %s" % (idx, val))
            print("   FKI %5.2f, %5.2f, %5.2f, %5.2f, %5.2f,%5.2f,%5.2f,%5.2f,%5.2f " % (FKI[0, img_x],FKI[1, img_x],FKI[2, img_x],FKI[3, img_x],FKI[4, img_x],FKI[5, img_x],FKI[6, img_x],FKI[7, img_x],FKI[8, img_x]))
            print("   y' %5.2f, x' %5.2f, y'' %5.2f x'' %5.2f curv %7.4f" % (y_d, x_d, y_dd, x_dd, curv_xy))
        if(val[2] != -1 and True): #original strok
            features = features +  ("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f," % 
                (x, y, x_d, y_d, x_dd, y_dd, FKI[0, img_x],FKI[1, img_x],FKI[2, img_x],FKI[3, img_x],FKI[4, img_x],FKI[5, img_x],FKI[6, img_x],FKI[7, img_x],FKI[8, img_x])) 
            #output =           ("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f," % 
            #    (x, y, x_d, y_d, x_dd, y_dd, FKI[0, img_x],FKI[1, img_x],FKI[2, img_x],FKI[3, img_x],FKI[4, img_x],FKI[5, img_x],FKI[6, img_x],FKI[7, img_x],FKI[8, img_x])) 
            #print(output)
#    return(output + "\"")
    return(features[:-1], binary_img_display) #remove last comma

# def extract_features_and_write_file( rfile, wfile = "hijk_output_features.csv", ifile="hijk_output_images.py", symbol = "x_2_left" ):
#     strk_read, key_read, length, num_id = read_stroke_file(rfile)
#     #"/Users/youngpark/Documents/handwritten-mathematical-expressions/Data_Stroke/abc_train_2011_12_13_new_regul_clean_x.csv")
#     output = []
#     imageFile = open(ifile, 'wb')
#     resultFile = open(wfile,'wb') #as resultFile:
#     for idx, val in enumerate(strk_read):
#     #    for idx, val in enumerate(strk_read[1:5]):
#         #    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
#         if( idx%1000 == 0):
#             print("%d, %s" % (idx, key_read[idx]))
#         feature_out, image = extract_features(val, verbose = False); 
#         #output.append("%d, %s, %s" % (idx, feature_out, key_read[idx]))
#         r = "\"%d\", \"%s\", \"%s\"" % (idx, feature_out, key_read[idx])
#         resultFile.write(r + "\n")
# #        if(key_read[idx] == symbol):
#         if(True):
#             imageFile.write( "------------------ %05d, %s\n" % (idx, key_read[idx]) ) #print(np.array2string(image))
#             for row in image:
#                 num = ""
#                 for col in row:
#                     num = num + ("%d"%(col))
#                 imageFile.write( num + "\n")
#                 #imageFile.write( "\n" )
#     resultFile.close()
#     imageFile.close()

def ordered_extract_features_and_write_file( rfile, wfile = "abc_features_extracted.csv", ifile="abc_images_ordered.py", symbol = "x_2_left" ):
    strk_read, key_read, length, num_id = read_stroke_file(rfile)
    #"/Users/youngpark/Documents/handwritten-mathematical-expressions/Data_Stroke/abc_train_2011_12_13_new_regul_clean_x.csv")
    feature_list = []
    image_list = []
    resultFile = open(wfile,'w') #as resultFile:
    print("=-=-=-=-=-=-=-=-=-=-=- Extract Features -=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-")
    for idx, val in enumerate(strk_read):
    #    for idx, val in enumerate(strk_read[1:5]):
        if( idx%1000 == 0):
            print("extracting features: %d, %s" % (idx, key_read[idx]))
        feature_out, image = extract_features(val, verbose = False); 
        feature_list.append(feature_out)
        #print(feature_out)
        image_list.append(image)
        #print_array(image)
        #output.append("%d, %s, %s" % (idx, feature_out, key_read[idx]))
        r = "\"%05d\", \"%s\", \"%s\"" % (idx, feature_out, key_read[idx])
        #print(r)
        resultFile.write(r + "\n")
    resultFile.close()
#
    print("=-=-=-=-=-=-=-=-=-=-= Image Writing ordered by symbol (ascending order) -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    imageFile = open(ifile, 'w')
    unique_keys = np.unique(key_read)
    print(unique_keys)
    for idx_key, key in enumerate(unique_keys):
        print("writing image for: %s [%d] out of %d" % (key, idx_key, len(unique_keys)))
        for idx, image in enumerate(image_list):
            if( key_read[idx] == key):
                imageFile.write( "------------------ %05d, %s\n" % (idx, key_read[idx]) ) #print(np.array2string(image))
                for row in image:   
                    num = ""
                    for col in row:
                        num = num + ("%d"%(col))
                    imageFile.write( num + "\n")
                    #imageFile.write( "\n" )
    imageFile.close()

#     #print(output)
#     with open("output_features.csv",'wb') as resultFile:
# #        wr = csv.writer(resultFile, dialect='excel')
#         # wr = csv.writer(resultFile)
#         # for i in output:
#         #     wr.writerow(i)
#         for r in output:
#             resultFile.write(r + "\n")
#         resultFile.close()

#def helper_print_symbol_image( rfile, wfile, symbol ):
#    strk_read, key_read, length, num_id = read_stroke_file(rfile)

def count_keys_in_file( rfile ):
    strk_read, key_read, length, num_id = read_stroke_file(rfile)
    count = Counter(key_read)
    print(count)

#{'2_1_1': 6774, '(_1_1': 3696, ')_1_1': 3671, 'x_2_left': 3021, 'x_2_right': 2963, '3_1_1': 2815, '1_1_1': 2406, 'a_1_1': 2212, 
#'n_1_1': 2061, '0_1_1': 1746, '\\sqrt_1_1': 1738, 'y_1_1': 1505, 'b_1_1': 1434, '4_2_nose': 1378, 'z_1_1': 1345, 'd_1_1': 925, 
#'c_1_1': 871, '6_1_1': 811, '7_1_1': 769, '8_1_1': 758, '9_1_1': 748, '5_2_hook': 681, 'i_2_1': 653, 'x_1_1': 634, '\\int_1_1': 594, 
#'\\alpha_1_1': 544, 't_2_tail': 541, 'e_1_1': 526, '4_1_1': 460, 'm_1_1': 443, '\\infty_1_1': 361, 'y_2_flower': 356, 'r_1_1': 355, 
#'\\beta_1_1': 354, '\\lt_1_1': 328, 'f_2_cobra': 317, 'j_2_1': 311, 'p_1_1': 269, 'g_1_1': 258, 'u_1_1': 244, 'k_1_1': 242, 
#'\\theta_1_1': 238, '5_1_1': 236, '\\sum_1_1': 223, 'v_1_1': 223, '\\sum_2_bot': 212, 'h_1_1': 202, ']_1_1': 194, 's_1_1': 172, 
#'p_2_ear': 159, '/_1_1': 157, 'q_1_1': 143, '[_1_1': 121, 'x_2_left_south_east': 113, '\\}_1_1': 101, '\\{_1_1': 101, '\\tan_3_an': 100, 
#'w_1_1': 99, 'L_1_1': 96, 'o_1_1': 85, '\\cos_2_co': 77, 'x_2_left_CC': 76, '\\gamma_1_1': 68, '\\log_2_lo': 68, 'x_2_right_CL': 66, 
#'\\lim_2_lim': 66, '\\gt_1_1': 57, '\\cos_1_cos': 48, '\\sigma_1_1': 42, 'l_1_1': 38, '\\mu_1_1': 28, '\\log_1_log': 26, '\\lim_3_li': 24, 
#'x_2_right_south_west': 1})


def read_feature_file( rfile = "./output_features.csv"):
    with open(rfile, 'r') as f:
    #reader = csv.reader(f, delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
        reader = csv.reader(f, skipinitialspace=True) # delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
        #next(reader, None)  #skipping header
        for row in reader:
            print(row[0])
            print(row[1])
            print(row[2])
        #trace_regular =  row[13] #prior to Oct 3, when I modified regulization, I shifted 5 metric in add_trace_bb R function
            #trace_regular =  row[18]
        #print(trace_regular)
            key =  row[-1]
            print(key)
#    if( verbose ):
#        print( filled_stroke )
#        print_array( binary_img_display )
#        print_array( x_der )
#        print_array( y_der )        
#        print_array( x_der_der )
#        print_array( y_der_der )
#        print(curv)
#        print( FKI )
#these are shapes of data from analyzing the first symbol of data, which is q symbole 
#SHAPES: digitized_strok (211, 5), binary_img (40, 34), FKI (9, 34), y' (40, 34), x' (40, 34), y'' (40, 34) x'' (40, 34) curv (211,)
#[ 0.86956522 13.04347826  1.          3.          8.        ]


# def predict_online_stroke_data(canvassData, symbol_list):
#     decoded_string = canvassData.decode("UTF-8")
#     refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", decoded_string)
#     number_array = np.array( list( map(float, refindout)))
# #    print(decoded)
# #    print(type(decoded))    #print("100")
#     #print(type("100"))
#     #print(imgData2.decode("UTF-8"))
# #    print("-------------------")
# #    print(decoded)
#     npa = np.array(number_array) #I don't think this is necessary to wrap using np.array --- again
# #    print("-------------------")
# #    print(npa)
# #    print(type(npa))
#     spl = np.flatnonzero( npa == -999)
# #    print(spl)
#     strks = np.split(npa, spl)[1:] #split and throw away the empty first
# #    print(strks)
#     np_strokes = [x[1:] for x in strks] #remove -999 from strokes
#     for n in np_strokes:
#         #out_string = out_string + predict_strokes(n) + "===="
#         np_shaped_xy = np.reshape( n , (-1, 2))
#         np_features = extract_features(np_shaped_xy, verbose = False)
#         tensor_x = strokes.reshape(1, len(np_features), 15) #why do I this again? Keras is expecting 3D tensor
# #print(x)
# #with graph.as_default():
# #perform the prediction
#         predict_out = model.predict(tensor_x)
#         out = predict_out.flatten().tolist()
# #
#         largest_three_symidx = [out.index(x) for x in sorted(out, reverse=True)[:3]]
#         print(largest_three_symidx)
# #
#         largest_three_out = [x for x in sorted(out, reverse=True)[:3]]
#         print(largest_three_out)
# #
#         num_1_sym  = symbol_list[ largest_three_symidx[0] ]
#         num_2_sym  = symbol_list[ largest_three_symidx[1] ]
#         num_3_sym  = symbol_list[ largest_three_symidx[2] ]
#         num_1_prob  =  largest_three_out[0]
#         num_2_prob  =  largest_three_out[1]
#         num_3_prob  =  largest_three_out[2]
#         predicted = " [%s, %s, %s]= [%3.2f,%3.2f,%3.2f] = %d" %(num_1_sym, num_2_sym, num_3_sym,
#         num_1_prob, num_2_prob, num_3_prob , len(strokes))
#         print (predicted)

def deploy_predict_online_stroke_data(canvassData, model, symbol_list):
    decoded_string = canvassData.decode("UTF-8")
    refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", decoded_string)
    number_array = np.array( list( map(float, refindout)))
    print(decoded_string)
#    print(type(decoded))    #print("100")
    #print(type("100"))
    #print(imgData2.decode("UTF-8"))
#    print("-------------------")
#    print(decoded)
    npa = np.array(number_array) #I don't think this is necessary to wrap using np.array --- again
#    print("-------------------")
#    print(npa)
#    print(type(npa))
    spl = np.flatnonzero( npa == -999)
#    print(spl)
    strks = np.split(npa, spl)[1:] #split and throw away the empty first
#    print(strks)
    np_strokes = [x[1:] for x in strks] #remove -999 from strokes
    send_to_http = ""
    for n in np_strokes:
        #out_string = out_string + predict_strokes(n) + "===="
        np_shaped_xy = np.reshape( n , (-1, 2))
        str_features, binary_img_display = extract_features(np_shaped_xy, verbose = False)
        print(str_features)
        np_features = str_to_np_array(str_features,15)
        tensor_x = np_features.reshape(1, len(np_features), 15) #why do I this again? Keras is expecting 3D tensor
        print_array(tensor_x)
        print(tensor_x.shape)
#with graph.as_default():
#perform the prediction
        predict_out = model.predict(tensor_x)
        out = predict_out.flatten().tolist()
#
        largest_three_symidx = [out.index(x) for x in sorted(out, reverse=True)[:3]]
        print(largest_three_symidx)
#
        largest_three_out = [x for x in sorted(out, reverse=True)[:3]]
        print(largest_three_out)
#
        num_1_sym  = symbol_list[ largest_three_symidx[0] ]
        num_2_sym  = symbol_list[ largest_three_symidx[1] ]
        num_3_sym  = symbol_list[ largest_three_symidx[2] ]
        num_1_prob  =  largest_three_out[0]
        num_2_prob  =  largest_three_out[1]
        num_3_prob  =  largest_three_out[2]
        predicted = " [%s, %s, %s]= [%3.2f,%3.2f,%3.2f] = %d" %(num_1_sym, num_2_sym, num_3_sym,
        num_1_prob, num_2_prob, num_3_prob , len(np_shaped_xy))
        print (predicted)
        send_to_http = send_to_http +  predicted + "++++++++++++++++>"
    return(send_to_http)

def stroke_to_features( s_string , verbose = False):
    np_array =  str_to_np_array(s_string, 2, verbose=False) #stroke string to col=2 np_array
    feature_out, image = extract_features(np_array, verbose = False) #[-1,2] stroke to [-1,15] features
    return(feature_out)

#feature created are:
#"0", "
#32.918 32.902 79.688 116.875 -15.938 -95.625 7.000 10.000 5.292 3.000 16.000 0.000 0.000 4.000 7.000,
#39.243 31.009 53.125 175.312 -31.875 -47.812 9.000 9.778 4.894 3.000 16.000 0.500 8.500 4.000 9.000,45.568 29.117 -31.875 63.750 -47.812 -79.688 34.000 20.500 9.811 4.000 37.000 0.500 2.000 2.000 34.000,49.369 25.315 -53.125 26.562 -111.562 0.000 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,50.000 18.991 -127.500 0.000 -63.750 -63.750 15.000 19.800 12.216 6.000 37.000 2.000 -0.500 4.000 15.000,48.738 12.019 -37.188 26.562 -95.625 -31.875 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,43.044 5.694 -21.250 26.562 -47.812 -95.625 26.000 19.885 8.478 4.000 33.000 0.500 10.500 4.000 26.000,34.180 1.262 -5.312 10.625 -15.938 -127.500 7.000 10.000 5.292 3.000 16.000 0.000 0.000 4.000 7.000,22.792 0.000 0.000 0.000 0.000 -127.500 6.000 9.500 5.560 3.000 16.000 0.000 0.000 4.000 6.000,12.666 1.893 15.938 42.500 -15.938 -95.625 7.000 9.000 5.292 3.000 16.000 0.000 0.000 4.000 7.000,5.063 6.956 5.312 5.312 -47.812 -47.812 13.000 10.000 3.742 4.000 16.000 -0.500 0.500 2.000 13.000,0.647 15.820 0.000 0.000 -127.500 0.000 11.000 10.000 3.162 5.000 15.000 -1.000 1.000 2.000 11.000,0.000 23.423 0.000 0.000 -127.500 0.000 11.000 10.000 3.162 5.000 15.000 -1.000 1.000 2.000 11.000,2.539 31.009 53.125 -15.938 -79.688 -31.875 11.000 10.000 3.162 5.000 15.000 -1.000 1.000 2.000 11.000,8.864 34.811 26.562 -79.688 -63.750 -95.625 8.000 10.000 4.637 4.000 16.000 -0.500 0.000 4.000 8.000,17.729 36.703 0.000 0.000 0.000 -127.500 6.000 9.500 5.560 3.000 16.000 0.000 0.000 4.000 6.000,29.748 36.703 47.812 -42.500 15.938 -127.500 7.000 10.000 5.292 3.000 16.000 0.000 0.000 4.000 7.000,38.612 35.442 -5.312 -154.062 15.938 -47.812 9.000 9.778 4.894 3.000 16.000 0.500 8.500 4.000 9.000,44.937 31.640 -85.000 -58.438 -63.750 -47.812 34.000 20.500 9.811 4.000 37.000 0.500 2.000 2.000 34.000,48.107 26.577 -53.125 26.562 -111.562 0.000 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,50.000 20.252 -127.500 0.000 -63.750 -63.750 15.000 19.800 12.216 6.000 37.000 2.000 -0.500 4.000 15.000,49.369 16.451 31.875 31.875 -127.500 0.000 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,48.738 13.297 0.000 21.250 -127.500 0.000 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,48.107 13.297 0.000 21.250 -127.500 0.000 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,48.107 13.927 0.000 21.250 -127.500 0.000 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,48.107 17.082 31.875 31.875 -127.500 0.000 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,48.107 24.054 15.938 -10.625 -127.500 15.938 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,46.845 32.902 -85.000 -58.438 -63.750 -47.812 34.000 20.500 9.811 4.000 37.000 0.500 2.000 2.000 34.000,46.215 43.675 -5.312 -5.312 -127.500 0.000 34.000 20.500 9.811 4.000 37.000 0.500 2.000 2.000 34.000,45.568 56.325 0.000 0.000 -127.500 0.000 34.000 20.500 9.811 4.000 37.000 0.500 2.000 2.000 34.000,44.937 68.344 0.000 0.000 -127.500 0.000 34.000 20.500 9.811 4.000 37.000 0.500 2.000 2.000 34.000,45.568 81.009 0.000 0.000 -127.500 0.000 34.000 20.500 9.811 4.000 37.000 0.500 2.000 2.000 34.000,46.845 89.874 42.500 -15.938 -95.625 -15.938 34.000 20.500 9.811 4.000 37.000 0.500 2.000 2.000 34.000,48.107 95.568 37.188 47.812 -111.562 -15.938 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,48.738 97.461 63.750 -21.250 -95.625 -63.750 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,48.738 100.000 47.812 -106.250 -79.688 -31.875 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,49.369 100.000 47.812 -106.250 -79.688 -31.875 33.000 21.000 9.522 5.000 37.000 1.000 0.000 2.000 33.000,50.631 99.369 -37.188 -26.562 -15.938 -111.562 15.000 19.800 12.216 6.000 37.000 2.000 -0.500 4.000 15.000,52.539 98.722 -37.188 -26.562 -15.938 -111.562 15.000 19.800 12.216 6.000 37.000 2.000 -0.500 4.000 15.000,56.341 97.461 -26.562 -79.688 -63.750 -95.625 4.000 34.500 1.118 33.000 36.000 11.500 0.000 2.000 4.000,61.404 94.306 -42.500 0.000 -31.875 -95.625 5.000 34.000 1.414 32.000 36.000 -0.500 -0.500 2.000 5.000,68.360 91.136 -10.625 -31.875 -31.875 -95.625 4.000 32.500 1.118 31.000 34.000 -0.500 -0.500 2.000 4.000,75.315 87.334 31.875 31.875 0.000 -95.625 4.000 31.500 1.118 30.000 33.000 -0.500 -0.500 2.000 4.000,80.379 86.073 15.938 -21.250 -47.812 -127.500 4.000 30.500 1.118 29.000 32.000 -0.500 -0.500 2.000 4.000,82.287 84.811 -63.750 -42.500 -63.750 -95.625 4.000 30.500 1.118 29.000 32.000 0.000 0.000 2.000 4.000,83.549 84.811 -63.750 -42.500 -63.750 -95.625 4.000 30.500 1.118 29.000 32.000 0.000 0.000 2.000 4.000,83.549 84.180 -47.812 85.000 -79.688 -63.750 4.000 30.500 1.118 29.000 32.000 0.000 0.000 2.000 4.000,82.918 84.180 -47.812 85.000 -79.688 -63.750 4.000 30.500 1.118 29.000 32.000 0.000 0.000 2.000 4.000,82.918 83.533 -47.812 85.000 -79.688 -63.750 4.000 30.500 1.118 29.000 32.000 0.000 0.000 2.000 4.000,82.287 83.533 -47.812 85.000 -79.688 -63.750 4.000 30.500 1.118 29.000 32.000 0.000 0.000 2.000 4.000", "q_1_1"


