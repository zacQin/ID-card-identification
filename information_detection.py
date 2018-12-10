import cv2
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pandas as pd

img_2value = io.imread('img.jpg')


plt.hist(img_2value.ravel(),256,[0,256]); plt.show()

img = img_2value
io.imshow(img_2value)

# ret, img_2value = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
ret, img_2value = cv2.threshold(img_2value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.figure('img_2value')
io.imshow(img_2value)

thesum = np.sum(img_2value,axis=1)
thesum = thesum/img_2value.shape[1]

plt.figure('map2x')
plt.plot(thesum)

# df = pd.DataFrame(thesum)
# writer = pd.ExcelWriter('Save_Excel.xlsx')
# df.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
# writer.save()

point = []
for i in range(1,len(thesum)-1):
    if thesum[i]<thesum[i+1] and thesum[i]<thesum[i-1]:
        point.append(thesum[i])

themin = np.where(thesum == min(point))
themin = themin[0][0]

flag_x = 0
flag_y = 0

x = themin-1
y = themin+1


while flag_x == 0:
    if thesum[x] >= (max(thesum)-thesum[themin])/7*6+thesum[themin]:
        leftpoint = x
        flag_x+=1
    else:
        x-=1

while flag_y == 0:
    if thesum[y] >= (max(thesum)-thesum[themin])/7*6+thesum[themin]:
        rightpoint = y
        flag_y+=1
    else:
        y+=1



idnumber = img[ leftpoint:rightpoint,:]
cv2.imwrite('id_number.jpeg',idnumber)
io.imshow(idnumber)
########################################################################

img_2value[ leftpoint:rightpoint,:] = 255
io.imshow(img_2value)


thesum_na = np.sum(img_2value,axis=0)
thesum_na = thesum_na/img_2value.shape[1]

# plt.figure('map2x')
# plt.plot(thesum_na)

point = []
for i in range(1,len(thesum_na)-1):
    if thesum_na[i]<thesum_na[i+1] and thesum_na[i]<thesum_na[i-1]:
        point.append(thesum_na[i])

themin_na = np.where(thesum_na == min(point))
themin_na = themin_na[0][0]
print()
print(  (max(thesum_na) - thesum_na[themin_na])/3*2 + thesum_na[themin_na]  )

flag_x = 0
flag_y = 0

x = themin_na-1
y = themin_na+1


while flag_x == 0:
    if thesum_na[x] >= (max(thesum_na)-thesum_na[themin_na])/7*6 + thesum_na[themin_na]:
        leftpoint_na = x
        flag_x+=1
    else:
        x-=1


nameaddress = img_2value[:, 0:leftpoint_na]
io.imshow(nameaddress)
#######################################################################################

thesum_ad = np.sum(nameaddress,axis=1)
thesum_ad = thesum_ad/nameaddress.shape[0]

# plt.figure('map2x')
# plt.plot(thesum_ad)

point_ad = []

for i in range(1,len(thesum_ad)-1):
    if thesum_ad[i] < thesum_ad [i+1] and thesum_ad [i] < thesum_ad [i-1]:
        point_ad.append ( thesum_ad [i] )

themin_ad = np.where(thesum_ad == min(point_ad))
themin_ad = themin_ad[0][0]

print()
print(  (max(thesum_ad) - thesum_ad[themin_ad]) / 7 * 6 + thesum_ad [themin_ad]  )

flag_x = 0
flag_y = 0

x = themin_ad-1
y = themin_ad+1


while flag_x == 0:
    if thesum_ad [x] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        leftpoint_ad = x
        flag_x+=1
    else:
        x-=1

while flag_y == 0:
    if thesum_ad [y] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        rightpoint_ad = y
        flag_y+=1
    else:
        y+=1

address1 = nameaddress [leftpoint_ad:rightpoint_ad,:]
cv2.imwrite('address1.jpeg',address1)
# io.imshow(address1)
nameaddress [leftpoint_ad:rightpoint_ad,:] = 255

#######################################################################################

thesum_ad = np.sum(nameaddress,axis=1)
thesum_ad = thesum_ad/nameaddress.shape[0]

# plt.figure('map2x')
# plt.plot(thesum_ad)

point_ad = []

for i in range(1,len(thesum_ad)-1):
    if thesum_ad[i] < thesum_ad [i+1] and thesum_ad [i] < thesum_ad [i-1]:
        point_ad.append ( thesum_ad [i] )

themin_ad = np.where(thesum_ad == min(point_ad))
themin_ad = themin_ad[0][0]

print()
print(  (max(thesum_ad) - thesum_ad[themin_ad]) / 7 * 6 + thesum_ad [themin_ad]  )

flag_x = 0
flag_y = 0

x = themin_ad-1
y = themin_ad+1


while flag_x == 0:
    if thesum_ad [x] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        leftpoint_ad = x
        flag_x+=1
    else:
        x-=1

while flag_y == 0:
    if thesum_ad [y] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        rightpoint_ad = y
        flag_y+=1
    else:
        y+=1

address2 = nameaddress [leftpoint_ad:rightpoint_ad,:]
io.imshow(address2)
cv2.imwrite('address2.jpeg',address2)
nameaddress [leftpoint_ad:rightpoint_ad,:] = 255
# io.imshow(address2)
##############################################################################################

thesum_ad = np.sum(nameaddress,axis=1)
thesum_ad = thesum_ad/nameaddress.shape[0]

# plt.figure('map2x')
# plt.plot(thesum_ad)

point_ad = []

for i in range(1,len(thesum_ad)-1):
    if thesum_ad[i] < thesum_ad [i+1] and thesum_ad [i] < thesum_ad [i-1]:
        point_ad.append ( thesum_ad [i] )

themin_ad = np.where(thesum_ad == min(point_ad))
themin_ad = themin_ad[0][0]

print()
print(  (max(thesum_ad) - thesum_ad[themin_ad]) / 7 * 6 + thesum_ad [themin_ad]  )

flag_x = 0
flag_y = 0

x = themin_ad-1
y = themin_ad+1


while flag_x == 0:
    if thesum_ad [x] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        leftpoint_ad = x
        flag_x+=1
    else:
        x-=1

while flag_y == 0:
    if thesum_ad [y] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        rightpoint_ad = y
        flag_y+=1
    else:
        y

birthday = nameaddress [leftpoint_ad:rightpoint_ad,:]
cv2.imwrite('birthday.jpeg',birthday)
io.imshow(birthday)
nameaddress [leftpoint_ad:rightpoint_ad,:] = 255
# io.imshow(nameaddress)
#############################################################################################
thesum_ad = np.sum(nameaddress,axis=1)
thesum_ad = thesum_ad/nameaddress.shape[0]

# plt.figure('map2x')
# plt.plot(thesum_ad)

point_ad = []

for i in range(1,len(thesum_ad)-1):
    if thesum_ad[i] < thesum_ad [i+1] and thesum_ad [i] < thesum_ad [i-1]:
        point_ad.append ( thesum_ad [i] )

themin_ad = np.where(thesum_ad == min(point_ad))
themin_ad = themin_ad[0][0]

print()
print(  (max(thesum_ad) - thesum_ad[themin_ad]) / 7 * 6 + thesum_ad [themin_ad]  )

flag_x = 0
flag_y = 0

x = themin_ad-1
y = themin_ad+1


while flag_x == 0:
    if thesum_ad [x] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        leftpoint_ad = x
        flag_x+=1
    else:
        x-=1

while flag_y == 0:
    if thesum_ad [y] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        rightpoint_ad = y
        flag_y+=1
    else:
        y+=1

name = nameaddress [leftpoint_ad:rightpoint_ad,:]
cv2.imwrite('name.jpeg',name)
io.imshow(name)
nameaddress [leftpoint_ad:rightpoint_ad,:] = 255
# io.imshow(nameaddress)
##################################################################################################
thesum_ad = np.sum(nameaddress,axis=1)
thesum_ad = thesum_ad/nameaddress.shape[0]

# plt.figure('map2x')
# plt.plot(thesum_ad)

point_ad = []

for i in range(1,len(thesum_ad)-1):
    if thesum_ad[i] < thesum_ad [i+1] and thesum_ad [i] < thesum_ad [i-1]:
        point_ad.append ( thesum_ad [i] )

themin_ad = np.where(thesum_ad == min(point_ad))
themin_ad = themin_ad[0][0]

print()
print(  (max(thesum_ad) - thesum_ad[themin_ad]) / 7 * 6 + thesum_ad [themin_ad]  )

flag_x = 0
flag_y = 0

x = themin_ad-1
y = themin_ad+1


while flag_x == 0:
    if thesum_ad [x] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        leftpoint_ad = x
        flag_x+=1
    else:
        x-=1

while flag_y == 0:
    if thesum_ad [y] >= ( max (thesum_ad) - thesum_ad [themin_ad]) / 7 * 6 + thesum_ad [themin_ad]:
        rightpoint_ad = y
        flag_y+=1
    else:
        y+=1

nation_sex = nameaddress [leftpoint_ad:rightpoint_ad,:]
cv2.imwrite('nation_sex.jpeg',nation_sex)
io.imshow(nation_sex)
nameaddress [leftpoint_ad:rightpoint_ad,:] = 255
# io.imshow(nameaddress)











print()