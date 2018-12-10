import cv2
import smooth_sharpen as ss
import pytesseract
from PIL import Image


show = cv2.imread('img.jpg')
# show = ss.sharpen(show)
# show = ss.smooth(show)
# show = ss.sharpen(show)
# show = ss.smooth(show)
# show = ss.sharpen(show)
# show = ss.smooth(show)
# show = ss.sharpen(show)
# show = ss.smooth(show)
# show = ss.sharpen(show)
# show = ss.smooth(show)
# show = ss.sharpen(show)

# cv2.imwrite('image/lala.png',show)
# im = Image.open('image/6273713-969204f566394e4c.png')
#  Image.open('lala.png'),lang='chi_sim'

text = pytesseract.image_to_string(show,lang='chi_sim')
print(text)

print()