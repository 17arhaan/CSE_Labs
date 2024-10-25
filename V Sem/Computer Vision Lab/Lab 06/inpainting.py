import cv2
import numpy as np
def inpainting(image, mask_path):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    inpainted_image_telea = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    inpainted_image_ns = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    return inpainted_image_telea, inpainted_image_ns
image = cv2.imread('inpainting.png')
mask_path = 'mask.png'
inpainted_telea, inpainted_ns = inpainting(image, mask_path)
cv2.imshow("Inpainted (Telea)", inpainted_telea)
cv2.imshow("Inpainted (Navier-Stokes)", inpainted_ns)
cv2.waitKey(0)
cv2.destroyAllWindows()