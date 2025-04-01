import numpy as np
import cv2 as cv
import base64
from io import BytesIO
from PIL import Image

class DisplayTumor:
    curImg = 0
    Img = 0
    kernel = np.ones((3, 3), np.uint8)  # Inicializamos kernel aquí

    def readImage(self, img):
        self.Img = np.array(img)
        self.curImg = np.array(img)
        gray = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)  # Convertimos a escala de grises
        self.ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # Binarizamos la imagen

    def getImage(self):
        return self.curImg

    # noise removal
    def removeNoise(self):
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        self.curImg = opening

    def displayTumor(self):
        # Asegurémonos de que la imagen esté en escala de grises y binarizada
        if len(self.curImg.shape) == 3:  # Si la imagen tiene 3 canales (RGB), convertirla a escala de grises
            gray = cv.cvtColor(self.curImg, cv.COLOR_BGR2GRAY)
        else:
            gray = self.curImg

        # Binario: Hacer seguro que la imagen es binaria
        _, binarized = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

        # sure background area
        sure_bg = cv.dilate(binarized, self.kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(binarized, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv.watershed(self.Img, markers)
        self.Img[markers == -1] = [255, 0, 0]

        tumorImage = cv.cvtColor(self.Img, cv.COLOR_HSV2BGR)
        self.curImg = tumorImage

    def get_base64_image(self):
        """Convierte la imagen procesada a base64"""
        # Convertimos la imagen a formato PIL para manejarla mejor
        pil_image = Image.fromarray(self.curImg)
        
        # Guardamos la imagen en un buffer de memoria
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        
        # Codificamos la imagen a base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str
