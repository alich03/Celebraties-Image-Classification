class Mycls:
    
    def Prepro(path='an.jpg'):
            import pywt
            import cv2
            import numpy as np
            

            face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")

            image_size=50
            image=cv2.imread(path)
            if image is not None:
                grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                faces=face_cascade.detectMultiScale(image,1.2,2)
                
                for x,y,w,h in faces:
                    roi_color=image[y:y+h,x:x+w]
                    roi_grey=grey[y:y+h,x:x+w]
                    eyes=eye_cascade.detectMultiScale(roi_grey,1.2,2)
                    if len(eyes) >= 2:
                        img=roi_color
                        imArray = img
                        #Datatype conversions
                        #convert to grayscale
                        imArray = cv2.cvtColor( imArray,cv2.COLOR_BGR2GRAY )
                        #convert to float
                        imArray =  np.float32(imArray)   
                        imArray /= 255
                        # compute coefficients 
                        coeffs=pywt.wavedec2(imArray,'haar', level=5)

                        #Process Coefficients
                        coeffs_H=list(coeffs)  
                        coeffs_H[0] *= 0;  

                        # reconstruction
                        imArray_H=pywt.waverec2(coeffs_H, 'haar')
                        imArray_H *= 255
                        imArray_H =  np.uint8(imArray_H)
                        img=imArray_H


                        img=cv2.resize(img, (image_size,image_size))
                        roi_color=cv2.resize(roi_color, (image_size,image_size))

                        img=img.reshape(1,image_size*image_size)
                        roi_color= roi_color.reshape(1,image_size*image_size*roi_color.shape[2])

                        img=np.hstack((roi_color,img))

                    
                        return img
            
                    else:
                        return 'Face not clear'
            else:
                return 'Imgae not loaded'
