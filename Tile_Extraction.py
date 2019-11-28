
import cv2
import copy
import numpy as np
from cv2 import cv as cv


class CaptureFrame:
   
   def __init__(self, videofile):      
       
       print ("Tile Extractor constructor...")       
       self.capture = cv2.VideoCapture(videofile)
       self.cnt = 0
       self.ret,self.tile = self.capture.read() # First Frame
       self.fps = self.capture.get(cv2.cv.CV_CAP_PROP_FPS)
       self.firstframe = 1
       self.outflag = 0
       self.MaxAreaTile = 0
       #self.ret = 1
       #self.tile = cv2.imread(videofile)
       print self.ret
       
   
   def ExtractTile(self):                    
       
       height, width, depth = self.tile.shape
       
       # Contour on Frame Boundary       
       img = np.zeros((height,width,3), np.uint8)
       cv2.rectangle(img,(0,0),(width,height),(0,0,255),2)       
       contour_image = cv2.cvtColor(img,cv2.cv.CV_BGR2GRAY)
       contours_frame, hierarchy = cv2.findContours(contour_image,cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE)      
       #cv2.drawContours(img, contours_frame, -1, (0, 255, 0), 3 )
       #cv2.imshow("Contour_Frame",img)       
       
       #Contour on Tiles     
       grey_image = cv2.cvtColor(self.tile,cv2.cv.CV_BGR2GRAY)                  
       thresh =cv2.threshold(grey_image,0,255,cv2.cv.CV_THRESH_BINARY+cv2.cv.CV_THRESH_OTSU)            
       canny_image = cv2.Canny(grey_image, thresh[0], thresh[0]*3)       
       contours, hierarchy = cv2.findContours(canny_image,cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_NONE)       
       length = len(contours) # check for tile-less frames          
       if length :
          # Find the index of the largest contour
          areas = [cv2.contourArea(c) for c in contours]
          #areas = [cv2.arcLength(c,True) for c in contours]              
          max_index = np.argmax(areas)          
          #cv2.drawContours(self.tile,contours[max_index], -1, (255, 0, 0), 3 )
          #cv2.imshow("Contour_Approx", self.tile)
          # To find indices of rotated rectangle
          bound_points = cv2.minAreaRect(contours[max_index])              
          #print (type(bound_points)) -->((c0,c1),(w,h),theta)
          #print bound_points[0][0],bound_points[0][1],bound_points[1][0],bound_points[1][1]
          #print bound_points[2]
          c0 = bound_points[0][0]
          c1 = bound_points[0][1]
          w = bound_points[1][0]
          h = bound_points[1][1]
          theta =(bound_points[2])          
        
          # Tile Boundness - In Frame - Test
          self.boundnessflag = 1
          self.areaflag = 1
          # Test For maxtile area

          if (self.firstframe == True):
             self.MaxAreaTile = areas[max_index]
             self.firstframe = 0
          else :
             if (areas[max_index]<self.MaxAreaTile):
                self.areaflag = 0
             else:
                self.areaflag = 1
                self.MaxAreaTile = areas[max_index]
                
          # Test Using boundingRect      
          x_br,y_br,w_br,h_br = cv2.boundingRect(contours[max_index])
          #print x_br,y_br,w_br,h_br
          ret_val_1 = cv2.pointPolygonTest(contours_frame[0], (x_br,y_br), False)
          ret_val_2 = cv2.pointPolygonTest(contours_frame[0], (x_br,(y_br+w_br)), False)
          ret_val_3 = cv2.pointPolygonTest(contours_frame[0], ((x_br+h_br),y_br), False)
          ret_val_4 = cv2.pointPolygonTest(contours_frame[0], ((x_br+h_br),(y_br+w_br)), False)
          #print ret_val_1,ret_val_2,ret_val_3,ret_val_4
          if((ret_val_1==0) or (ret_val_2==0) or (ret_val_3==0) or (ret_val_4==0)):
             self.boundnessflag=0       
        
          #cv2.rectangle(tile,(x_br,y_br),(x_br+w_br,y_br+h_br),(255,0,0),2)
          #cv2.imshow("Boundrect", tile)
          # Test Using approxPolyDP        
          """
          approx = cv2.approxPolyDP(contours[max_index],0.1*cv2.arcLength(contours[max_index],True),True)
          #cv2.drawContours(tile, approx, -1, (255, 0, 0), 3 )
          cv2.imshow("Contour_Approx", tile)
          out = range(len(approx))              
          print len(approx),type(approx),approx
          for i in range(len(approx)):
             a,b =  approx[i][0]
             print a , b
             out[i] = cv2.pointPolygonTest(contours_frame[0],(a,b) , False)
             print out[i]
             if (out[i] == False):
                flag = 0
                break
          """         
          # Create output image              
          #if(c0 and c1 and w and h):
          if(self.boundnessflag and self.areaflag):              
              rot_mat = cv2.getRotationMatrix2D((bound_points[0][0],bound_points[0][1]), theta, 1)
              #print(type(rot_mat),rot_mat)          
              rotated = cv2.warpAffine(self.tile,rot_mat,(width,height))              
              #print rotated
              self.final = cv2.getRectSubPix(rotated,((int)(w),(int)(h)),(c0,c1))
              self.outflag = 1
              #cv2.imshow("Final", self.final)
          else :
             self.outflag = 0
              
              
# For integration to CTIF  
                   
def loadInfo():
        theTileExtractor.TileExtractor.append(CaptureFrame)
     
# Stand-Alone application           
"""     
if __name__=="__main__":

    frame = CaptureFrame()
    frame.capture = cv2.VideoCapture("v2.avi")
    frame.cnt = 0
    ret,frame.tile = frame.capture.read() # First Frame
    fps = frame.capture.get(cv2.cv.CV_CAP_PROP_FPS)
    while ret:
       frame.ExtractTile()
       if (frame.flag):
         OutTile = copy.copy(frame.final)
         cv2.imwrite('InputTile{}.jpg'.format(frame.cnt), frame.tile)
         cv2.imwrite('OutTile{}.jpg'.format(frame.cnt),OutTile)        
       cv2.waitKey(int((1/fps)*1000)) #To display in FPS
       #cv2.waitKey(1000)
       ret,frame.tile = frame.capture.read() # Read further frames   
       frame.cnt = frame.cnt+1
       if frame.cnt==200: # Temporary code to save only 50 files
         frame.cnt = 0
    #cv2.destroyAllWindows()
"""    
