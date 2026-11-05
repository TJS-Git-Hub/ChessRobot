"""Optional measured lens correction, including hand-region coordinate mapping."""
from dataclasses import replace
import cv2
import numpy as np


class ImageGeometry:
    def __init__(self,calibration):
        self.calibration=calibration
        self.maps=None
        self.image_space='undistorted' if calibration.camera_matrix else 'camera_uncorrected'

    def apply(self,image):
        cal=self.calibration
        if not cal.camera_matrix:return image
        size=image.shape[1],image.shape[0]
        if size!=tuple(cal.image_size):raise ValueError('Lens calibration resolution differs from input')
        if self.maps is None:
            k=np.asarray(cal.camera_matrix,float)
            self.maps=cv2.initUndistortRectifyMap(k,np.asarray(cal.distortion,float),None,k,size,cv2.CV_32FC1)
        return cv2.remap(image,*self.maps,cv2.INTER_LINEAR)

    def hands(self,record,shape):
        cal=self.calibration
        if record is None or record.regions is None or not cal.camera_matrix:return record
        height,width=shape[:2];k=np.asarray(cal.camera_matrix,float)
        regions=[]
        for points in record.regions:
            pixels=np.asarray(points,np.float32)*[width,height]
            corrected=cv2.undistortPoints(pixels.reshape(-1,1,2),k,np.asarray(cal.distortion,float),P=k).reshape(-1,2)
            regions.append(tuple(map(tuple,corrected/[width,height])))
        return replace(record,regions=tuple(regions))
