
import numpy as np
from scipy import ndimage
import multiprocessing as mp
from tqdm import tqdm
class voxelSpace:
    def __init__(self, voxelDim,lengthDim,r0=10,r=10):
        self.voxelDim = voxelDim
        self.lengthDim = lengthDim
        self.positionMatrix = self.buildPositionMatrix()
        self.r0 = r0
        self.r = r

    def buildPositionMatrix(self):
        lenX,lenY,lenX = self.lengthDim
        voxX,voxY,voxZ = self.voxelDim
        xCenter = lenX/2
        yCenter = lenY/2
        zCenter = lenZ/2

        x = np.linspace(0, lenX, voxX)
        y = np.linspace(0, lenY, voxY)
        z = np.linspace(0, lenZ, voxZ)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        coor = np.array([xx - x_center, yy - y_center, zz - z_center,0])
        return coor

    def buildSpherePhantom(self,r0):
        coor = self.positionMatrix
        temp1 = np.sum((coor-self.r0)**2,axis=0)
        temp2 = coor[:,temp1<self.r**2][0]
        coor[:,temp1<self.r**2]+=np.array([np.ones(temp2.shape),np.zeros(temp2.shape),np.zeros(temp2.shape),np.zeros(temp2.shape)])
        return coor

    def rotate(self, theta):
        rot = self.rotationMatrix(theta)
        out = np.einsum('ij,kj->ki',rot,self.r0)
        return out
        
    def rotationMatrix(self,angle):
        rad = np.radians(angle)
        sin = np.sin(rad)
        cos = np.cos(rad)
        rot = [[cos, -sin, 0],
               [sin,  cos, 0],
               [0, 0, 1]]
        return np.array(rot)
