
import numpy as np
from scipy import ndimage
import multiprocessing as mp
from tqdm import tqdm
class voxelSpace:
    def __init__(self, voxelDim,lengthDim,r0=10,r=10):
        self.voxelDim = voxelDim
        self.lengthDim = lengthDim
        self.r0 = r0
        self.r = r
        self.positionMatrix = self.buildPositionMatrix()


    def buildPositionMatrix(self):
        lenX,lenY,lenZ = self.lengthDim
        voxX,voxY,voxZ = self.voxelDim
        xCenter = lenX/2
        yCenter = lenY/2
        zCenter = lenZ/2

        x = np.linspace(0, lenX, voxX)
        y = np.linspace(0, lenY, voxY)
        z = np.linspace(0, lenZ, voxZ)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        coor = np.array([np.zeros(xx.shape), xx - xCenter, yy - yCenter, zz - zCenter])
        temp0 = np.zeros((4,1,1,1))
        temp0[:,0,0,0] = [0, self.r0[0],self.r0[1],self.r0[2]]
        temp1 = np.sum((coor-temp0)**2,axis=0)
        temp2 = coor[:,temp1<self.r**2][0]
        coor[:,temp1<self.r**2]+=np.array([np.ones(temp2.shape)])
        return coor

    def buildSpherePhantom(self,r0P):
        coor = self.positionMatrix
        temp0 = np.zeros((4,1,1,1))
        temp0[:,0,0,0] = [0, r0P[0],r0P[1],r0P[2]]
        temp1 = np.sum((coor-temp0)**2,axis=0)
        temp2 = coor[:,temp1<self.r**2][0]
        coor[:,temp1<self.r**2]+=np.array([np.ones(temp2.shape)])
        return coor

    def rotate(self, theta):
        rot = self.rotationMatrix(theta)
        out = np.einsum('ij,j',rot,np.array(self.r0).T)
        return out

    def rotationMatrix(self,angle):
        rad = np.radians(angle)
        sin = np.sin(rad)
        cos = np.cos(rad)
        rot = [[cos, -sin, 0],
               [sin,  cos, 0],
               [0, 0, 1]]
        return np.array(rot)

class detectorSpace(voxelSpace):
    def __init__(self,dim,lengthDim,r0,r,beta,phi,numDet,posDet ):
        super().__init__(dim,lengthDim,r0,r)
        self.beta = beta
        self.phi = phi
        self.numDet = numDet
        self.posDet = posDet


    def g(self,x,r):
        v = x-r
        norm_v = np.linalg.norm(v)
        return np.sign(np.dot(v/norm_v,self.beta)-np.cos(self.phi))

    def I(self,vj,r):
        dx,dy,dz = np.array([self.lengthDim[0]/self.voxelDim[0],\
                             self.lengthDim[1]/self.voxelDim[1],\
                             self.lengthDim[2]/self.voxelDim[2]   ])
        #Corners
        w1 = self.g(vj + np.array([dx,dy,dz]),r)
        w2 = self.g(vj + np.array([-dx, dy,dz]),r)
        w3 = self.g(vj + np.array([dx,-dy,dz]),r)
        w4 = self.g(vj + np.array([-dx,-dy,dz]),r)
        w5 = self.g(vj + np.array([dx,dy,-dz]),r)
        w6 = self.g(vj + np.array([-dx,dy,-dz]),r)
        w7 = self.g(vj + np.array([dx,-dy,-dz]),r)
        w8 = self.g(vj + np.array([-dx,-dy,-dz]),r)

        return not (np.abs(w1+w2+w3+w4+w5+w6+w7+w8) == \
                    np.abs(w1)+np.abs(w2)+np.abs(w3)+np.abs(w4)+np.abs(w5)+np.abs(w6)+np.abs(w7)+np.abs(w8)).all()

    def coneSurfaceProjection(self,i, theta, q):
        """
        This is where the cone-surface projections are calculated
        """
        spaceTheta = self.buildSpherePhantom(self.rotate(theta))
        projection = np.zeros(self.numDet)
        indices = np.where(spaceTheta[0,:,:,:] == 1)
        for detIndex, detPosition in enumerate(self.posDet):
            for i,j,k in zip(indices[0],indices[1],indices[2]):
                position = spaceTheta[:,i,j,k][1:]
                if self.I(position,detPosition):
                    projection[detIndex] += 1.0

        q.put((i,projection))

    def getConeSurfaceProjections(self, inputValues):
        qOut = mp.Queue()
        processes = [mp.Process(target=self.coneSurfaceProjection, args=(ind, val, qOut)) for ind, val in enumerate(inputValues)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        unsortedResult = [qOut.get() for p in processes]
        sortedResult = [t[1] for t in unsortedResult]
        return sortedResult
