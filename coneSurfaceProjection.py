import numpy as np
from scipy import ndimage
import multiprocessing as mp
from tqdm import tqdm
class voxelSpace:
    def __init__(self, dim,lengthDim,r0=10,r=10):
        self.voxelDim = dim
        self.lengthDim = lengthDim
        self.positionMatrix = []
        self.positionMatrixExists = False
        self.spherePhantom = False
        self.r0 = r0
        self.r = r
        self.slice = []
        self.xMin = None
        self.xMax = None
        self.yMin = None
        self.yMax = None
        self.zMin = None
        self.zMax = None
        self.maskMatrixExists = False
        self.maskMatrix = []
    def getPosition(self,voxelIndex):
        i,j,k= self.to3D(voxelIndex)
        dX,dY,dZ = self.voxelDim
        return np.array([   (i+0.5)*self.lengthDim[0]/dX-self.lengthDim[0]/2,\
                            (j+0.5)*self.lengthDim[1]/dY-self.lengthDim[1]/2,\
                            (k+0.5)*self.lengthDim[2]/dZ])

    def getVoxelIndex(self, position):

        return

    def to1D(self, coordinates):
      x,y,z=coordinates
      xMax,yMax,zMax=self.voxelDim
      return (z * xMax * yMax) + (y * xMax) + x

    def to3D(self, index):
        xMax,yMax,zMax=self.voxelDim
        z = index // (xMax * yMax)
        index -= (z * xMax * yMax)
        y = index // xMax
        x = index % xMax
        return [x, y, z]

    def getValue(self, voxelIndex):
        if not self.spherePhantom:
            return 0
        elif positionMatrixExists:
            return self.positionMatrix[self.to3D(voxelIndex)]
        else:
            x,y,z = self.getPosition(voxelIndex)
            if n(x-self.r0[0])**2+(y-self.r0[1])**2 + (z-self.r0[2])**2 <r**2:
                return 1
            else:
                return 0
        return 0
    def initiateSpherePhantom(self, r0,r):
        if self.spherePhantom:
            return
        else:
            self.spherePhantom = True
            self.r = r
            self.r0 = r0
            return

    def buildPositionMatrix(self):
        if self.positionMatrixExists:
            return
        else:
            self.positionMatrixExists = True
            dX,dY,dZ = self.voxelDim
            self.positionMatrix= [(np.array([ (i+0.5)*self.lengthDim[0]/dX-self.lengthDim[0]/2,\
                                (j+0.5)*self.lengthDim[1]/dY-self.lengthDim[1]/2,\
                                (k+0.5)*self.lengthDim[2]/dZ]))\
                       for k in range(dX) for j in range(dY) for i in range(dZ)]
            return

    def buildMaskMatrix(self):
        if self.maskMatrixExists:
            return
        else:
            self.maskMatrixExists = True
            dX,dY,dZ = self.voxelDim
            maskMatrix = np.zeros((dX,dY,dZ))
            for i in range(dX):
                for j in range(dY):
                    for k in range(dZ):
                        n = self.to1D([i,j,k])
                        position = self.getPosition(n)
                        if np.dot(position-self.r0, position-self.r0)<self.r**2:
                            maskMatrix[i,j,k] = 1

            self.maskMatrix = maskMatrix

    def showCentralSlice(self):
        if self.spherePhantom:
            image = self.slice
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
            ax1.set_xlabel("y")
            ax1.set_ylabel("x")
            ax1.set_title("Slice")
            ax1.imshow(
                image,
                cmap=plt.cm.Greys_r,
                aspect='auto'#,
                #extent =[-49.5,49.5, -49.5, 49.5]
                    )
    def rotate(self, theta):
        if not self.maskMatrixExists:
            self.buildMaskMatrix()
        out = ndimage.rotate(self.maskMatrix, theta, reshape=False)
        out[out<0.5] = 0.0
        out[out>0.5] = 1.0
        return out



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
        spaceTheta = self.rotate(theta)
        projection = np.zeros(self.numDet)
        indices = np.where(spaceTheta == 1)
        for detIndex, detPosition in enumerate(self.posDet):
            for i,j,k in zip(indices[0],indices[1],indices[2]):
                n = self.to1D([i,j,k])
                position = self.getPosition(n)
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
