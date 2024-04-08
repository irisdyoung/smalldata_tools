import numpy as np
import smalldata_tools.utilities as smd_utils
from skbeam.core.stats import statistics_1D
from skbeam.core import roi, utils
import pickle, os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpiSize = comm.Get_size()

from smalldata_tools.DetObjectFunc import DetObjectFunc

class radialAverageFunc(DetObjectFunc):
    """
    function to generate a radial average of an image in q
    """
    def __init__(self, **kwargs):
        self._name = kwargs.get('name','rad')
        super(radialAverageFunc, self).__init__(**kwargs)
        self.ctrX, self.ctrY = kwargs.get('center',None) # px
        self.distance = kwargs.get('distance',None) # mm
        #self.eBeam =  kwargs.pop("eBeam",11.2) # TODO: pull from event
        #self.lam = smd_utils.E2lam(self.eBeam)*1e10 if self.eBeam else None
        #self.dmax = kwargs.get('dmax',None) # optional inner radius in Angstroms
        #self.dmin = kwargs.get('dmin',None) # optional outer radius in Angstroms
        self.innerRadius = kwargs.get('innerRadius',None) # optional, px
        self.outerRadius = kwargs.get('outerRadius',None) # optional, px
        self.ringWidth = kwargs.get('ringWidth',1) # binning, px
        self.ringGap = kwargs.get('ringGap',0) # gap between radial bins
        self.numRings = kwargs.get('numRings',None) # number of radial bins
        self.pxSize = kwargs.get('pxSize',None) # mm
        self.mask = kwargs.get('userMask',None)
        if self.mask is not None:
            self.mask = np.asarray(self.mask,dtype=np.bool).flatten()

    def setFromDet(self, det):
        if det.mask is not None and det.cmask is not None:
            if self.mask is not None and self.mask.flatten().shape == det.mask.flatten().shape:
                self.mask = ~(self.mask.flatten().astype(bool)&det.mask.astype(bool).flatten())
            else:
                self.mask = ~(det.cmask.astype(bool)&det.mask.astype(bool))
        self.mask = self.mask.flatten()
        if det.z is not None and self.distance is None:
            self.distance = det.z.flatten() # /1e3 comment out -> keep in mm
        self.imgShape = det.imgShape

        if self.numRings and not self.ringWidth:
            numPix = min(self.ctrX, self.imgShape[0]-self.ctrX,
                         self.ctrY, self.imgShape[1]-self.ctrY)
            self.ringWidth = numPix/self.numRings

        self.edges = roi.ring_edges(self.innerRadius, self.ringWidth, self.ringGap, self.numRings)
        self.rings = roi.rings(self.edges, (self.ctrX, self.ctrY), self.imgShape)
        self.rings_1d = self.rings.flatten()

    def process(self, data):
        stats = statistics_1D(self.rings_1d, data.flatten(), stat='median', nx=self.numRings)
        binEdges, self.medianI = stats
        binCenters = .5*(binEdges[:-1]+binEdges[1:])

        # wavelength independent, distance and pixel size dependent:
        self.twoTheta = utils.radius_to_twotheta(self.distance, self.pxSize*binCenters)
        if len(self.twoTheta) < 2:
            return {'twoTheta':[], 'medianI':[]}
        else:
            return {'twoTheta':self.twoTheta[1:], 'medianI':self.medianI[1:]}

        # wavelength, distance and pixel size dependent:
        #self.qCenters = utils.twotheta_to_q(twoThetaCenters, self.lam)
        #if len(self.qCenters) < 2:
        #    return {'q_centers':[], 'median_I':[]}
        #else:
        #    return {'q_centers':self.qCenters[1:], 'median_I':self.median_I[1:]}

    # alternative approach without flattening:
    #def process(self, data):
    #    radii, average = roi.circular_average(data,
    #                                          calibrated_center=(self.ctrX, self.ctrY),
    #                                          nx=self.numRings,
    #                                          pixel_size=(self.pxSize, self.pxSize),
    #                                          min_x=self.innerRadius,
    #                                          max_x=self.outerRadius,
    #                                          mask=self.mask)
    #    twoThetaArr = utils.radius_to_twotheta(self.distance, radii)
    #    qArr = utils.twotheta_to_q(twoThetaArr, self.lam)
    #    self.dat = {'radialAvgQ':qArr, 'radialAvgI':average}
    #    return {'rad':(qArr, average)}


