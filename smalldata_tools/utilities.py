import os
import json
import copy
import numpy as np
import time
import h5py
import tables
from scipy import optimize
from scipy import ndimage
from scipy import signal
from matplotlib import pyplot as plt
import resource
from itertools import izip, count
import psana
import RegDB.experiment_info

from collections import deque
from itertools import islice
from bisect import insort, bisect_left
 
from numba import jit

import sys

def MAD(a, c=0.6745, axis=None):
  """
  Median Absolute Deviation along given axis of an array:
  
  median(abs(a - median(a))) / c
  
  c = 0.6745 is the constant to convert from MAD to std; it is used by
  default  
  """
  
  a = np.ma.masked_where(a!=a, a)
  if a.ndim == 1:
    d = np.ma.median(a)
    m = np.ma.median(ma.fabs(a - d) / c)
  else:
    d = np.ma.median(a, axis=axis)
    # I don't want the array to change so I have to copy it?
    if axis > 0:
      aswp = np.ma.swapaxes(a,0,axis)
    else:
      aswp = a
      m = np.ma.median(ma.fabs(aswp - d) / c, axis=0)
      
  return m
    
def nanmedian(arr, **kwargs):
      """
      Returns median ignoring NAN
      """
      return np.ma.median( np.ma.masked_where(arr!=arr, arr), **kwargs )
    
def running_median_insort(seq, windowsize=10):
    isArray= isinstance(seq, np.ndarray)
    if isArray:
        seq = seq.tolist()
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, windowsize):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = windowsize // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    if isArray:
        result = np.array(result)
    return result

def running_median(seq, windowsize=10):
    isArray= isinstance(seq, np.ndarray)
    if isArray:
        seq = seq.tolist()
    result=[]
    for ipseq,pseq in enumerate(seq):
        seqMin = max(0,ipseq-windowsize+1)
        result.append(np.median(seq[seqMin:ipseq+1]))
    if isArray:
        result = np.array(result)
    return result
    
def rebinFactor(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def rebin(a, shape):
    if isinstance(shape, float) or isinstance(shape, int):
      shape = [shape, shape]
    if (a.shape[0]%shape[0]) == 0 and (a.shape[1]%shape[1]) == 0:
      rebinFactor(a, shape)
    else:
      factor = [ float(int(a.shape[0]/shape[0])+1), float(int(a.shape[1]/shape[1])+1)]
      bigImg = ndimage.zoom(a, [shape[0]*factor[0]/a.shape[0],shape[1]*factor[1]/a.shape[1]])
      img = rebinFactor(bigImg, shape)
    return img

def rebinShape( a, newshape ):
  '''Rebin an array to a new shape.
  '''
  assert len(a.shape) == len(newshape)

  slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
  coordinates = np.mgrid[slices]
  indices = coordinates.astype('i')   #choose the biggest smaller integer index
  return a[tuple(indices)]


def reduceVar(vals, sigROI,threshold=-1e25):
    if threshold!=-1e25:
      vals = vals[vals<threshold]=0
    print 'array shape: ',len(vals.shape)
    if len(vals.shape)>1 and sigROI!=[]:
      if len(vals.shape)==2:
        if not isinstance(sigROI, list):
          return vals[:,sigROI]
        elif len(sigROI)>1:
          return vals[:,sigROI[0]:sigROI[1]]
        else:
          return vals[:,sigROI[0]]
      elif len(vals.shape)==3:
        if not isinstance(sigROI, list):
          return vals[:,sigROI,:]
        elif len(sigROI)==1:
          return vals[:,sigROI[0],:]
        elif len(sigROI)==2:
          return vals[:,sigROI[0]:sigROI[1],:]
        else:
          return vals[:,sigROI[0]:sigROI[1],sigROI[2]:sigROI[3]]
      elif len(vals.shape)==4:
        if not isinstance(sigROI, list):
          return vals[:,sigROI,:,:]
        elif len(sigROI)==1:
          return vals[:,sigROI[0],:,:]
        elif len(sigROI)==2:
          return vals[:,sigROI[0]:sigROI[1],:,:]
        elif len(sigROI)==4:
          return vals[:,sigROI[0]:sigROI[1],sigROI[2]:sigROI[3],:]
        else:
          return vals[:,sigROI[0]:sigROI[1],sigROI[2]:sigROI[3],sigROI[4]:sigROI[5]]

    print 'this dimension is not yet implemented:',vals.shape,' ROI: ',sigROI
    return vals

def getBins(bindef=[], debug=False):
  #have full list of bin boundaries, just return it
  if len(bindef)>3:
    return bindef
  
  if len(bindef)==3:
    if type(bindef[2]) is int:
      Bins=np.linspace(min(bindef[0],bindef[1]),max(bindef[0],bindef[1]),bindef[2]+1,endpoint=True)
    else:
      Bins=np.arange(min(bindef[0],bindef[1]),max(bindef[0],bindef[1]),bindef[2])
    if Bins[-1]<bindef[1]:
      Bins = np.append(Bins,max(bindef[0],bindef[1]))
    return Bins

  if len(bindef)==2:
    if debug:
      print 'only two bin boundaries, so this is effectively a cut...cube will have a single image'
    Bins = np.array([min(bindef[0],bindef[1]),max(bindef[0],bindef[1])])
    return Bins

  #fallback.
  return None
    
#from ixppy
def E2lam(E,o=0):
    """ Computes photon wavelength in m
        E is photon energy in eV or keV
        set o to 0 if working at sub-100 eV energies
    """
    if o:
      E=E
    else:
      E=eV(E)
    lam=(12398.4/E)*1e-10
    return lam
  #lam = 12.39842 /E
  #return lam

def eV(E):
  if E < 100:
    E=E*1000.0;
    return E*1.0

def checkDet(env, detname):
  for key in env.configStore().keys():
    if key.alias()==detname:
      return True
  for key in env.configStore().keys():
    #print key.src().__repr__(),detname,key.src().__repr__()==detname
    if key.src().__repr__()==detname:
      return True
  return False
        
def printR(rank=0, message=''):
  if rank==0:
    print message

def printMsg(eventNr, run, rank=0, size=1):
  printFreq = 10
  #if eventNr > 10000:
  #  printFreq = 10000
  if eventNr > 1000:
    printFreq = 1000
  elif eventNr > 120:
    printFreq = 100
    
  if eventNr%printFreq == 0:
    if rank == 0:
      usage = resource.getrusage(resource.RUSAGE_SELF)
      print "*** In Event: run", run, ",event# in single job =", eventNr,', total about ',eventNr*size,' memory used: ',usage[2]*resource.getpagesize()/1000000.,' at ',time.strftime('%X')

def getExpName(env):  
  if (env.jobName()).find('shmem')>=0:
    return RegDB.experiment_info.active_experiment('XPP')[1]
  else:
    return env.experiment()

##########################################################################################
###  helper classes & functions
##########################################################################################

def hasKey(inkey, inh5=None, printThis=False):
    hasKey = False
    if inh5 is None:
      print 'no input file given'
      return hasKey
    if not isinstance(inh5, tables.file.File):
      print 'input file is not a tables h5file'
      return hasKey

    try:
      if inkey[0]==['/']:
        inkey=inkey[1:]
      if inkey.find('/')>=0:
        inh5.get_node('/'+inkey.split('/')[0],inkey.split('/')[1])
      else:
        inh5.get_node('/'+inkey)
      hasKey=True
    except:
      pass
    return hasKey

def getVar(fh5, plotvar):
  if not hasKey(plotvar,fh5):
    print 'signal variable %s not in littleData file'%(plotvar)
    return

  if plotvar[0]!=['/']:
    plotvar='/'+plotvar
  try:
    if len(plotvar.split('/'))>2:
      vals = fh5.get_node('/'.join(plotvar.split('/')[:-1]),plotvar.split('/')[-1]).read()
    else:
      vals = fh5.get_node(plotvar).read()
    return vals.squeeze()
  except:
    print 'failed to get data for ',plotvar
    return

def getTTstr(fh5):
  """
  function to determine the string for the timetool variables in the desired run
  necessary as naming scheme evolved over time
  """
  ttCorr = None
  ttBaseStr = 'tt/'
  if hasKey('tt/ttCorr',fh5):
    ttCorr = 'tt/ttCorr'
  elif hasKey('ttCorr/tt',fh5):
    ttCorr = 'ttCorr/tt'
  if not hasKey(ttBaseStr+'AMPL',fh5):
    if hasKey('tt/XPP_TIMETOOL_AMPL',fh5):
      ttBaseStr = 'tt/XPP_TIMETOOL_'
    elif hasKey('tt/TIMETOOL_AMPL',fh5):
      ttBaseStr = 'tt/TIMETOOL_'
    elif hasKey('tt/TTSPEC_AMPL',fh5):
      ttBaseStr = 'tt/TTSPEC_'
  return ttCorr, ttBaseStr

def getDelay(fh5, use_ttCorr=True, addEnc=False):
  """
  function to get the xray-laser delay from the data
  usage:
  getDelay(fh5): get the delay from lxt and/or encoder stage, add the timetool correction
  getDelay(fh5, use_ttCorr=False): get the delay from lxt and/or encoder stage, NO timetool correction
  getDelay(fh5, addEnc=True): get the delay from lxt, add encoder stage and timetool correction
  fh5 is the pointer to the h5py or tables objext
  """
  ttCorrStr, ttBaseStr = getTTstr(fh5)
  if ttCorrStr is not None:
    ttCorr=getVar(fh5,ttCorrStr)
    if (np.nanstd(ttCorr)==0):
      ttCorr=getVar(fh5,ttBaseStr+'FLTPOS_PS')
  nomDelay=np.zeros_like(ttCorr)

  isDaqDelayScan=False
  for node in fh5.get_node('/scan')._f_list_nodes():
    if node.name.find('var')<0 and node.name.find('none')<0 and node.name.find('lxt')>=0 and node.name.find('damage')<0:
      isDaqDelayScan=True
      #print 'DEBUG: found that we have a delay scan'
      nomDelay=node.read()*1e12

  if not isDaqDelayScan:
    if hasKey('enc/lasDelay',fh5):
      encVal = getVar(fh5,'enc/lasDelay')
      #print 'DEBUG: encoder info',encVal.std()
      if encVal.std()>1e-9:
        nomDelay=encVal
        addEnc=False
      elif encVal.std()>1e-15:
        nomDelay=encVal*1e12
        addEnc=False
    elif hasKey('enc/ch0',fh5):
      encVal = getVar(fh5,'enc/ch0')
      if encVal.std()>1e-15 and encVal.std()<1e-9:
        nomDelay=encVal*1e12
        #now look at the EPICS PV if everything else has failed.
    else:
      epics_delay = getVar(fh5,'epics/lxt_ttc')
      if epics_delay.std()!=0:
        nomDelay = epics_delay

  if addEnc and not hasKey('enc/lasDelay',fh5):
    print 'required to add encoder value, did not find encoder!'
  if addEnc and hasKey('enc/lasDelay',fh5):            
    if getVar(fh5,'enc/lasDelay').std()>1e-6:
      nomDelay+=getVar(fh5,'enc/lasDelay').value

  if use_ttCorr:
    #print 'DEBUG adding ttcorr,nomdelay mean,std: ',ttCorr.mean(),nomDelay.mean(),ttCorr.std(),nomDelay.std()
    return (ttCorr+nomDelay)
  else:
    return nomDelay


def addToHdf5(fh5, key, npAr):
    arShape=()
    for i in range(0,len(npAr.shape)):
        arShape+=(npAr.shape[i],)
    dset = fh5.create_dataset(key, arShape)
    dset[...] = npAr.astype(float)


###
# utility functions for getting indices of matching off events
###
def get_startOffIdx(evtTime, filterOff, nNbr=3, nMax=-1):
    ##switch to next set of off events when:
    #t_next - t_evt < t_evt - t_last ----> (t_next + t_last) * 0.5 < t_evt    
    #calculate times where the offIdx_start will switch: (t[offIdx_start+nNbr]+t[offIdx_start])*0.5
    evtTimeOff = evtTime[filterOff]
    t_switch = (evtTimeOff[:-nNbr] - evtTimeOff[nNbr:])/2 + evtTimeOff[nNbr:]
      
    #now digitize times
    all_startOffIdx = np.digitize(evtTime, t_switch)
    #to use
    #datOffNbr[i] = dat_array[filterOff][all_startOffIdx[i]:all_startOffIdx[i]+nNbr] 
    return np.array(all_startOffIdx)

#speed gain factor 2-3 depending on data (better for single#)
@jit
def get_offVar(varArray, filterOff, startOffIdx, nNbr=3, mean=True):
    assert (varArray.shape[0] == filterOff.shape[0])
    assert (varArray[filterOff].shape[0] >= (startOffIdx[-1]+nNbr))
    varOff=[]
    for idx in startOffIdx:
        if mean:
            varOff.append((varArray[filterOff][idx:idx+nNbr]).mean())
        else:
            varOff.append(varArray[filterOff][idx:idx+nNbr])
    return np.array(varOff)
_ = get_offVar(np.arange(5), np.ones(5).astype(int), np.array([0,0,1,1,2]), nNbr=1)

###
# utility functions for droplet stuff
###
def gaussian(x, amp, cen, wid):
                return amp * exp(-(x-cen)**2 /(2*wid**2))
def lorentzian(x,p0,p1):
		return (p0**2)/(p0**2 + (x-p1)**2)

def neighborImg(img):
    img_up = np.roll(img,1,axis=0); img_up[0,:]=0
    img_down = np.roll(img,-1,axis=0); img_down[-1,:]=0
    img_left = np.roll(img,1,axis=1); img_left[:,0]=0
    img_right = np.roll(img,-1,axis=1); img_right[:,-1]=0
    return np.amax(np.array([img_up, img_down, img_left, img_right]),axis=0)

def cm_epix(img,rms,maxCorr=30, histoRange=30, colrow=3, minFrac=0.25, normAll=False, mask=None):
    #make a mask: all pixels > 10 rms & neighbors & pixels out of historange
    imgThres = img.copy()
    imgThres[img>=rms*10]=1
    imgThres[img<rms*10]=0
    imgThres+=neighborImg(imgThres)
    imgThres+=imgThres+(abs(img)>histoRange)
    if mask is not None:
      imgThres+=(mask==0).astype(int)

    maskedImg = np.ma.masked_array(img, imgThres)
    if normAll:
      maskedImg -= maskedImg.mean()
    if colrow%2==1:
        rs = maskedImg.reshape(704/2,768*2,order='F')
        rscount = np.ma.count_masked(rs,axis=0)
        rsmed = np.ma.median(rs,axis=0)
        rsmed[abs(rsmed)>maxCorr]=0
        rsmed[rscount>((1.-minFrac)*352)]=0
        imgCorr = np.ma.masked_array((rs.data-rsmed[None,:]).data.reshape(704,768,order='F'),imgThres)
    else:
        imgCorr = maskedImg.copy()

    if colrow>=2:
        rs = imgCorr.reshape(704*8,96)
        rscount = np.ma.count_masked(rs,axis=1)
        rsmed = np.ma.median(rs,axis=1)
        rsmed[abs(rsmed)>maxCorr]=0
        rsmed[rscount>((1.-minFrac)*96)]=0
        imgCorr = (np.ma.masked_array(rs.data-rsmed[:,None], imgThres)).reshape(704,768)
                                                
    return imgCorr.data  

###
# utility functions for plotting data as 2-d histogram
###
def hist2d(ar1, ar2,limits=[1,99.5],numBins=[100,100],histLims=[np.nan,np.nan, np.nan, np.nan],weights=None, doPlot=True):
        pmin0 = np.nanmin(ar1); pmin1 = np.nanmin(ar2)
        pmax0 = np.nanmax(ar1); pmax1 = np.nanmax(ar2)
        if not np.isnan(np.percentile(ar1,limits[0])):
            pmin0 = np.percentile(ar1,limits[0])
        if not np.isnan(np.percentile(ar2,limits[0])):
            pmin1 = np.percentile(ar2,limits[0])
        if limits[1]<100:
            if not np.isnan(np.percentile(ar1,limits[1])):
                pmax0 = np.percentile(ar1,limits[1])
            if not np.isnan(np.percentile(ar2,limits[1])):
                pmax1 = np.percentile(ar2,limits[1])
        if histLims[0] is not np.nan:
            pmin0 = histLims[0]
            pmax0 = histLims[1]
            pmin1 = histLims[2]
            pmax1 = histLims[3]
        v0 = ar1
        v1 = ar2
        binEdges0 = np.linspace(pmin0, pmax0, numBins[0])
        binEdges1 = np.linspace(pmin1, pmax1, numBins[1])
        ind0 = np.digitize(v0, binEdges0)
        ind1 = np.digitize(v1, binEdges1)
        ind2d = np.ravel_multi_index((ind0, ind1),(binEdges0.shape[0]+1, binEdges1.shape[0]+1)) 
        if weights is None:
                iSig = np.bincount(ind2d, minlength=(binEdges0.shape[0]+1)*(binEdges1.shape[0]+1)).reshape(binEdges0.shape[0]+1, binEdges1.shape[0]+1) 
        else:
                iSig = np.bincount(ind2d, weights=weights, minlength=(binEdges0.shape[0]+1)*(binEdges1.shape[0]+1)).reshape(binEdges0.shape[0]+1, binEdges1.shape[0]+1)    
        if doPlot:
          plt.imshow(iSig,aspect='auto', interpolation='none',origin='lower',extent=[binEdges1[1],binEdges1[-1],binEdges0[1],binEdges0[-1]],clim=[np.percentile(iSig,limits[0]),np.percentile(iSig,limits[1])])
          plt.colorbar()
        return iSig

###
def dictToHdf5(filename, indict):
  f = h5py.File(filename,'w')
  for key in indict.keys():
    npAr = np.array(indict[key])
    dset = f.create_dataset(key, npAr.shape, dtype='f')
    dset[...] = npAr
  f.close()

#
# code for peak finding.
#
def fitTrace(data, kind="stepUp", weights=None, iterFrac=-1., maxPeak=100):
    if data is None or len(data)<10:
        return
    nWeights = 100
    if weights is not None and isinstance(weights, int):
        nWeights = weights
    if weights is None or isinstance(weights, int):
        weights = np.ones(nWeights)
        if kind == "stepUp":
            weights[nWeights/2:]=0
        elif kind == "stepDown":
            weights[0:nWeights/2]=0
        else:
            print 'for fits not using stepUp or stepDown you need to provide weights!'
            return
    weights = np.array(weights).squeeze()
    retDict = {}
    retDict['pos']=0.
    retDict['amp']=0.
    retDict['fwhm']=0.
    weights = weights/sum(weights)     #normalize
    weights = weights-weights.mean()   #center around 0

    nWeights = len(weights)
    halfrange = round(nWeights/10)
    if np.isnan(data).sum()==data.shape[0]:
        return retDict
        
    f0 = np.convolve(np.array(weights).ravel(),data,'same')
    f = f0[nWeights/2:len(f0)-nWeights/2-1]
    retDict['fConv']=f
    mpr = f.argmax()
    # now do a parabolic fit around the max
    xd = np.arange(max(0,mpr-halfrange),min(mpr+halfrange,len(f)-1))
    yd = f[int(max(0,mpr-halfrange)):int(min(mpr+halfrange,len(f)-1))]

    p2 = np.polyfit(xd,yd,2)
    tpos = -p2[1]/2./p2[0]
    tamp = np.polyval(p2,tpos)
    try:
        below = (f<((f[-25:].mean()+tamp)/2.)).nonzero()[0]-mpr
        tfwhm = np.abs(np.min(below[below>0])-np.max(below[below<0]))
        retDict['below']=below
    except:
        tfwhm = 0.
    retDict['pos']=tpos + nWeights/2.
    retDict['amp']=tamp

    if np.isnan(tamp): 
        retDict['fwhm']=np.nan
    else:
        retDict['fwhm']=tfwhm 

    if iterFrac>0.:
        allPos=[retDict['pos']]
        allAmp=[retDict['amp']]
        allFwhm=[retDict['fwhm']]
        fP = f.copy()        

        mprIter = mpr
        currPeakMin = max(0,mprIter-allFwhm[-1])
        currPeakMax = min(mprIter+allFwhm[-1],len(fP))
        print 'zero array: ',currPeakMin, currPeakMax
        fP[currPeakMin:currPeakMax]=[0]*(currPeakMax-currPeakMin)
        while np.max(fP)>iterFrac*allAmp[0]:
            retDict['fp_%d'%len(allPos)]=np.array(fP)#fP.deep_copy()
            mprIter = fP.argmax()
            # now do a parabolic fit around the max
            xd = np.arange(max(0,mprIter-halfrange),min(mprIter+halfrange,len(fP)-1))
            yd = fP[max(0,mprIter-halfrange):min(mprIter+halfrange,len(fP)-1)]
            p2 = np.polyfit(xd,yd,2)
            tpos = -p2[1]/2./p2[0]
            tamp = np.polyval(p2,tpos)
            try:
                below = (fP<((fP[-25:].mean()+tamp)/2.)).nonzero()[0]-mprIter
                try:
                    hm_above = np.min(below[below>0])
                except:
                    hm_above = len(below)
                try:
                    hm_below = np.max(below[below<0])
                except:
                    hm_below=0
                tfwhm = np.abs(hm_above - hm_below)

                currPeakMin = max(0,mprIter-tfwhm)
                currPeakMax = min(mprIter+tfwhm,len(fP))
                #print len(allPos),' new peak vals: ',tpos, tamp, tfwhm, ' zero array: ',currPeakMin, currPeakMax
                fP[currPeakMin:currPeakMax]=[0]*(currPeakMax-currPeakMin)

                allPos.append(tpos + nWeights/2.)
                allAmp.append(tamp)
                if np.isnan(tamp): 
                    allFwhm.append(np.nan)
                    break
                allFwhm.append(tfwhm)
            except:
                break
            if len(allPos)>maxPeak:
                break
        retDict['allPos'] = allPos
        retDict['allAmp'] = allAmp
        retDict['allFwhm'] = allFwhm

    #for k in retDict.keys():
        #print 'fit Trace test: ',k,retDict[k]

    return retDict

###
#simple peak finding in traces (double pulse acqiris data)
###
def findPeakSimple(trace, nMaxPeak=2, peakWid=20, peakOff=5):
  
    peakIdx = signal.find_peaks_cwt(trace, widths=[3])
    maxIdx = [ x for _,x in sorted(zip(trace[peakind],peakIdx), key=lambda pair: pair[0], reverse=True)]
    peakIdxS=[]
    peakIdxV=[]
    peakIdxSum=[]
    for peak in maxIdx:
        if len(peakIdxS)>=nMaxPeak:
            break
        for peakNear in np.arange(peak-peakWid/4, peak+peakWid/4):
            if peakNear in peakIdxS:
                continue
        peakIdxS.append(peak)
        peakIdxV.append(trace[peak])
        peakIdxSum.append(trace[peak-peakWid/2+peakOff:peak+peakWid/2+peakOff].sum())
    retDict={'peakIdx': np.array(peakIdxS)}
    retDict['peakMax'] =  np.array(peakIdxV)
    retDict['peakSum'] =  np.array(peakIdxSum)
    return retDict


#
# should work on removing it
#
class dropObject(object):
  def __init__(self,name='noname',parent=None):
    self._name = name
    self._parent = parent

  def add(self,name,data):
    if (name not in self.__dict__):
      self._add(name,data)
    else:
      self.__dict__[name].append(data)
  def _add(self,name,data):
    self.__dict__[name]=[data]
  def addField(self,name,data):
    if (name not in self.__dict__):
      if isinstance(data, list) or  isinstance(data, np.ndarray):
        self.__dict__[name]=data
      else:
        self.__dict__[name]=[data]
    else:
      print 'field ',name,' already in dropObject: ',self._name
  def __repr__(self):
    return "dropObject with fields: "+str(self.__dict__.keys())
  def __getitem__(self,x):
    return self.__dict__[x]
  def __setitem__(self,name,var,setParent=True):
    self._add(name,var)
    if setParent:
      try:
        self[name]._parent = self
      except:
	pass
  def _get_keys(self):
    return [tk for tk in self.__dict__.keys() if not tk[0]=='_']


def shapeFromKey_h5(fh5, thiskey):
    if len(thiskey.split('/')) > 2:
        return fh5.get_node('/'.join(thiskey.split('/')[:-1]),thiskey.split('/')[-1]).shape
    else:
        return fh5.get_node(thiskey).shape

