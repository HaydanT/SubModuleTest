import wave as waveReader
import pywt as pyWavelet
import matplotlib.pyplot as plt
import time
from statsmodels import robust
import numpy
import copy
import csv
import sys

global wavPos
global chunkSize
global wavLen
global wavRate
global wavRead
global threshs
global zetas
global inst
global instNext
global state
global stateNext
global instLine
global classification
global sampleL

def freqSine ( samples ):
     sine = numpy.sin(2*numpy.pi*numpy.arange(samples)/float(samples/10))*20000+30000
     noise = numpy.random.normal(0, 1000, samples)
     return sine + noise - 30000

def processWavL( wavReadFile, samples):
    global sampleL
    sampleL = []
    for x in range(0, samples):
        frames = wavReadFile.readframes(1)
        sampleL.append(int.from_bytes(frames[0:1], byteorder='big', signed=True))
    return  sampleL

def wavletTrans (signal):
    return pyWavelet.wavedec(signal, wavelet='db6', mode='periodization', level=None)

def getThreshs(wavLets):
    thresh = []
    for i in range(0, len(wavLets)):
        sigma = robust.mad(wavLets[i]) / 0.6745
        thresh.append(sigma * numpy.sqrt(2 * numpy.log10(len(wavLets[i])) / len(wavLets[i])))
    return thresh

def optimizeZetas ( wavelets, thresholds ):
    zetas = []
    for x in range (0, len(wavelets)):
        zeta = 0
        N0 = 0
        N = len(wavelets[x])
        absWavSum = 0
        GCVLow = -1
        for z in range (0, 11):
            correction = (1-z/10)*thresholds[x]
            for y in range (0, len(wavelets[x])):
                w = wavelets[x][y]
                if (abs(w) < thresholds[x]):
                    N0 = N0 + 1
                    absWavSum += abs(w)
                else:
                    if (w < 0):
                        absWavSum += abs(w + correction)
                    else:
                        absWavSum += abs(w - correction)
            if (N0 > 0):
                GCV = ((1 / N) * absWavSum ** 2)/((N0 / N) ** 2)
                if ((GCV < GCVLow) | (GCVLow == -1)):
                    zeta = z
                    GCVLow = GCV
        zetas.append(zeta)
    return zetas

def thresholdData ( wavelets, threshArray, zetaArray): #has soft thresholding removed
    output = copy.deepcopy(wavelets)
    for x in range (0, len(output)):
        correction = ((1 - zetaArray[x] / 10) * threshArray[x])
        for y in range (0, len(output[x])):
            w = output[x][y]
            if (abs(w) > threshArray[x]):
                #if (w < 0):
                #    output[x][y] += correction
                #else :
                #    output[x][y] -= correction
            else:
                output[x][y] = 0
    return output

def createClassificationArray():
    global instNext, state, stateNext, instLine, wavPos, wavRate, chunkSize, classification, instNext, instLine, inst
    if (int(wavPos / wavRate * 1000) > instNext):
        switch = int(chunkSize - (wavPos - instNext / 1000 * chunkSize))
        classification[:switch] = [state] * (chunkSize - switch)
        classification[switch:] = [stateNext] * (chunkSize - switch)
        state = stateNext
        instLine = instLine + 1
        if (len(inst) > instLine):
            instNext = int(float(inst[instLine][0]))
            stateNext = int(float(inst[instLine][1]))
    else:
        classification[:] = [state] * chunkSize

def cleanWave(sample):
    global threshs, zetas
    wavelet = wavletTrans(sample)
    cleanWavLet = thresholdData(wavelet, threshs, zetas)
    return pyWavelet.waverec(cleanWavLet, 'db6', mode='periodization')

def getZetaThresh(sampsToProcess):
    global wavLen, chunkSize, wavRead, zetas, threshs, wavPos
    zetaRaw = []
    threshRaw = []
    sampleTest = []
    aveZeta = []
    aveTheta = []
    zetas = []
    threshs = []
    for x in range(0, sampsToProcess):
        if ((wavPos + chunkSize) <= wavLen):
            wavPos = wavPos + chunkSize
            sampleTest = processWavL(wavRead, chunkSize)
            wavelet = wavletTrans(sampleTest)
            threshRaw.append(getThreshs(wavelet))
            zetaRaw.append(optimizeZetas(wavelet, threshRaw[x]))
    for i in range(0, len(wavelet)):
        aveZeta.append(0)
        aveTheta.append(0)
        for x in range(0, len(threshRaw)):
            aveZeta[i] = aveZeta[i] + zetaRaw[x][i]
            aveTheta[i] = aveTheta[i] + threshRaw[x][i]
        zetas.append(aveZeta[i] / len(threshRaw))
        threshs.append(aveTheta[i] / len(threshRaw))
    wavRead.rewind()
    wavPos = 0

def dataSetup(wavName):
    #-------------------------Open Wav
    global wavRead, sampleL, sampleR, wavLen, wavPos, wavRate, threshs, zetas, dataClean, dataRecon, chunkSize, instNext, classification, state, stateNext, instLine, inst
    global sampleL
    sampleL = []
    sampleR = []
    sampleL.clear()
    chunkSize = 44100
    wavRead = waveReader.open('C:\Cloud\Study\CNC Chatter\Audio' + wavName + '.wav', 'r')
    wavLen = wavRead.getnframes()
    wavPos = chunkSize
    wavRate = wavRead.getframerate()
    getZetaThresh(10)
    sampleL = processWavL(wavRead, chunkSize)
    dataRecon = cleanWave(sampleL)
    #===Create Classification Array====
    classReader = csv.reader(open('C:\Cloud\Study\CNC Chatter\Audio' + wavName + '.csv', 'rt'))
    instLine = 0
    inst = []
    instNext = 0
    classification = []
    for row in classReader:
        inst.append(row)
    state = inst[instLine][1]
    instLine = instLine + 1
    if (len(inst) > instLine):
        instNext = int(float(inst[instLine][0]))
        stateNext = int(float(inst[instLine][1]))
        createClassificationArray()

wavName = '\lathe_chatter'
dataSetup(wavName)
#|||||Process File|||||
t = time.process_time()
while ((wavPos + chunkSize) <= wavLen):
    createClassificationArray()
    processWavL(wavRead, chunkSize)
    dataRecon = cleanWave(sampleL)
    wavPos += chunkSize
u = time.process_time()
q = u - t
done = 1
