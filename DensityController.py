import tensorflow as tf
import numpy as np
import copy as cp
import random as rd
import serial
import time
import math
import matplotlib.pyplot as plt
from dataglove import *

Mode = 0 #0:TrainData, 1:ReadData, 2:CombineData
MotionIndex = 0
hyper = 1200
choice = 35
glove = Forte_CreateDataGloveIO(0)# right:0 left:1
def HyperSampling(data,time,sample):
    temp1 = []
    temp2 = []
    it = len(data) * 2
    dt = time / len(data)
    for i in range(0,it - 4,2):
        inclination1 = (np.array(data[i + 1]) - np.array(data[i])) / dt
        inclination2 = (np.array(data[i + 2]) - np.array(data[i + 1])) / dt
        doubleinc = (inclination2 - inclination1) / (2 * dt)
        data = np.insert(data,i + 1,(inclination1 + doubleinc * (dt / 2)) * (dt / 2) + data[i],axis=0)
        if len(data) == sample:
            break

    return data

def Choice(data,sample):
    temp1 = []#data
    temp2 = []#lable
    
    idx = rd.sample(range(len(data)),sample)
    idx.sort()
    
    for i in idx:
        temp1.append(data[i])
    
    return temp1

def ShowGraph(data,index):

    plt.title('Data')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.subplot(index)
    plt.plot(data,'.r') 

def HapticOn(wave,ampitude):
        Forte_SendHaptic(glove,0,wave,ampitude)
        Forte_SendHaptic(glove,1,wave,ampitude)
        Forte_SendHaptic(glove,2,wave,ampitude)
        Forte_SendHaptic(glove,3,wave,ampitude)
        Forte_SendHaptic(glove,4,wave,ampitude)
        Forte_SendHaptic(glove,5,wave,ampitude)

def HapticOff():
        Forte_SendHaptic(glove,0,0,0)
        Forte_SendHaptic(glove,1,0,0)
        Forte_SendHaptic(glove,2,0,0)
        Forte_SendHaptic(glove,3,0,0)
        Forte_SendHaptic(glove,4,0,0)
        Forte_SendHaptic(glove,5,0,0)

def HapticShot(wave,amplitude):
    Forte_SendOneShotHaptic(glove,0,wave,amplitude)
    Forte_SendOneShotHaptic(glove,1,wave,amplitude)
    Forte_SendOneShotHaptic(glove,2,wave,amplitude)
    Forte_SendOneShotHaptic(glove,3,wave,amplitude)
    Forte_SendOneShotHaptic(glove,4,wave,amplitude)
    Forte_SendOneShotHaptic(glove,5,wave,amplitude)

def PoseTrigger(Hand,FlexSensors):
    Thumb = FlexSensors[0] + FlexSensors[1]
    Index = FlexSensors[2] + FlexSensors[3]
    Middle = FlexSensors[4] + FlexSensors[5]
    Ring = FlexSensors[6] + FlexSensors[7]
    Pinky = FlexSensors[8] + FlexSensors[9]
    if Thumb >= 50 and Index <= 40 and Middle >= 90 and Ring >= 90 and Pinky >= 60:

        HapticOn(126,0.7)
        return 1
    else:
        HapticOff()
        #HapticShot(Hand,1,1)
        return 0
        
def delzero(data):
    temp = []
    for i in range(len(data)):
        if abs(data[i]) >= 0.1:
            temp.append(data[i])
    return temp

def InclinationCapture(InclinationData,AmountOfChange,NumberOfFilter,SamplePerFilter):

    FilterSize = AmountOfChange / NumberOfFilter
    Delta = 0
    d = 0
    Filter = []
    retData = []
    DD = []
    print(len(InclinationData))
    for i in range(len(InclinationData)):
        Delta +=abs(InclinationData[i])
        
        if Delta >= FilterSize:
            print(len(Filter))
            d+=len(Filter)
            #process
            retData.extend(DensityController2(Filter,SamplePerFilter))
            Filter = []
            Delta = 0
        else :
            Filter.append(InclinationData[i])
    retData.extend(Filter)
    print(len(Filter))
    d+=len(Filter)
    temp = 0
    for i in range(len(retData)):
        temp+=retData[i]
        DD.append(temp)
    print(d)
    return DD

def Integration(data):
    res = []
    temp = 0
    for i in range(len(data)):
        temp+=data[i]
        res.append(temp)
    #return np.array(res)
    return np.array(res)/(max(res)+abs(min(res))) 

def DensityController(Data, num, combineCount):
    retData = []
    dataTemp = cp.copy(Data)
    It = 0
    if len(Data) > num:
        while 1:
            if It >= len(dataTemp) - 1:
                It = 0
                continue
            dataTemp[It]+=dataTemp[It + 1]
            del dataTemp[It + 1]
            if len(dataTemp) == num:
                break
            It+=1
    elif 2 < len(Data) < num:
        while 1:
            for i in range(0,len(dataTemp),2):
                dataTemp[i + 1]/=2
                dataTemp.insert(i,dataTemp[i + 1] / 2)
                if len(dataTemp) == num:
                    break
            if len(dataTemp) == num:
                break
 
    return dataTemp
   
def DensityController2(Data, num):
    retData = []
    Element=0
    dataTemp = cp.copy(Data)
    dataTemp.append(0)
    In=int(len(dataTemp)/num)+1 
    Rn=(len(dataTemp)/num)%1    
    dlt=0
    for i in range(num):
        dataTemp[i*(In-1)]
        for j in range(In):
            Element+=dataTemp[i*In]*(1-Rn)
            if j==(In-1):
                dlt=dataTemp[i*(In)+j]*Rn
                Element+=dlt
        retData.append(Element)
        Element=0
        Rn=(Rn*(i+1))%1
        In=int((In+Rn)*(i+1))
    return retData

                
def GenerateData2(Mode,MotionIndex):

    FileName = ''
    Lable = tf.one_hot([0,1,2,3,4,5,6,7,8,9,10],depth=11)
    save = []
    data = [[[],[],[]],[]]
    InclinationData = [[[],[],[]],[]]
    hypersave = []
    choicesave = []
    IMU = []
    DeltaIMU = []
    InitializedData = []
    Iterator = 0
    preTrigger = 0
    AmountOfChange = [0,0,0]
    print("start")
    if Mode == 0:
        try:
            while True:
                try:
                    if preTrigger == 0 and PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #CaptureStart
                        print('Capture Start')
                        InitializedData = Forte_GetEulerAngles(glove)
                        BeforeData = Forte_GetEulerAngles(glove)
                        AmountOfChange = [0,0,0]
                        startTime = time.time()
                        preTrigger = 1
                    elif preTrigger == 1 and PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #Is Capturing
                        IMU = np.array(Forte_GetEulerAngles(glove)) - np.array(InitializedData)
                        DeltaIMU = np.array(Forte_GetEulerAngles(glove)) - np.array(BeforeData)
                        BeforeData = Forte_GetEulerAngles(glove)

                        data[0][0].append(IMU[0])
                        data[0][1].append(IMU[1])
                        data[0][2].append(IMU[2])

                        InclinationData[0][0].append(DeltaIMU[0])
                        InclinationData[0][1].append(DeltaIMU[1])
                        InclinationData[0][2].append(DeltaIMU[2])


                    elif preTrigger == 1 and PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 0: #End Capturing
                        deltaTime = time.time() - startTime
                        plt.figure(figsize=(7,7))           

                        #print(deltaTime)
                        #print(len(data),len(data[0][0]),len(data[0][1]),len(data[0][2]))
                        data[1].extend(np.array(Lable[MotionIndex]))
                        InclinationData[1].extend(np.array(Lable[MotionIndex]))

                        AmountOfChange[0] = sum(list(map(abs,InclinationData[0][0])))
                        AmountOfChange[1] = sum(list(map(abs,InclinationData[0][1])))
                        AmountOfChange[2] = sum(list(map(abs,InclinationData[0][2])))

                        print(AmountOfChange[1],'->',AmountOfChange[1] / 100)
                        print(np.array(data[1]))

                        P = InclinationCapture(delzero(InclinationData[0][1]),AmountOfChange[1],10,10)

                        #print(AmountOfChange[0],AmountOfChange[1],AmountOfChange[2])

                        ShowGraph(InclinationData[0][1],321)
                        ShowGraph(Integration(InclinationData[0][1]),322)

                        ShowGraph(delzero(InclinationData[0][1]),323)
                        ShowGraph(Integration(delzero(InclinationData[0][1])),324)
                        ShowGraph(P,325)

                        plt.show()

                        #ShowGraph(InclinationCapture(delzero(InclinationData[0][1]),AmountOfChange[1]))
                        #ShowGraph(Integration(InclinationCapture(InclinationData[0][1],AmountOfChange[1])))
                        
                        save.append(data)
                        save.append(data)
                        data = [[[],[],[]],[]]
                        InclinationData = [[[],[],[]],[]]
                        
                        preTrigger = 0
                        Iterator+=1                    
                    #print(imu)
                    #else:
                    #    print(PoseTrigger(glove,Forte_GetSensorsRaw(glove)),Forte_GetSensorsRaw(glove))
                except(GloveDisconnectedException):
                    print("Glove is Disconnected")
                    pass
        except(KeyboardInterrupt):
            Forte_DestroyDataGloveIO(leftHand)
            exit()

    elif Mode == 1:
        print("Read Data Mode")
        LoadChoice = np.load('./Data/ChoiceSample/Train1.npy',allow_pickle=True)
        LoadHyper = np.load('./Data/HyperSample/Train1.npy',allow_pickle=True)
        print(len(LoadChoice[0][0]))
        print(len(LoadHyper[0][0]))
        #ShowGraph([],LoadChoice[0][0],LoadHyper[0][0],0)
  


    elif Mode == 2:
        print("Combine Mode")
        savetemp = []
        Motion1 = []
        Motion2 = []
        Motion3 = []

        Motion1 = np.load('./Data/HyperSample/Train1.npy',allow_pickle=True)
        Motion2 = np.load('./Data/HyperSample/Train2.npy',allow_pickle=True)
        Motion3 = np.load('./Data/HyperSample/Train3.npy',allow_pickle=True)
        savetemp.extend(Motion1) 
        savetemp.extend(Motion2) 
        savetemp.extend(Motion3) 
        np.random.shuffle(savetemp)
        np.save("./Data/HyperSample/" + str(len(savetemp)),savetemp,True)
        print(len(savetemp),"HyperCombined")


        savetemp = []
        Motion1 = []
        Motion2 = []
        Motion3 = []

        Motion1 = np.load('./Data/ChoiceSample/Train1.npy',allow_pickle=True)
        Motion2 = np.load('./Data/ChoiceSample/Train2.npy',allow_pickle=True)
        Motion3 = np.load('./Data/ChoiceSample/Train3.npy',allow_pickle=True)
        savetemp.extend(Motion1) 
        savetemp.extend(Motion2)
        savetemp.extend(Motion3) 
        np.random.shuffle(savetemp)
        np.save("./Data/ChoiceSample/" + str(len(savetemp)),savetemp,True)
        print(len(savetemp),"ChoiceCombined")

#GenerateData2(Mode,MotionIndex)

plt.figure(figsize=(7,7))
a=[]
b=[]
for i in range(22):
    a.append(i)


print(sum(a))
c=DensityController2(a,10)
print(sum(c))

#print((a))
#print((b))

ShowGraph((a),221)
ShowGraph((c),222)
ShowGraph(Integration(a),223)
ShowGraph(Integration(c),224)
plt.show()    
 