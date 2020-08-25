import tensorflow as tf
import numpy as np
import copy as cp
import random as rd
import serial
import time
import math
import matplotlib.pyplot as plt
from dataglove import *

Mode = 2 #0:TrainData, 1:ReadData, 2:CombineData, 3:DummyData 
MotionIndex = 0 # 0 : dummy
hyper = 1200
choice = 35
Routine=100
FileName=''
FilePath='C:/Users/gun18/source/repos/Data_200824/'
glove = Forte_CreateDataGloveIO(0)# right:0 left:1
#plt.figure(figsize=(12,4))           

class GloveControl:
    
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
        if Thumb >= 100 and Index <= 100 and Middle >= 120 and Ring >= 100 and Pinky >= 120:

            GloveControl.HapticOn(126,0.7)
            return 1
        else:
            GloveControl.HapticOff()
            #HapticShot(Hand,1,1)
            return 0

class DataProcess:
    
    def HyperSampling(data,time,sample):
   
        if len(data)<sample:
            while 1:
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
                if len(data) == sample:
                    break
        return data

    def RandomSelect(data,sample):
        temp1 = []#data
        temp2 = []#lable
    
        idx = rd.sample(range(len(data)),sample)
        idx.sort()
    
        for i in idx:
            temp1.append(data[i])
    
        return temp1

    def Expansion(DeltaData):
        dataTemp = cp.copy(DeltaData)
        retData = []
        for i in range(0,len(dataTemp)-1,1):
            retData.append(dataTemp[i])
            retData.append((dataTemp[i]+dataTemp[i+1])/2)
        retData.append(dataTemp[len(dataTemp)-1])
        return retData

    def Constraction(DeltaData):
        dataTemp = cp.copy(DeltaData)
        retData=[]
        for i in range(0,len(dataTemp)-1,2):
            retData.append(dataTemp[i]+dataTemp[i+1])

        return retData    

    def delzero(data,f=0.1):
        temp = []
        for i in range(len(data)):
            if abs(data[i]) >= 0.1:
                temp.append(data[i])
        return temp

    def delover(data,f=20):
        temp = []
        for i in range(len(data)):
            if abs(data[i]) <= f:
                temp.append(data[i])
        return temp

    def Integration(data):
        res = []
        temp = 0
        for i in range(len(data)):
            temp+=data[i]
            res.append(temp)
        return np.array(res)
        return np.array(res)

    def Nomalize(data):

        return (np.array(data)/(max(data)+abs(min(data))))

    def Differential(D):
        Data = cp.copy(D)
        retData=[]
        for i in range(len(Data)-1):
            retData.append(Data[i+1]-Data[i])
        return retData

class DataIO:

    def DataRead(FileName,x,y,Trigger=False):    
        Data = np.load('./'+FileName+'.npy',allow_pickle=True)
        np.random.shuffle(Data)
        DD=[]
        DT=[0,0,0]
        for i in range(len(Data)):
            DT[0]=(Data[i][0][0])
            DT[1]=(Data[i][0][1])
            DT[2]=(Data[i][0][2])

            DD.append(DT)
            DT =[0,0,0]

        train_X=[]
        train_Y=[]
        test_X=[]
        test_Y=[]
        for i in range(len(DD)):
            if Trigger:
            
                DD[i]=np.array(DD[i]).reshape(x,y,1)

            if i <= len(DD)*0.80:
                train_X.append(DD[i])
                train_Y.append(Data[i][1])


            else:
                #if Data[i][1]!=0:
                    test_X.append(DD[i])
                    test_Y.append(Data[i][1])

        return (train_X,train_Y),(test_X,test_Y)
              
    def ShowGraph(data,index=111,form='.r'):

        #plt.title('Data')
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.subplot(index)
        plt.plot(data,form) 

    def GetTest(model):
        try:
            while True:
                print(np.argmax(model.predict(np.array(GetTestData()).reshape(1,3,300,1))))
        except(KeyboardInterrupt):
            Forte_DestroyDataGloveIO(glove)
            exit    

def GenerateData(Mode,MotionIndex):

    FileName='Motion'+str(MotionIndex)+'.npy'
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
    try:
        if Mode == 0:
            try:
                while True:
                    try:
                        if preTrigger == 0 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #CaptureStart
                            print('Capture Start')
                            BeforeData = Forte_GetEulerAngles(glove)
                            AmountOfChange = [0,0,0]
                            startTime = time.time()
                            preTrigger = 1

                        elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #Is Capturing
                            DeltaIMU = np.array(Forte_GetEulerAngles(glove)) - np.array(BeforeData)
                            BeforeData = Forte_GetEulerAngles(glove)
                            InclinationData[0][0].append(DeltaIMU[0])
                            InclinationData[0][1].append(DeltaIMU[1])
                            InclinationData[0][2].append(DeltaIMU[2])

                        elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 0: #End Capturing
                            if len(DataProcess.delzero(InclinationData[0][1]))<20 or len(DataProcess.delzero(InclinationData[0][2]))<20 or len(DataProcess.delzero(InclinationData[0][0]))<20:
                                InclinationData = [[[],[],[]],[]]
                                preTrigger = 0
                                print('re')
                                continue

                            deltaTime = time.time() - startTime
                            InclinationData[1]=MotionIndex
                            C= input('save?')
                            if C=='':
                                save.append(InclinationData)
                                print('saved',Iterator)
                                Iterator+=1                    
                            InclinationData = [[[],[],[]],[]]
                            preTrigger = 0
                            if Iterator>=Routine:
                                np.save(FilePath+FileName,save,True)
                                print("SaveComplete")
                                break
            
                    except(GloveDisconnectedException):
                        print("Glove is Disconnected")
                        pass

            except(KeyboardInterrupt):
                Forte_DestroyDataGloveIO(glove)
                exit() #TrainData

        elif Mode == 1:
            print("Read Data Mode")
            Motion =(np.load(FilePath+FileName,allow_pickle=True))

            print(len(Motion[0][0]))
            print('Motion'+str(0)+' loaded')
            print(Motion[0][1])
            for i in range(len(Motion)):
                plt.figure(figsize=(12,8))           
                DataIO.ShowGraph(DataProcess.delover(DataProcess.delzero(Motion[i][0][0])),231)
                DataIO.ShowGraph(DataProcess.delover(DataProcess.delzero(Motion[i][0][1])),232)
                DataIO.ShowGraph(DataProcess.delover(DataProcess.delzero(Motion[i][0][2])),233)
                DataIO.ShowGraph((DataProcess.Integration((DataProcess.delzero(Motion[i][0][0])))),234)
                DataIO.ShowGraph((DataProcess.Integration((DataProcess.delzero(Motion[i][0][1])))),235)
                DataIO.ShowGraph((DataProcess.Integration((DataProcess.delzero(Motion[i][0][2])))),236)
                print(Motion[i][1])
                plt.show() #DataRead
      
        elif Mode == 2:
            print("Combine Mode")
            savetemp = []
            Motion=[]
            for i in range(11):
                Motion.append(np.load(FilePath+'Motion'+str(i)+'.npy',allow_pickle=True))
                savetemp.extend(Motion[i])
                print('Motion'+str(i)+' added')
        
            for i in range(len(savetemp)):
                #if i==0:
                #    savetemp[i][0][0]=DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration(((savetemp[i][0][0]))),1,300))
                #    savetemp[i][0][1]=DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration(((savetemp[i][0][1]))),1,300))
                #    savetemp[i][0][2]=DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration(((savetemp[i][0][2]))),1,300))
                savetemp[i][0][0]=DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration((DataProcess.delzero(savetemp[i][0][0]))),1,300))
                savetemp[i][0][1]=DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration((DataProcess.delzero(savetemp[i][0][1]))),1,300))
                savetemp[i][0][2]=DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration((DataProcess.delzero(savetemp[i][0][2]))),1,300))
   
            np.save('CombinedData'+str(len(savetemp)),savetemp,True)
            print("CombinedData","Operation Complete") #CombineData

        elif Mode==3:#DummyData
            t=time.time()
            tt=[[[],[],[]],0]
            savetemp=[]
            BeforeData = 0
            DummyTime=300

            InclinationData=[[[],[],[]],0]
            print('Generate Dummy Data')
            try:
                while True:
                    try:
                            DeltaIMU = np.array(Forte_GetEulerAngles(glove)) - np.array(BeforeData)

                            BeforeData = Forte_GetEulerAngles(glove)

                            InclinationData[0][0].append(DeltaIMU[0])
                            InclinationData[0][1].append(DeltaIMU[1])
                            InclinationData[0][2].append(DeltaIMU[2])                  
                        
                   
                            if time.time()-t>=DummyTime:
                                InclinationData[0][0]=DataProcess.Integration(DataProcess.delover(DataProcess.delzero(InclinationData[0][0])))
                                InclinationData[0][1]=DataProcess.Integration(DataProcess.delover(DataProcess.delzero(InclinationData[0][1])))
                                InclinationData[0][2]=DataProcess.Integration(DataProcess.delover(DataProcess.delzero(InclinationData[0][2])))
                                for i in range(int(min([len(InclinationData[0][0]),len(InclinationData[0][1]),len(InclinationData[0][2])])/60)):
                                    for j in range(60):
                                        tt[0][0].append(InclinationData[0][0][i*60+j])
                                        tt[0][1].append(InclinationData[0][1][i*60+j])
                                        tt[0][2].append(InclinationData[0][2][i*60+j])
                                    savetemp.append(tt)
                                    tt=[[[],[],[]],0]
                                np.save(FilePath+"Motion0.npy",savetemp,True)
                                print("SaveComplete"+str(len(savetemp)))

                                for i in range(len(savetemp)):
                                    plt.figure(figsize=(12,4))
                                    DataIO.ShowGraph(DataProcess.Integration(DataProcess.delzero(savetemp[i][0][0])),131)
                                    DataIO.ShowGraph(DataProcess.Integration(DataProcess.delzero(savetemp[i][0][1])),132)
                                    DataIO.ShowGraph(DataProcess.Integration(DataProcess.delzero(savetemp[i][0][2])),133)
                                    plt.show()
                                break
                            BeforeData = Forte_GetEulerAngles(glove)

                    except(GloveDisconnectedException):
                        print("Glove is Disconnected")
                        pass
            except(KeyboardInterrupt):
                Forte_DestroyDataGloveIO(glove)
                exit()
    except(KeyboardInterrupt):
        Forte_DestroyDataGloveIO(glove)
        exit()
def GetTestData():
    InclinationData = [[],[],[]]
    DeltaIMU = []
    InitializedData = []
    preTrigger = 0
    try:
        while True:
            try:
                if preTrigger == 0 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #CaptureStart
                    print('Capture Start')
                    BeforeData = Forte_GetEulerAngles(glove)
                    preTrigger = 1

                elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 1: #Is Capturing
                    

                    DeltaIMU = np.array(Forte_GetEulerAngles(glove)) - np.array(BeforeData)
                    BeforeData = Forte_GetEulerAngles(glove)

                    InclinationData[0].append(DeltaIMU[0])
                    InclinationData[1].append(DeltaIMU[1])
                    InclinationData[2].append(DeltaIMU[2])

                elif preTrigger == 1 and GloveControl.PoseTrigger(glove,Forte_GetSensorsRaw(glove)) == 0: #End Capturing
                    
                    InclinationData[0]=DataProcess.delzero(InclinationData[0])
                    InclinationData[1]=DataProcess.delzero(InclinationData[1])
                    InclinationData[2]=DataProcess.delzero(InclinationData[2])

                    if len(InclinationData[1])<20 or len(InclinationData[2])<20 or len(InclinationData[0])<20:
                        InclinationData = [[],[],[]]
                        preTrigger = 0
                        print('re')
                        continue
                    InclinationData[0]=DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration(InclinationData[0]),1,300))
                    InclinationData[1]=DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration(InclinationData[1]),1,300))
                    InclinationData[2]=DataProcess.Nomalize(DataProcess.HyperSampling(DataProcess.Integration(InclinationData[2]),1,300))
                    C= input('save?')
                    if C=='':
                        return InclinationData
                        print('end')
                    InclinationData = [[],[],[]]
                    preTrigger = 0

            except(GloveDisconnectedException):
                print("Glove is Disconnected")
                pass

    except(KeyboardInterrupt):
        Forte_DestroyDataGloveIO(glove)
        exit()


#GenerateData(Mode,MotionIndex)
