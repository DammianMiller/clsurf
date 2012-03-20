#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Woz data graph
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import operator
import sys

class KernelClass:
    def __init__(self):
        self.name = None
        self.minTime = -1
        self.maxTime = 0
        self.starts = []
        self.ends = []
        self.enqs = []
        self.submits = []

    def printData(self):
        print "--------------------"
        print self.name
        print "Min:",self.minTime
        print "Max:",self.maxTime
        print
        print "Enqueues:",self.enqs
        print
        print "Submits:",self.submits
        print
        print "Starts:",self.starts
        print
        print "Ends:",self.ends
        print "....................."

"""
dataReader = csv.reader(open('wozData.csv','rb'))
"""

dataReader = csv.reader(open('testdata_nv.txt','rb'))
header = dataReader.next()

Category,KernelName,CallNo,Enqueue,Submit,Start,End = zip(*dataReader)

Enqueue = map(int,Enqueue)
Submit = map(int,Submit)
Start = map(int,Start)
End = map(int,End)

kernels = ["Scan","Transpose", "BuildHessianDet","NonMaxSupression","GetOrientations1","GetOrientations2","CreateDescriptors","NormalizeDescriptors"]


#kernels = ["copyBufferToDevice","copyImageToDevice","Scan","Transpose", #"BuildHessianDet","NonMaxSupression","GetOrientations","GetOrientations2","CreateDescriptors","NormalizeDescriptors"]
colors = ['gray','red','blue','green','cyan','magenta','yellow','purple','orange','green','pink']

kernelClasses = []
maxTime = 0

for k in kernels:
    kClass = KernelClass();
    kClass.name = k
    for idx,kname in enumerate(KernelName):
        if kname.find(k) == 0:
            #if EventType[idx]=='ENQ':
            kClass.enqs.append(Enqueue[idx])
            #elif EventType[idx]=='SUBMIT':
            kClass.submits.append(Submit[idx])
            #elif EventType[idx]=='START':
            kClass.starts.append(Start[idx])
            #elif EventType[idx]=='END':
            kClass.ends.append(End[idx])
            if kClass.minTime == -1 or kClass.minTime > Enqueue[idx]:
                kClass.minTime = Enqueue[idx]
            if kClass.maxTime < End[idx]:
                kClass.maxTime = End[idx]
            if maxTime < End[idx]:
                maxTime = End[idx]
    kernelClasses.append(kClass)

# sort the class from min enqueue time to max time
kernelClasses.sort(key=operator.attrgetter('minTime'))

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.broken_barh([(kernel.minTime,kernel.maxTime-kernel.minTime)],(idx*10,10),facecolors=colors[idx],label=kernel.name)

kernelLabels=[]
yticksLocations=[]
for idx,kernel in enumerate(kernelClasses):
    #if(kernelClasses[idx].minTime != -1):
    ax.broken_barh([(kernel.minTime,kernel.maxTime-kernel.minTime)],(idx*10,10),facecolors="white",label=kernel.name)
    #else:
    #    print "else case"
    kernelLabels.append(kernel.name)
    yticksLocations.append(idx*10 + 10 / 2)
    #ax.annotate(kernel.name,(0,idx*10+10/2),xytext=(-3,-3),textcoords='offset points')
    for t in kernel.enqs:
        ax.annotate('+',(t,idx*10+10/2),xytext=(-3,-18),textcoords='offset points',size=18)
    for t in kernel.submits:
        ax.annotate('^',(t,idx*10+10/2),xytext=(-3,-13),textcoords='offset points',size=18)
    for t in kernel.starts:
        ax.annotate('*',(t,idx*10+10/2),xytext=(-3,-5),textcoords='offset points',size=18)
    for t in kernel.ends:
        ax.annotate(u'•',(t,idx*10+10/2),xytext=(-3,8),textcoords='offset points',size=18)

p1 = plt.Rectangle((0, 0), 0.5, 0.5, fc="b")
topOfGraphY = len(kernelClasses)*10
graphXRange = maxTime
ax.text(0.1 * graphXRange,topOfGraphY,"Legend")
ax.text(0.1 * graphXRange,topOfGraphY-5,"+ Enqueue")
ax.text(0.1 * graphXRange,topOfGraphY-10,"^ Submit")
ax.text(0.1 * graphXRange,topOfGraphY-15,"* Start")
ax.text(0.1 * graphXRange,topOfGraphY-20,u"• End")
legendBox = patches.Rectangle([0.08*graphXRange,topOfGraphY-22], 0.20*graphXRange, 26, facecolor="white", edgecolor="blue")
plt.gca().add_patch(legendBox)
ax.set_ylim(0,len(kernelClasses)*11)
ax.set_xlim(0,maxTime+5*maxTime/100) #extra 5% on the right so the boxes are seen
ax.set_yticklabels(kernelLabels)
ax.set_yticks(yticksLocations)
plt.xlabel('Time (us)')
plt.title('Execution Flow of SURF OpenCL kernels')
plt.ylabel('Kernel Name')

"""
ax.broken_barh([ (10, 50), (100, 20),  (130, 10)] , (20, 9),
                facecolors=('red', 'yellow', 'green'))
ax.set_ylim(5,35)
ax.set_xlim(0,200)
ax.set_xlabel('seconds since start')
ax.set_yticks([15,25])
ax.set_yticklabels(['Bill', 'Jim'])
ax.grid(True)
"""
"""
ax.annotate('race interrupted', (61, 25),
            xytext=(0.8, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')
"""
plt.show()
fig.savefig('test.eps')
quit()
