import os
import sys
import glob
import re
import csv
import operator

# Imports for plotting
import matplotlib
matplotlib.use('gtkagg')
import matplotlib.pylab as pylab
import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np

colorlist=['g','b','c','r','m','k','y','w']
EPSILON = 0.00000001

# This class holds the information from an OpenCL event
class Event:
   def __init__(self, type, desc, time):
      self.type = type
      self.desc = desc
      self.time = time
   
   def __str__(self):
      return '%s Event: \n\tName: %s\n\tTime: %f' % (self.type, self.desc, self.time)	
      
# This class holds the information related to the OpenCL host 
# that was printed in the dum
class Info: 
   def __init__(self, desc):
      self.desc = desc[0]
   
   def __str__(self):
      return self.desc
      
# An EventGroup combines all events and info from a single dumped file
class EventGroup:
   def __init__(self):
      self.filename = None
      self.infolist = []
      self.kernel_eventlist = []
      self.io_eventlist = []
      self.compile_eventlist = []
      self.user_eventlist = []
      
   def __str__(self):
      return "Event group for %s" % self.filename	
      
   def compare(self, other):
      # Check to make sure the lists are all the same length
      if(len(self.kernel_eventlist) != len(other.kernel_eventlist) or
         len(self.io_eventlist) != len(other.io_eventlist) or
         len(self.compile_eventlist) != len(other.compile_eventlist) or
         len(self.user_eventlist) != len(other.user_eventlist)):
            print('Lists are mismatched')
            return -1
      else:
         print('List lengths are identical')
         
      # Check to make sure all the kernel event descriptions match
      for index in range(0,len(self.kernel_eventlist)):
         if self.kernel_eventlist[index].desc != other.kernel_eventlist[index].desc:
            print('List mismatch [%s,%s]' % (self.kernel_eventlist[index].desc, other.kernel_eventlist[index].desc))
            return -1
      # Check to make sure all the IO event descriptions match
      for index in range(0,len(self.io_eventlist)):
         if self.io_eventlist[index].desc != other.io_eventlist[index].desc:
            print('List mismatch [%s,%s]' % (self.io_eventlist[index].desc, other.io_eventlist[index].desc))
            return -1
      # Check to make sure all the compile event descriptions match
      for index in range(0,len(self.compile_eventlist)):
         if self.compile_eventlist[index].desc != other.compile_eventlist[index].desc:
            print('List mismatch [%s,%s]' % (self.compile_eventlist[index].desc, other.compile_eventlist[index].desc))
            return -1	
      # Check to make sure all the user event descriptions match
      for index in range(0,len(self.user_eventlist)):
         if self.user_eventlist[index].desc != other.user_eventlist[index].desc:
            print('List mismatch [%s,%s]' % (self.user_eventlist[index].desc, other.user_eventlist[index].desc))
            return -1	
            
      return 0
   
   def getDescription(self):
      return self.infolist[0].__str__()
   
   def printEvents(self):
      for entry in self.infolist:
         print entry
      for event in self.kernel_eventlist:
         print event
      for event in self.io_eventlist:
         print event
      for event in self.compile_eventlist:
         print event
      for event in self.user_eventlist:
         print event

 
# This can probably go into a class function later
def searchForEvent(desc, eventList):

   for event in eventList:
      if event.desc == desc:
         return event
         
   return []
         
# This function parses the events out of each file and builds
# a list of each event type.  It returns a list of event lists.
def parse(filename):

   print('Parsing %s' % filename)
   
   # Store the filename for this event group
   eg = EventGroup()
   eg.filename = filename;
   
   # Open the file and read in the lines
   lines = open(filename, 'r').readlines()
   
   # Create a tokenizer using a semicolon delimiter
   tokenizer = re.compile(r'\;')
   
   # Iterate through the lines of the file, parsing each line using
   # the tokenizer and storing the data in the event group
   for line in lines:
      
      # Tokenize the line
      tokens = tokenizer.split(line)
      
      # Store the line appropriately based on the contents
      if tokens[0] == 'Info':
         eg.infolist.append(Info([tokens[1].strip()]))
      elif tokens[0] == 'Kernel':
         eg.kernel_eventlist.append(Event('Kernel',tokens[1].strip(),float(tokens[2].strip())))
      elif tokens[0] == 'IO':
         eg.io_eventlist.append(Event('IO', tokens[1].strip(),float(tokens[2].strip())))
      elif tokens[0] == 'Compile':
         eg.compile_eventlist.append(Event('Compile', tokens[1].strip(),float(tokens[2].strip())))
      elif tokens[0] == 'User':
         eg.user_eventlist.append(Event('User', tokens[1].strip(),float(tokens[2].strip())))
      else:
         print('invalid list identifier: %s' % tokens[0])
         exit()
   
#	print('Printing event group')
#	print eg.printEvents()
   
   return eg
   
# Given a list, return a list with all duplicates removed
def uniquify(seq):

    # Uniquify the list (order preserving)
    noDupes = []
    [noDupes.append(i) for i in seq if not noDupes.count(i)]
    return noDupes
   
# Get the events from all lists and create a event group 
# that can be used for plotting
def createMasterLists(allLists):
   
   descGroup = []
   
   # Lists that will hold descriptions only
   kernelDescList = []
   userDescList = [] 
   ioDescList = []
   compileDescList = []

   # For each event type, make a 'total' event, which sums
   # up all of the times for a separate plot
   for eventlist in allLists:
      # Kernel total
      sum = 0
      for kernelEvent in eventlist.kernel_eventlist:
         sum = sum + kernelEvent.time
      eventlist.kernel_eventlist.append(Event('Kernel', 'Total', sum))
      # IO total
      sum = 0
      for ioEvent in eventlist.io_eventlist:
         sum = sum + ioEvent.time
      eventlist.io_eventlist.append(Event('IO', 'Total', sum))
      # Compile total
      sum = 0
      for compileEvent in eventlist.compile_eventlist:
         sum = sum + compileEvent.time   
      eventlist.compile_eventlist.append(Event('Compile', 'Total', sum))
      # User total
      sum = 0
      for userEvent in eventlist.user_eventlist:
         sum = sum + userEvent.time
      eventlist.user_eventlist.append(Event('User', 'Total', sum))
   
   # Concatenate all the kernel descriptions together
   for eventlist in allLists:
      for kernelEvent in eventlist.kernel_eventlist:
         kernelDescList.append(kernelEvent.desc)
      for ioEvent in eventlist.io_eventlist:
         ioDescList.append(ioEvent.desc)
      for compileEvent in eventlist.compile_eventlist:
         compileDescList.append(compileEvent.desc)
      for userEvent in eventlist.user_eventlist:
         userDescList.append(userEvent.desc)
         
   # Then uniquify the lists
   kernelDescList = uniquify(kernelDescList)
   userDescList = uniquify(userDescList)
   ioDescList = uniquify(ioDescList)
   compileDescList = uniquify(compileDescList)
   
   descGroup = [kernelDescList, userDescList, ioDescList, compileDescList]
   
   return descGroup

def roundUp(value):
   
   return (int(value/0.5)+1)*0.5;
   
#-------------------------------------------------------
#            Main program starts here
#-------------------------------------------------------

# Read in the files from disk and parse each one into event groups
path = None
if len(sys.argv) > 1:
   path = sys.argv[1]
else:
   path = '../bin/eventlogs/'

eventGroups = []
for infile in glob.glob(os.path.join(path, '*.surflog')):
   print('Found %s' % infile)
   eventGroups.append(parse(infile))

numEventGroups = len(eventGroups)
   
# Exit if no event groups were found
if numEventGroups == 0:
   print('No event groups found')
   exit(0);

masterLists = createMasterLists(eventGroups)

##
## PLOT EVENTS
##
patches = ['']*numEventGroups

for eventType in range(0,4):

   # Make a new figure
   figure = pyplot.figure(figsize=(16,8), facecolor='w')
   figure.clear()
   
   print('EVENT TYPE: %d' % eventType)
   if eventType == 0:
      chartTitle = 'Kernel Events'
   elif eventType == 1:
      chartTitle = 'User Events'
   elif eventType == 2:
      chartTitle = 'IO Events'
   elif eventType == 3:
      chartTitle = 'Compile Events'
   else:
      print('Unknown event type')
      exit(-1)
      
   figure.suptitle(chartTitle, size=18)
   pyplot.rc('font', size=8)
   
   # Add a subplot for each of the events

   # Compute the number of subplots and their dimensions
   numEvents = len(masterLists[eventType]);
   chartsPerRow = 8
   cols = chartsPerRow
   if numEvents < chartsPerRow:
      cols = numEvents;
   displayCols = cols+1
   rows = numEvents/chartsPerRow
   if numEvents % chartsPerRow != 0:
      rows += 1

   eventCtr = 0
   displayCtr = 1

   for item in masterLists[eventType]:

      # This leaves the last column empty to allow for the legend
      if displayCtr % displayCols == 0:
         displayCtr += 1
      
      # Plot the subplot
      ax = figure.add_subplot(rows, displayCols, displayCtr)

      # Set the X ticks
      ax.set_xticks([x + 0.5 for x in range(0, numEvents)])
      ax.set_xticklabels(range(1,numEvents))
      ax.set_xlabel('Item %d:\n%s' % (eventCtr, item))

      barCtr = 0
      maxY = -1
      
      print "Item = ",
      print item
      # For each event group, search to see if the event is present
      # If so, add it to the plot
      for group in eventGroups:
      
         if eventType == 0:
            eventList = group.kernel_eventlist
         elif eventType == 1:
            eventList = group.user_eventlist
         elif eventType == 2:
            eventList = group.io_eventlist
         elif eventType == 3:
            eventList = group.compile_eventlist
         else:
            print('Unknown event type')
            exit(-1)
            
         event = searchForEvent(item, eventList)
         print event 
         if event != []: 
            # Event exists for this log
            if event.time == 0:
               # The chart will not plot 0
               event.time = EPSILON
            if event.time > maxY:
               maxY = event.time
            patches[barCtr] = pyplot.bar(barCtr, event.time, width=1.0, color=colorlist[barCtr])
         else:
            # Event does not exist
            patches[barCtr] = pyplot.bar(barCtr, EPSILON, width=1.0, color=colorlist[barCtr])
         barCtr += 1
      
     
      # Set the Y ticks
      maxY = roundUp(maxY)
      ax.set_yticks([0, maxY/2, maxY])
      ax.set_yticklabels([0, maxY/2, maxY])
      ax.set_ylim(0,maxY)
      ax.set_ylabel('time (ms)')
      
      displayCtr += 1
      eventCtr += 1

   figure.subplots_adjust(hspace=1.0, wspace=1.0)

   legendStrs = []
   for group in eventGroups:
      legendStrs.append(group.getDescription())
   leg_prop = matplotlib.font_manager.FontProperties(size=8)
   figure.legend((patches), legendStrs, 'right', prop=leg_prop)
      
   pyplot.show()


print('Done')
