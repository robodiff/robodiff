import numpy as np
import os
from matplotlib import pyplot as plt

def Plot_Robot_At_Epoch(ax,frame,positions,colors,xMin,xMax,yMin,yMax):

   x = positions[:,0]
   y = positions[:,1]

   ax.scatter( x , y , s=1 , c=colors) # , cmap='gray' )

   ax.set_aspect('equal')

   ax.set_xticks([])
   ax.set_yticks([])

   ax.set_xlim([xMin, xMax])
   ax.set_ylim([yMin, yMax])

def Save_Frame(frame):

   # plt.show()

   if frame<10:
      frameName = '000' + str(frame)
   elif frame<100:
      frameName = '00' + str(frame)
   elif frame<1000:
      frameName = '0' + str(frame)
   else:
      frameName = str(frame)
   plt.savefig('../frames/'+frameName+'.png',bbox_inches='tight',dpi=300)

   plt.cla()

# ----------------- Main function ----------------------------------------

tmp  = np.load('../frames/epoch0/positions0.npy')

arr = np.zeros([2,tmp.shape[0],tmp.shape[1],tmp.shape[2]+1],dtype='f')

arr[0,:,:,0:2] = np.load('../frames/epoch0/positions0.npy')
arr[0,:,:,  2] = np.load('../frames/epoch0/actuation0.npy')

arr[1,:,:,0:2] = np.load('../frames/epoch9/positions9.npy')
arr[1,:,:,  2] = np.load('../frames/epoch9/actuation9.npy')

xMin = np.min(np.min(np.min(arr[:,:,:,0])))
xMax = np.max(np.max(np.max(arr[:,:,:,0])))

yMin = np.min(np.min(np.min(arr[:,:,:,1])))
yMax = np.max(np.max(np.max(arr[:,:,:,1])))

numFrames = arr.shape[1]

for frame in range(1,numFrames,5):

   print('constructing frame ' + str(frame) + ' of ' + str(numFrames) )

   fig, ax = plt.subplots(nrows=2, ncols=1)

   for epoch in [0,1]:

      positions = arr[epoch,frame,:,0:2]
      colors    = arr[epoch,1    ,:,2]

      Plot_Robot_At_Epoch(ax[epoch],frame,positions,colors,xMin,xMax,yMin,yMax)

   Save_Frame(frame)

   plt.close()
