import matplotlib.pyplot as mp4
import numpy as np
n=6400
sim_tedad = 1000
boz = 0
pi = np.pi

# Dayere
ls = np.linspace (0,2*pi, n+1)
x = np.cos(ls)
y = np.sin(ls)
mp4.plot(x,y)

# noghte ha
pointha1 =  np.random.rand(sim_tedad)
pointha2 = np.random.rand(sim_tedad)

poi1 = pointha1 * 2*pi
xl1 = np.cos(poi1)
yl1 = np.sin(poi1)
mp4.plot(xl1,yl1,'ro', markersize=3)
poi2 = pointha2 * 2*pi
xl2 = np.cos(poi2)
yl2 = np.sin(poi2)
mp4.plot(xl2,yl2,'go', markersize=3)

#Khat
for i in range (sim_tedad):
    mokhtasatx = [xl1[i],xl2[i]]
    mokhtasaty = [yl1[i],yl2[i]]
    line =mp4.plot(mokhtasatx, mokhtasaty,'b-', linewidth = .1)

    if (np.sqrt(((yl2[i]-yl1[i])**2)+((xl2[i]-xl1[i])**2)) > 1.732):
        boz += 1

print ('bozorgtarha =', boz)
mp4.show()

