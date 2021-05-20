from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation


def dotstoimage(thisx, thisy):
    field = np.zeros((100, 100), dtype=float)
    for x in thisx:
        for y in thisy:
            field[int(np.around(x*5)) + 10, int(np.around(y*5)) + 10] +=1
    return field

fig = plt.figure(figsize=(10, 8))
a = 15
n = 1000
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0], autoscale_on=False, xlim=(-a, a), ylim=(-a, a))
ax2 = fig.add_subplot(gs[0, 1], autoscale_on=False, xlim=(-a, a), ylim=(-a, a))

#(ax1, ax2) = fig.add_subplot(121, autoscale_on=False, xlim=(-a, a), ylim=(-a, a))
#ax2 = fig.add_subplot(121, autoscale_on=False, xlim=(-a, a), ylim=(-a, a))

ax1.set_aspect('equal')
ax1.grid()

ax2.set_aspect('equal')
ax2.grid()

line1, = ax1.plot([], [], 'o-', ls='none', ms = 1)
line2, = ax2.plot([], [], 'o-', ls='none', ms = 1)

time_template = 'time = %.1fs'
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

data = np.loadtxt('results/data/result3drop.txt')

def animate(i):
    thisx = data[i*10][np.array([a for a in np.arange(0, n*3, 3)])]
    thisy = data[i*10][np.array([a+1 for a in np.arange(0, n*3, 3)])]
    thisz = data[i*10][np.array([a+2 for a in np.arange(0, n*3, 3)])]

    line2.set_data(thisz, thisy)
    line1.set_data(thisx, thisy)
    
    #time_text.set_text(time_template % (i*dt))
    return line1, line2, time_text

plt.style.use('seaborn-pastel')
ani = animation.FuncAnimation(
    fig, animate, interval=1, blit=True)
ani.save('results/visuals/fluid3drop.gif', writer='imagemagick', fps = 20)
#plt.show()
'''
np.random.seed(19680801)


fig, ax1 = plt.subplots()
images = []
for i in range(int(20/0.006)):
    images.append(dotstoimage(data[i*10][np.array([a for a in np.arange(0, 400*3, 3)])], data[i*10][np.array([a+1 for a in np.arange(0, 400*3, 3)])]))
    
for i in range(int(20/0.006)):
    ax1.cla()
    ax1.imshow(images[i], interpolation='gaussian')
    #images.append(dotstoimage(data[i*10][np.array([a for a in np.arange(0, 400*3, 3)])], data[i*10][np.array([a+1 for a in np.arange(0, 400*3, 3)])]))
    ax1.set_title("frame {}".format(i))
    # Note that using time.sleep does *not* work here!
    plt.pause(0.001)
'''
#plt.show()
