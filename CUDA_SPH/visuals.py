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

fig = plt.figure(figsize=(5, 4))
a = 5
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-a, a), ylim=(-a, a))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', ls='none', ms = 1)

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

data = np.loadtxt('test.txt')

def animate(i):
    thisx = data[i*10][np.array([a for a in np.arange(0, 2*3, 3)])]
    thisy = data[i*10][np.array([a+1 for a in np.arange(0, 2*3, 3)])]

    line.set_data(thisx, thisy)
    
    #time_text.set_text(time_template % (i*dt))
    return line, time_text

plt.style.use('seaborn-pastel')
ani = animation.FuncAnimation(
    fig, animate, interval=1, blit=True)
ani.save('walltest.gif', writer='imagemagick', fps = 20)
#plt.show()
'''
np.random.seed(19680801)


fig, ax = plt.subplots()
images = []
for i in range(int(20/0.006)):
    images.append(dotstoimage(data[i*10][np.array([a for a in np.arange(0, 400*3, 3)])], data[i*10][np.array([a+1 for a in np.arange(0, 400*3, 3)])]))
    
for i in range(int(20/0.006)):
    ax.cla()
    ax.imshow(images[i], interpolation='gaussian')
    #images.append(dotstoimage(data[i*10][np.array([a for a in np.arange(0, 400*3, 3)])], data[i*10][np.array([a+1 for a in np.arange(0, 400*3, 3)])]))
    ax.set_title("frame {}".format(i))
    # Note that using time.sleep does *not* work here!
    plt.pause(0.001)
'''
#plt.show()
