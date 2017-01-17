import pylab as plt
import numpy as np


# counter = 0
# x = []
# x.append(counter)

# y = []
# y.append(100000)
# plt.plot(x,y)
# # in your loop, append new data values to the two lists
# while True:
# 	counter += 1
# 	x.append(counter)
# 	y.append(np.random.random()*70000)
# 	# all 
# 	plt.gca().lines[0].set_xdata(x)
# 	plt.gca().lines[0].set_ydata(y)
# 	plt.gca().relim()
# 	plt.gca().autoscale_view()
# 	plt.pause(1.0);


counter = 0
x_axis, y_axis = [],[]

first_plot = True

while True:
	if (first_plot):
		plt.xlabel('Epoch and Iterations')
		plt.ylabel('Loss')
		plt.title('Loss graph')
		plt.plot(x_axis,y_axis)
		first_plot = False
	counter += 1
	x_axis.append(counter)
	y_axis.append(np.random.random()*70000)
	# all 
	plt.gca().lines[0].set_xdata(x_axis)
	plt.gca().lines[0].set_ydata(y_axis)
	plt.gca().relim()
	plt.gca().autoscale_view()
	plt.pause(1.0);
