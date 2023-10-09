import numpy as np

a = np.load('/home/ana/Study/CVPR/lego/dataset/20200820-subject-03/20200820_135508/836212060125/labels_000001.npz')
# a = np.load('/home/ana/Study/CVPR/lego/dataset/20200820-subject-03/20200820_135508/pose.npz')
for i in a:
    print(i)
b = np.absolute(a['joint_3d'])
# print(b)
# print(a['joint_3d'])
ac = a['joint_3d'] * -1
print(ac)
x = ac[0][:,1]
y = ac[0][:,2]
z = -ac[0][:,0]
print(x.shape)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(x, y, z, s=1)
ax.plot(x, y, z, color='black')
# for i in range(len(x)):
#     ax.plot([x[i], x[(i + 1) % len(x)]], [y[i], y[(i + 1) % len(y)]], [z[i], z[(i + 1) % len(z)]], c='b')

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()