import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class BlochSphereSimulator(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Bloch Sphere Simulator")
		self.setGeometry(100, 100, 1200, 800)

		main_widget = QWidget()
		self.setCentralWidget(main_widget)
		layout = QHBoxLayout(main_widget)

		# Matplotlib Figure
		self.figure = Figure(figsize=(10, 10))
		self.canvas = FigureCanvas(self.figure)
		layout.addWidget(self.canvas)

		self.plot_bloch_sphere()
		theta, phi = self.getAnglesForQubit()
		x, y, z = self.transformAnglesToSphereCoordinates(theta, phi)
		self.drawQuiver(x, y, z)
		self.ax = None

	def plot_bloch_sphere(self):
		self.figure.clear()
		self.ax = self.figure.add_subplot(111, projection='3d')
	
		
		# Add ticks for -1, 0, 1 on each axis
		self.ax.set_xticks([-1, 0, 1])
		self.ax.set_yticks([-1, 0, 1])
		self.ax.set_zticks([-1, 0, 1])
		
		# Remove tick labels
		self.ax.set_xticklabels([])
		self.ax.set_yticklabels([])
		self.ax.set_zticklabels([])
		
		# Set the limits to create a perfect sphere
		self.ax.set_xlim([-1, 1])
		self.ax.set_ylim([-1, 1])
		self.ax.set_zlim([-1, 1])
		
		# Set equal aspect ratio
		self.ax.set_box_aspect((1, 1, 1))
		
		# Remove grid and background
		self.ax.grid(False)
		self.ax.xaxis.pane.fill = False
		self.ax.yaxis.pane.fill = False
		self.ax.zaxis.pane.fill = False
		self.ax.xaxis.pane.set_edgecolor('none')
		self.ax.yaxis.pane.set_edgecolor('none')
		self.ax.zaxis.pane.set_edgecolor('none')
	
		
  		#Draw the sphere 

		#This creates a grid for the sphere -> it means that we are going to have 20 points in the theta direction and 10 points in the phi direction
		u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
		x = np.cos(u)*np.sin(v)
		y = np.sin(u)*np.sin(v)
		z = np.cos(v)

		#Plot the sphere
		self.ax.plot_wireframe(x,y,z, color="gray", alpha=0.2)

		#Draw the axes
		#self.ax.quiver(0,0,0,1,0,0, color='r', arrow_length_ratio=0.1)
		#self.ax.quiver(0,0,0,0,1,0, color='g', arrow_length_ratio=0.1)
		#self.ax.quiver(0,0,0,0,0,1, color='b', arrow_length_ratio=0.1)

		#Set the limits for the axes
		self.ax.set_ylim([-1, 1])
		self.ax.set_zlim([-1, 1])
		self.ax.set_xlim([-1, 1])

		#Set the labels for the axes
		self.ax.set_xlabel("X")
		self.ax.set_ylabel("Y")
		self.ax.set_zlabel("Z")

		self.canvas.draw()

	def apply_gate(self):
		# Implement gate application logic here
		pass

	def apply_custom_unitary(self):
		# Implement custom unitary application logic here
		pass
 
	def getAnglesForQubit(self):
		
		#|1>
		complexA = {'Re': 1, 'Im': 0}
		complexB = {'Re': 0, 'Im': 0}
		
		phi = 0
		#If the real part of complexA is 0, it means that the it will be |0> so the tangent part is positive infinity
		if (complexA['Re'] == 0):
			phi = np.pi/2
		elif (complexB['Re'] == 0):
			phi = np.pi
		else:
			phi = np.arctan(complexB['Im']/complexB['Re']) - np.arctan(complexA['Im']/complexA['Re'])

		theta = np.arcsin(np.sqrt(complexB['Re']**2 + complexB['Im']**2))

		return theta, phi
	
	def drawQuiver(self, x, y, z):
		self.ax.quiver(0,0,0,x,y,z, color='r', arrow_length_ratio=0.1)
		self.canvas.draw()

	def transformAnglesToSphereCoordinates(self, theta, phi):
     
		#Transform the angles to the sphere
  
		#We need to transform the angles to the sphere -> we need to subtract the phi by 90 degrees
  
		print("theta, phi: ", theta, phi)
		x = np.cos(phi) * np.cos(theta)
		y = np.cos(phi) * np.sin(theta)
		z = np.sin(phi)
  
		print("x, y, z: ", x, y, z)
		return x, y, z
   
		

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BlochSphereSimulator()
    main_window.show()
    sys.exit(app.exec_())