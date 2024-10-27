import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, TypedDict
from qubit import Qubit, qubit0, qubit1, qubitPlus, qubitMinus
from gates import PauliX, PauliY, PauliZ, Gate, Hadamard

#----------------------------------

	
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
		#Ploting out the Block Sphere
		self.plot_bloch_sphere()
		self.applyGates(qubit0, [Hadamard()])  # Should rotate from |0⟩ to |+⟩
		self.applyGates(qubit1, [Hadamard()])  # Should rotate from |1⟩ to |-⟩
		self.applyGates(qubitPlus, [Hadamard()])  # Should rotate from |+⟩ to |0⟩
		self.applyGates(qubitMinus, [Hadamard()])  # Should rotate from |-⟩ to |1⟩
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
		
	def drawQuiver(self, x, y, z, color: str):
		self.ax.quiver(0,0,0,x,y,z, color=color, arrow_length_ratio=0.1)
		self.canvas.draw()

	def draw_great_circle_arc(self, arc_points):
		arc_array = np.array(arc_points)
		arc_array = arc_array / np.linalg.norm(arc_array, axis=1)[:, np.newaxis]
		self.ax.plot(arc_array[:, 0], arc_array[:, 1], arc_array[:, 2], color='b', linewidth=2)
		self.canvas.draw()
	
	def drawQubit(self, qubit: Qubit, color: str):
		theta, phi = qubit.getAnglesForQubit()
		x, y, z = qubit.transformAnglesToSphereCoordinates(theta, phi)
		self.drawQuiver(x, y, z, color)
 
	def applyGates(self, qubit: Qubit, gates: List[Gate]):
		#Draw the initial qubit
		self.drawQubit(qubit, 'r')
		final_qubit = qubit
		for gate in gates:
			#Get the qubit's coordinates before applying the gate -> Copy of the qubit
			previous_qubit = qubit
			
			#Apply the gate
			qubit = gate.apply(qubit)
			final_qubit = qubit

			#Draw the arc for the gate
			print(previous_qubit.getXYZ())
			print(qubit.getXYZ())
			arc_points = gate.getArcPointsForRotation(previous_qubit.getXYZ(), qubit.getXYZ())
			self.draw_great_circle_arc(arc_points)

		#Draw the final qubit
		self.drawQubit(final_qubit, 'b')



 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BlochSphereSimulator()
    main_window.show()
    sys.exit(app.exec_())