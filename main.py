import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, TypedDict

class ComplexNumber(TypedDict):
    Re: float
    Im: float

class Qubit():
	def __init__(self, alfa: ComplexNumber, beta: ComplexNumber):
		self.alfa = alfa
		self.beta = beta
    
	def getAnglesForQubit(self) -> Tuple[float, float]: 
		theta = 0
		phi = 0
		#If the real part of complexA is 0, it means that the it will be |0> so the tangent part is positive infinity
		if (self.alfa['Re'] == 0):
			theta = np.pi
			phi = 0
		elif (self.beta['Re'] == 0):
			theta = 0
			phi = 0
		else:
			theta = 2*np.arcsin(np.sqrt(self.beta['Re']**2 + self.beta['Im']**2))
			phi = np.arctan(self.beta['Im']/self.beta['Re']) - np.arctan(self.alfa['Im']/self.alfa['Re'])

		print("Qubit: ", self.alfa['Re'], "+", self.alfa['Im'], "i * |0> + ", self.beta['Re'], "+", self.beta['Im'], "i * |1>")
		print("theta: ", theta * 180 / np.pi)
		print("phi: ", phi * 180 / np.pi)

		return theta, phi
	
	def __matmul__(self, other):
		if isinstance(other, np.ndarray):
			if other.shape != (2, 2):
				raise ValueError("Matrix must be 2x2 for qubit operations")
			
			# Convert qubit state to numpy array
			state_np = np.array([
				self.alfa['Re'] + 1j * self.alfa['Im'],
				self.beta['Re'] + 1j * self.beta['Im']
			])

			# Perform matrix multiplication
			result = state_np @ other  # Changed the order here

			# Convert result back to ComplexNumbers
			new_alfa = {'Re': result[0].real, 'Im': result[0].imag}
			new_beta = {'Re': result[1].real, 'Im': result[1].imag}
			return Qubit(new_alfa, new_beta)
		else:
			raise TypeError("Unsupported operand type for @. Expected numpy array.")

	def __rmatmul__(self, other):
		raise TypeError("Matrix multiplication with Qubit must have Qubit as the left operand")

	def __str__(self):
		return f"({self.alfa['Re']} + {self.alfa['Im']}i)|0> + ({self.beta['Re']} + {self.beta['Im']}i)|1>"
		
    
class PauliX():			
	def apply(self, qubit: Qubit):
		theta = np.pi
		matrix = np.array([
			[np.cos(theta/2), -1j * np.sin(theta/2)],
			[-1j * np.sin(theta/2), np.cos(theta/2)]
		], dtype=np.complex128)
		
		# Handle numerical precision for cos(π/2)
		matrix = np.round(matrix, decimals=10)
		
		new_qubit: Qubit = qubit @ matrix

		return new_qubit

	def getArcPointsForRotation(self, start: Tuple[float, float, float], end: Tuple[float, float, float], num_points: int = 100):
		start = np.array(start)
		end = np.array(end)

		# For Pauli X rotation, we rotate around the X-axis
		# The X coordinate remains constant, while Y and Z change
		t = np.linspace(0, 1, num_points)
     
		
		# Normalize start and end points
		start = start / np.linalg.norm(start)
		end = end / np.linalg.norm(end)
		
		# Calculate the angle between start and end
		angle = np.arccos(np.dot(start, end))
		
		arc_points = []
 
		for t_i in t:
			rotation_matrix = np.array([
				[1, 0, 0],
				[0, np.cos(angle * t_i), -np.sin(angle * t_i)],
				[0, np.sin(angle * t_i), np.cos(angle * t_i)]
			])
			point = rotation_matrix @ start
			arc_points.append(tuple(point))
        
		return arc_points

class PauliY():
	def apply(self, qubit: Qubit):
		theta = np.pi
		matrix = np.array([
			[np.cos(theta/2), -np.sin(theta/2)],
			[np.sin(theta/2), np.cos(theta/2)]
		], dtype=np.complex128)
		
		# Handle numerical precision for cos(π/2)
		matrix = np.round(matrix, decimals=10)
		
		new_qubit: Qubit = qubit @ matrix

		return new_qubit
    
	def getArcPointsForRotation(self, start: Tuple[float, float, float], end: Tuple[float, float, float], num_points: int = 100):
		start = np.array(start)
		end = np.array(end)

		# For Pauli Y rotation, we rotate around the Y-axis
		# The Y coordinate remains constant, while X and Z change
		t = np.linspace(0, 1, num_points)
		
		# Normalize start and end points
		start = start / np.linalg.norm(start)
		end = end / np.linalg.norm(end)
		
		# Calculate the angle between start and end
		angle = np.arccos(np.dot(start, end))
		
		arc_points = []
		
		for t_i in t:
			rotation_matrix = np.array([
                [np.cos(angle * t_i), 0, np.sin(angle * t_i)],
                [0, 1, 0],
                [-np.sin(angle * t_i), 0, np.cos(angle * t_i)]
            ])
			
		point = rotation_matrix @ start
		arc_points.append(tuple(point))
			
		return arc_points
		
        
        
class PauliZ():
    def apply(self, qubit: Qubit):
        theta = np.pi
        matrix = np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
        
        # Handle numerical precision for cos(π/2)
        matrix = np.round(matrix, decimals=10)
        


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

		# |0>
		qubit1: Qubit = Qubit({'Re': 1, 'Im': 0}, {'Re': 0, 'Im': 0})
		self.drawQubit(qubit1, 'r')
		
		# |0> -> |1>
		gate = PauliX()
		qubit2 = gate.apply(qubit1)
		self.drawQubit(qubit2, 'g')
  
		qubits = [qubit1, qubit2]
		x1, y1, z1 = self.getXYZForQubit(qubits[0])	
		x2, y2, z2 = self.getXYZForQubit(qubits[1])

		print("x1, y1, z1: ", x1, y1, z1)
		print("x2, y2, z2: ", x2, y2, z2)
  
		arc_points = gate.getArcPointsForRotation((x1, y1, z1), (x2, y2, z2))
		self.draw_great_circle_arc(arc_points)
  
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

	def transformAnglesToSphereCoordinates(self, theta, phi):
		#The calculated theta angle is good, but we need to transform it by -90 degrees to match the Bloch Sphere
  
		x = np.cos(phi) * np.sin(theta)
		y = np.sin(phi) * np.sin(theta)
		z = np.cos(theta)

		#Round the x, y, z to 10 decimal places
		x = round(x, 10)
		y = round(y, 10)
		z = round(z, 10)
  
		return x, y, z

	def getXYZForQubit(self, qubit: Qubit):
		theta, phi = qubit.getAnglesForQubit()
		x, y, z = self.transformAnglesToSphereCoordinates(theta, phi)
		return x, y, z
 
	def draw_great_circle_arc(self, arc_points):
		arc_array = np.array(arc_points)
		self.ax.plot(arc_array[:, 0], arc_array[:, 1], arc_array[:, 2], color='b', linewidth=2)
		self.canvas.draw()
	

	def drawQubit(self, qubit: Qubit, color: str):
		theta, phi = qubit.getAnglesForQubit()
		x, y, z = self.transformAnglesToSphereCoordinates(theta, phi)
		self.drawQuiver(x, y, z, color)


#----------------------------------

#--------HELPER FUNCTIONS----------

#----------------------------------

def getArcPointsForSpecificAxis(t, start, end, axis, num_points=100):
    start = np.array(start)
    end = np.array(end)
    
    # Normalize start and end points
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    
    # Calculate the angle between start and end
    angle = np.arccos(np.dot(start, end))
    
    arc_points = []
    for t_i in t:
        if axis == 'X':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle * t_i), -np.sin(angle * t_i)],
                [0, np.sin(angle * t_i), np.cos(angle * t_i)]
            ])
        elif axis == 'Y':
            rotation_matrix = np.array([
                [np.cos(angle * t_i), 0, np.sin(angle * t_i)],
                [0, 1, 0],
                [-np.sin(angle * t_i), 0, np.cos(angle * t_i)]
            ])
        elif axis == 'Z':
            rotation_matrix = np.array([
                [np.cos(angle * t_i), -np.sin(angle * t_i), 0],
                [np.sin(angle * t_i), np.cos(angle * t_i), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Invalid axis. Must be 'X', 'Y', or 'Z'.")
        
        point = rotation_matrix @ start
        arc_points.append(tuple(point))
    
    return arc_points
   


 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BlochSphereSimulator()
    main_window.show()
    sys.exit(app.exec_())