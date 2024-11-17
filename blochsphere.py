import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, TypedDict
from qubit import Qubit, qubit0, qubit1, qubitPlus, qubitMinus
from gates import PauliX, PauliY, PauliZ, Gate, Hadamard


class BlochSpherePlot(QMainWindow):
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

        # Ploting out the Block Sphere with the initial state
        self.initialState = qubit0
        self.plot_bloch_sphere()
        self.drawQubit(self.initialState, 'r')

        # Should rotate from |0⟩ to |+⟩
        # self.applyGates(qubit0, [PauliY(np.pi/4), PauliX(np.pi/4), Hadamard()])
        # self.ax = None

        # Set the default gates to an empty list
        self.gates = []
        # Adding history
        self.state_history = [(qubit0, [])]  # List of (state, gates) tuples
        self.current_state_index = 0

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

        # Draw the sphere
        # This creates a grid for the sphere -> it means that we are going to have 20 points in the theta direction and 10 points in the phi direction
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)

        # Plot the sphere
        self.ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)

        # Draw coordinate axes
        # X axis
        self.ax.plot([-1, 1], [0, 0], [0, 0], 'black', linewidth=1, zorder=1)

        # Y axis
        self.ax.plot([0, 0], [-1, 1], [0, 0], 'black', linewidth=1, zorder=1)

        # Z axis
        self.ax.plot([0, 0], [0, 0], [-1, 1], 'black', linewidth=1, zorder=1)

        # Add small axis labels near the origin
        self.ax.text(0, 0, 1.2, '+z', color='black', fontsize=10)
        self.ax.text(0, 0, -1.2, '-z', color='black', fontsize=10)
        self.ax.text(0, 1.2, 0, '+y', color='black', fontsize=10)
        self.ax.text(0, -1.2, 0, '-y', color='black', fontsize=10)
        self.ax.text(1.2, 0, 0, 'x', color='black', fontsize=10)
        self.ax.text(-1.2, 0, 0, '-x', color='black', fontsize=10)

        # Set the limits for the axes

        self.add_state_labels()
        self.canvas.draw()

    def drawQuiver(self, x, y, z, color: str):
        self.ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.1)
        self.canvas.draw()

    # Addin lbales to the spehre
    def add_state_labels(self):
        """Add state labels inside the Bloch sphere"""
        # Add |0⟩ and |1⟩ labels (on z-axis)
        # Moved from ±1.2 to ±0.9 and slightly offset from axis for visibility
        self.ax.text(0.1, 0, 0.9, '$|0\\rangle$',
                     fontsize=12, ha='left', va='bottom')
        self.ax.text(0.1, 0, -0.9, '$|1\\rangle$',
                     fontsize=12, ha='left', va='top')

        # Add |+⟩ and |-⟩ labels (on x-axis)
        # Moved from ±1.2 to ±1.0 and slightly offset from axis for visibility
        self.ax.text(0.9, 0, 0.1, '$|+\\rangle$',
                     fontsize=12, ha='right', va='bottom')
        self.ax.text(-0.9, 0, 0.1, '$|-\\rangle$',
                     fontsize=12, ha='left', va='bottom')

    def draw_great_circle_arc(self, arc_points):
        arc_array = np.array(arc_points)
        arc_array = arc_array / \
            np.linalg.norm(arc_array, axis=1)[:, np.newaxis]
        self.ax.plot(arc_array[:, 0], arc_array[:, 1],
                     arc_array[:, 2], color='b', linewidth=2)
        self.canvas.draw()

    def drawQubit(self, qubit: Qubit, color: str):
        theta, phi = qubit.getAnglesForQubit()
        x, y, z = qubit.transformAnglesToSphereCoordinates(theta, phi)
        self.drawQuiver(x, y, z, color)

    def applyGates(self, qubit: Qubit, gates: List[Gate]):
        # Draw the initial qubit
        self.drawQubit(qubit, 'r')
        final_qubit = qubit
        for gate in gates:
            # Get the qubit's coordinates before applying the gate -> Copy of the qubit
            previous_qubit = qubit

            # Apply the gate
            qubit = gate.apply(qubit)
            final_qubit = qubit

            # Draw the arc for the gate
            print(
                f"Previous qubit position {previous_qubit.getXYZ()} with angles {qubit.getAnglesForQubitInDegrees()}")
            print(
                f"Next qubit position {final_qubit.getXYZ()} with angles {final_qubit.getAnglesForQubitInDegrees()}")
            arc_points = gate.getArcPointsForRotation(
                previous_qubit.getXYZ(), qubit.getXYZ())
            self.draw_great_circle_arc(arc_points)

        # Draw the final qubit
        self.drawQubit(final_qubit, 'b')

    def applyAddedGates(self):
        self.plot_bloch_sphere()
        final_qubit = self.initialState
        qubit = self.initialState
        self.drawQubit(self.initialState, 'r')
        for gate in self.gates:
            # Get the qubit's coordinates before applying the gate -> Copy of the qubit
            previous_qubit = qubit

            # Apply the gate
            qubit = gate.apply(qubit)
            final_qubit = qubit

            # Draw the arc for the gate
            print(
                f"Previous qubit position {previous_qubit.getXYZ()} with angles {qubit.getAnglesForQubitInDegrees()}")
            print(
                f"Next qubit position {final_qubit.getXYZ()} with angles {final_qubit.getAnglesForQubitInDegrees()}")
            arc_points = gate.getArcPointsForRotation(
                previous_qubit.getXYZ(), qubit.getXYZ())
            self.draw_great_circle_arc(arc_points)

        # Draw the final qubit
        self.drawQubit(final_qubit, 'b')

    def setInitialState(self, qubit: Qubit):
        self.initialState = qubit
        self.drawQubit(self.initialState, 'r')
        self.gates = []
        self.state_history = [(qubit, [])]

    def addGate(self, gate: Gate):
        self.gates.append(gate)
        self.state_history = self.state_history[:self.current_state_index + 1]
        self.state_history.append((self.initialState, self.gates.copy()))
        self.current_state_index += 1
        
    def undo(self):
        if self.current_state_index > 0:
            self.current_state_index -= 1
            self.initialState, self.gates = self.state_history[self.current_state_index]
            self.figure.clear()
            self.plot_bloch_sphere()
            self.applyAddedGates()

    def redo(self):
        if self.current_state_index < len(self.state_history) - 1:
            self.current_state_index += 1
            self.initialState, self.gates = self.state_history[self.current_state_index]
            self.figure.clear()
            self.plot_bloch_sphere()
            self.applyAddedGates()
