import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, TypedDict
from qubit import Qubit


class Gate():

    def __init__(self, rotation_angle: float = np.pi):
        self.rotation_angle = rotation_angle

    def apply(self, qubit: Qubit):
        pass


class PauliX(Gate):
    def apply(self, qubit: Qubit):
        theta = self.rotation_angle
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
        angle = self.rotation_angle

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


class PauliY(Gate):
    def apply(self, qubit: Qubit):
        theta = -self.rotation_angle
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
        angle = self.rotation_angle

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


class PauliZ(Gate):
    def apply(self, qubit: Qubit):
        theta = self.rotation_angle
        matrix = np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ], dtype=np.complex128)

        # Handle numerical precision for cos(π/2)
        matrix = np.round(matrix, decimals=10)

        new_qubit: Qubit = qubit @ matrix

        return new_qubit

    def getArcPointsForRotation(self, start: Tuple[float, float, float], end: Tuple[float, float, float], num_points: int = 100):
        start = np.array(start)
        end = np.array(end)

        # For Pauli Z rotation, we rotate around the Z-axis
        # The Z coordinate remains constant, while X and Y change
        t = np.linspace(0, 1, num_points)

        # Normalize start and end points
        start = start / np.linalg.norm(start)
        end = end / np.linalg.norm(end)

        # Calculate the angle between start and end
        angle = self.rotation_angle

        arc_points = []

        for t_i in t:
            rotation_matrix = np.array([
                [np.cos(angle * t_i), -np.sin(angle * t_i), 0],
                [np.sin(angle * t_i), np.cos(angle * t_i), 0],
                [0, 0, 1]
            ])
            point = rotation_matrix @ start
            arc_points.append(tuple(point))

        return arc_points


class Hadamard(Gate):
    def apply(self, qubit: Qubit):
        # Hadamard matrix
        matrix = (1 / np.sqrt(2)) * np.array([
            [1, 1],
            [1, -1]
        ], dtype=np.complex128)

        # Apply the gate
        new_qubit: Qubit = qubit @ matrix

        return new_qubit

    def getArcPointsForRotation(self, start: Tuple[float, float, float], end: Tuple[float, float, float], num_points: int = 100):
        start = np.array(start)
        end = np.array(end)

        # Normalize start and end points
        start = start / np.linalg.norm(start)
        end = end / np.linalg.norm(end)

        # For Hadamard, we rotate around the y=x axis in the x-z plane
        # This is equivalent to rotating around the [1, 1, 0] axis
        axis = np.array([1, 0, 1]) / np.sqrt(2)

        # The rotation angle for Hadamard is always pi
        angle = np.pi

        t = np.linspace(0, 1, num_points)
        arc_points = []

        for t_i in t:
            # Create rotation matrix for rotation around [1,1,0] axis
            c = np.cos(angle * t_i)
            s = np.sin(angle * t_i)
            rotation_matrix = np.array([
                [(1+c)/2, -s/np.sqrt(2), (1-c)/2],
                [s/np.sqrt(2), c, -s/np.sqrt(2)],
                [(1-c)/2, s/np.sqrt(2), (1+c)/2]
            ])
            point = rotation_matrix @ start
            arc_points.append(tuple(point))

        return arc_points
