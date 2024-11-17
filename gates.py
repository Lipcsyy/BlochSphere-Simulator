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


class UnitaryGate(Gate):
    def __init__(self, matrix: np.ndarray, tolerance: float = 1e-3):
        super().__init__()
        # Verify the matrix is 2x2
        if matrix.shape != (2, 2):
            raise ValueError("Unitary matrix must be 2x2")

        # Verify the matrix is unitary
        # A unitary matrix must satisfy U†U = UU† = I
        identity = np.eye(2)  # Creating identity matrix
        # Multiply matrix by its conjugate transpose
        u_dagger_u = matrix @ matrix.conj().T
        u_u_dagger = matrix.conj().T @ matrix  # Multiply conjugate transpose by matrix

        if not (np.allclose(u_dagger_u, identity, rtol=tolerance, atol=tolerance) and
                np.allclose(u_u_dagger, identity, rtol=tolerance, atol=tolerance)):
            raise ValueError(f"""Matrix is not unitary within tolerance {tolerance}.
                U†U deviation from identity: {np.max(np.abs(u_dagger_u - identity))}
                UU† deviation from identity: {np.max(np.abs(u_u_dagger - identity))}""")

        self.matrix = matrix

        # Calculate rotation angle from trace
        trace = np.trace(matrix)  # Summing up the diagonal elements
        # Getting the real part of the trace, dividing by two to get cos(θ/2) and than calculating the rotation angle
        self.rotation_angle = 2 * np.arccos(np.clip(np.real(trace) / 2, -1, 1))

        # Calculate rotation axis using quaternion method
        # U = cos(θ/2)I - i sin(θ/2)(rx*X + ry*Y + rz*Z)
        # I is identity matrix [[1,0],[0,1]]
        # X is Pauli-X [[0,1],[1,0]]
        # Y is Pauli-Y [[0,-i],[i,0]]
        # Z is Pauli-Z [[1,0],[0,-1]]
        if np.abs(self.rotation_angle) < tolerance:
            self.rotation_axis = np.array([0, 0, 1])
        else:
            # Extract the components that determine the rotation axis
            rx = np.imag(matrix[1, 0] + matrix[0, 1]) / \
                np.sin(self.rotation_angle/2)
            ry = np.real(matrix[1, 0] - matrix[0, 1]) / \
                np.sin(self.rotation_angle/2)
            rz = np.imag(matrix[0, 0] - matrix[1, 1]) / \
                np.sin(self.rotation_angle/2)

            self.rotation_axis = np.array([rx, ry, rz])
            norm = np.linalg.norm(self.rotation_axis)  # Normalizing

            if norm > tolerance:
                self.rotation_axis = self.rotation_axis / norm
            else:
                # Default to Z-axis if no clear axis is determined
                self.rotation_axis = np.array([0, 0, 1])

    def apply(self, qubit: Qubit):
        return qubit @ self.matrix

    def getArcPointsForRotation(self, start: Tuple[float, float, float], end: Tuple[float, float, float], num_points: int = 100):
        start = np.array(start)
        end = np.array(end)

        # Normalize start and end points
        start = start / np.linalg.norm(start)
        end = end / np.linalg.norm(end)

        # Get the rotation axis perpendicular to both start and end points
        if np.allclose(start, end) or np.allclose(start, -end):
            rotation_axis = self.rotation_axis
        else:
            # Use the cross product to find the rotation axis between points
            rotation_axis = np.cross(start, end)
            if np.linalg.norm(rotation_axis) < 1e-10:
                rotation_axis = self.rotation_axis
            else:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        t = np.linspace(0, 1, num_points)
        arc_points = []

        for t_i in t:
            angle = self.rotation_angle * t_i
            cos_t = np.cos(angle)
            sin_t = np.sin(angle)

            # Rodrigues rotation formula with better numerical stability
            R = np.eye(3) + \
                sin_t * np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                  [rotation_axis[2], 0, -rotation_axis[0]],
                                  [-rotation_axis[1], rotation_axis[0], 0]]) + \
                (1 - cos_t) * (np.outer(rotation_axis, rotation_axis) - np.eye(3))

            point = R @ start
            # Ensure the point stays normalized
            point = point / np.linalg.norm(point)
            arc_points.append(tuple(point))

        return arc_points


class SGate(Gate):
    def apply(self, qubit: Qubit):
        # S gate matrix = [[1, 0], [0, i]]
        matrix = np.array([
            [1, 0],
            [0, 1j]
        ], dtype=np.complex128)

        # Apply the gate
        new_qubit: Qubit = qubit @ matrix

        return new_qubit

    def getArcPointsForRotation(self, start: Tuple[float, float, float], end: Tuple[float, float, float], num_points: int = 100):
        start = np.array(start)
        end = np.array(end)

        # For S gate rotation, we rotate around the Z-axis by π/2
        # The Z coordinate remains constant, while X and Y change
        t = np.linspace(0, 1, num_points)

        # Normalize start and end points
        start = start / np.linalg.norm(start)
        end = end / np.linalg.norm(end)

        # Fixed π/2 rotation for S gate
        angle = np.pi/2
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


class TGate(Gate):
    def __init__(self):
        # T gate is a fixed π/4 rotation, no angle parameter needed
        super().__init__(rotation_angle=np.pi/4)

    def apply(self, qubit: Qubit):
        # T gate matrix = [[1, 0], [0, e^(iπ/4)]]
        matrix = np.array([
            [1, 0],
            [0, np.exp(1j * np.pi/4)]
        ], dtype=np.complex128)

        # Apply the gate
        new_qubit: Qubit = qubit @ matrix

        return new_qubit

    def getArcPointsForRotation(self, start: Tuple[float, float, float], end: Tuple[float, float, float], num_points: int = 100):
        start = np.array(start)
        end = np.array(end)

        # The Z coordinate remains constant during T gate rotation
        t = np.linspace(0, 1, num_points)

        # Normalize start and end points
        start = start / np.linalg.norm(start)
        end = end / np.linalg.norm(end)

        # T gate is a π/4 rotation around Z axis
        angle = np.pi/4
        arc_points = []

        for t_i in t:
            # Rotation matrix for Z-axis rotation
            rotation_matrix = np.array([
                [np.cos(angle * t_i), -np.sin(angle * t_i), 0],
                [np.sin(angle * t_i), np.cos(angle * t_i), 0],
                [0, 0, 1]
            ])
            point = rotation_matrix @ start
            arc_points.append(tuple(point))

        return arc_points

    def __str__(self):
        return "T Gate (π/4 phase rotation)"


class TDaggerGate(Gate):
    def __init__(self):
        # T† gate is a fixed -π/4 rotation
        super().__init__(rotation_angle=-np.pi/4)

    def apply(self, qubit: Qubit):
        # T† gate matrix = [[1, 0], [0, e^(-iπ/4)]]
        matrix = np.array([
            [1, 0],
            [0, np.exp(-1j * np.pi/4)]
        ], dtype=np.complex128)

        return qubit @ matrix

    def getArcPointsForRotation(self, start: Tuple[float, float, float], end: Tuple[float, float, float], num_points: int = 100):
        start = np.array(start)
        end = np.array(end)

        t = np.linspace(0, 1, num_points)

        # Normalize start and end points
        start = start / np.linalg.norm(start)
        end = end / np.linalg.norm(end)

        # T† gate is a -π/4 rotation around Z axis (counterclockwise)
        angle = -np.pi/4
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

    def __str__(self):
        return "T† Gate (-π/4 phase rotation)"


class SDaggerGate(Gate):
    def __init__(self):
        # S† gate is a fixed -π/2 rotation
        super().__init__(rotation_angle=-np.pi/2)

    def apply(self, qubit: Qubit):
        # S† gate matrix = [[1, 0], [0, -i]]
        matrix = np.array([
            [1, 0],
            [0, -1j]
        ], dtype=np.complex128)

        return qubit @ matrix

    def getArcPointsForRotation(self, start: Tuple[float, float, float], end: Tuple[float, float, float], num_points: int = 100):
        start = np.array(start)
        end = np.array(end)

        t = np.linspace(0, 1, num_points)

        # Normalize start and end points
        start = start / np.linalg.norm(start)
        end = end / np.linalg.norm(end)

        # S† gate is a -π/2 rotation around Z axis (counterclockwise)
        angle = -np.pi/2
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

    def __str__(self):
        return "S† Gate (-π/2 phase rotation)"
