from complex import ComplexNumber
from typing import Tuple
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, TypedDict


class Qubit():
    def __init__(self, alfa: ComplexNumber, beta: ComplexNumber):
        self.alfa = alfa
        self.beta = beta

    def getAnglesForQubit(self):
        theta = 0
        phi = 0

        alfa_mag = np.sqrt(self.alfa['Re']**2 + self.alfa['Im']**2)
        beta_mag = np.sqrt(self.beta['Re']**2 + self.beta['Im']**2)

        theta = 2*np.arccos(alfa_mag)

        if beta_mag == 0:
            phi = 0
        elif alfa_mag == 0:
            phi = np.arctan2(self.beta['Im'], self.beta['Re'])
        else:
            phi = np.arctan2(self.beta['Im'], self.beta['Re']) - \
                np.arctan2(self.alfa['Im'], self.alfa['Re'])

        # Ensure phi is in the range [-pi, pi]
        phi = (phi + np.pi) % (2 * np.pi) - np.pi

        return theta, phi

    def getAnglesForQubitInDegrees(self):
        theta_rad, phi_rad = self.getAnglesForQubit()
        theta_deg = np.degrees(theta_rad)
        phi_deg = np.degrees(phi_rad)
        return theta_deg, phi_deg

    def getXYZ(self):
        theta, phi = self.getAnglesForQubit()
        x, y, z = self.transformAnglesToSphereCoordinates(theta, phi)
        return x, y, z

    def transformAnglesToSphereCoordinates(self, theta, phi):
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(theta)

        # Round the x, y, z to 10 decimal places
        # There are cases where the values are not exactly 0 or 1, but very close to it, so we need to round them to 0 or 1

        x = round(x, 10)
        y = round(y, 10)
        z = round(z, 10)

        # And there are cases when the value is -0.0, so we need to convert it to 0.0
        if x == -0.0:
            x = 0.0
        if y == -0.0:
            y = 0.0
        if z == -0.0:
            z = 0.0

        return x, y, z

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
            raise TypeError(
                "Unsupported operand type for @. Expected numpy array.")

    def __rmatmul__(self, other):
        raise TypeError(
            "Matrix multiplication with Qubit must have Qubit as the left operand")

    def __add__(self, other):
        return Qubit({'Re': self.alfa['Re'] + other.alfa['Re'], 'Im': self.alfa['Im'] + other.alfa['Im']}, {'Re': self.beta['Re'] + other.beta['Re'], 'Im': self.beta['Im'] + other.beta['Im']})

    def __sub__(self, other):
        return Qubit({'Re': self.alfa['Re'] - other.alfa['Re'], 'Im': self.alfa['Im'] - other.alfa['Im']}, {'Re': self.beta['Re'] - other.beta['Re'], 'Im': self.beta['Im'] - other.beta['Im']})

    def __truediv__(self, other):
        return Qubit({'Re': self.alfa['Re'] / other, 'Im': self.alfa['Im'] / other}, {'Re': self.beta['Re'] / other, 'Im': self.beta['Im'] / other})

    def __str__(self):
        return f"({self.alfa['Re']} + {self.alfa['Im']}i)|0> + ({self.beta['Re']} + {self.beta['Im']}i)|1>"

    def getState(self):
        return self.alfa, self.beta


qubit0 = Qubit({'Re': 1, 'Im': 0}, {'Re': 0, 'Im': 0})
qubit1 = Qubit({'Re': 0, 'Im': 0}, {'Re': 1, 'Im': 0})

qubitPlus = Qubit({'Re': 1/np.sqrt(2), 'Im': 0}, {'Re': 1/np.sqrt(2), 'Im': 0})
qubitMinus = Qubit({'Re': 1/np.sqrt(2), 'Im': 0},
                   {'Re': -1/np.sqrt(2), 'Im': 0})
