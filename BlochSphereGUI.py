#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:48:50 2024

@author: lipcseymarton
"""
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
import numpy as np
from blochsphere import BlochSpherePlot
from qubit import qubit0, qubit1, qubitPlus, qubitMinus
from gates import PauliX, PauliY, PauliZ, Hadamard


class BlochSphereGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bloch Sphere Simulator")
        # Wider window to accommodate both
        self.setGeometry(100, 100, 1600, 800)

        # Create main widget with horizontal layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left side: Bloch Sphere
        self.bloch_plot = BlochSpherePlot()
        main_layout.addWidget(self.bloch_plot.canvas)

        # Right side: Control Panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # State mapping dictionary
        self.state_map = {
            "|0⟩": qubit0,
            "|1⟩": qubit1,
            "|+⟩": qubitPlus,
            "|-⟩": qubitMinus
        }

        # Initial state selection
        state_group = QWidget()
        state_layout = QHBoxLayout(state_group)
        state_layout.addWidget(QLabel("Initial State:"))

        self.state_buttons = []
        for state in ["|0⟩", "|1⟩", "|+⟩", "|-⟩"]:
            btn = QPushButton(state)
            btn.clicked.connect(
                lambda checked, s=state: self.set_initial_state(s))
            state_layout.addWidget(btn)
            self.state_buttons.append(btn)

        control_layout.addWidget(state_group)

        # Gate controls
        gates_group = QWidget()
        gates_layout = QVBoxLayout(gates_group)
        gates_layout.addWidget(QLabel("Gates:"))

        # Gate buttons
        self.gate_buttons = {}
        for gate in ["X", "Y", "Z", "H"]:
            gate_row = QWidget()
            row_layout = QHBoxLayout(gate_row)

            btn = QPushButton(f"Apply {gate}")
            btn.clicked.connect(lambda checked, g=gate: self.apply_gate(g))
            row_layout.addWidget(btn)

            if gate in ["X", "Y", "Z"]:
                angle_input = QLineEdit()
                angle_input.setPlaceholderText("Angle (degrees)")
                row_layout.addWidget(angle_input)
                self.gate_buttons[gate] = (btn, angle_input)
            else:
                self.gate_buttons[gate] = (btn, None)

            gates_layout.addWidget(gate_row)

        control_layout.addWidget(gates_group)

        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_simulation)
        control_layout.addWidget(reset_btn)

        # Add stretcher to keep controls at top
        control_layout.addStretch()

        # Initialize current state
        self.current_state = qubit0

        # Draw initial state
        # $self.bloch_plot.setInitialState(self.current_state)

    def set_initial_state(self, state):
        """Set initial state and update the visualization"""
        if state in self.state_map:
            self.bloch_plot.figure.clear()
            self.bloch_plot.plot_bloch_sphere()
            self.bloch_plot.setInitialState(self.state_map[state])

    def apply_gate(self, gate):
        """Apply the selected gate with given angle if applicable"""
        gate_map = {
            "X": PauliX,
            "Y": PauliY,
            "Z": PauliZ,
            "H": Hadamard
        }

        # Check if the gate we want to apply is valid
        if gate in gate_map:
            # Get the btn and the angle input
            btn, angle_input = self.gate_buttons[gate]

            # Convert the angle input from text to rad
            if angle_input and angle_input.text():
                try:
                    angle = float(angle_input.text())
                    # Convert degrees to radians
                    angle_rad = np.pi * angle / 180.0
                    gate_obj = gate_map[gate](angle_rad)
                except ValueError:
                    print("Invalid angle value")
                    return
            else:
                # Default to pi rotation if no angle specified
                gate_obj = gate_map[gate]()

            # Apply the gate
            self.bloch_plot.addGate(gate_obj)
            self.bloch_plot.figure.clear()
            self.bloch_plot.applyAddedGates()

    def reset_simulation(self):
        """Reset the Bloch sphere to initial state"""
        self.current_state = qubit0
        self.bloch_plot.figure.clear()
        self.bloch_plot.plot_bloch_sphere()
        self.bloch_plot.setInitialState(self.current_state)
