#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:48:50 2024

@author: lipcseymarton
"""
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QGridLayout, QMessageBox
import numpy as np
from blochsphere import BlochSpherePlot
from qubit import qubit0, qubit1, qubitPlus, qubitMinus
from gates import PauliX, PauliY, PauliZ, Hadamard, UnitaryGate, SGate, TGate, SDaggerGate, TDaggerGate


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
        for gate in ["X", "Y", "Z", "H", "S", "S†", "T", "T†"]:
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

        # Unitary matrix
        unitary_group = QWidget()
        unitary_layout = QVBoxLayout(unitary_group)
        unitary_layout.addWidget(QLabel("Unitary Matrix:"))

        # Matrix input fields
        self.matrix_inputs = []
        matrix_widget = QWidget()
        matrix_layout = QGridLayout(matrix_widget)

        for i in range(2):
            row_inputs = []
            for j in range(2):
                # Create input fields for real and imaginary parts
                real_input = QLineEdit()
                imag_input = QLineEdit()
                real_input.setPlaceholderText(f"Real({i},{j})")
                imag_input.setPlaceholderText(f"Imag({i},{j})")

                # Add to grid layout
                element_widget = QWidget()
                element_layout = QVBoxLayout(element_widget)
                element_layout.addWidget(real_input)
                element_layout.addWidget(imag_input)
                matrix_layout.addWidget(element_widget, i, j)

                row_inputs.append((real_input, imag_input))
            self.matrix_inputs.append(row_inputs)

        unitary_layout.addWidget(matrix_widget)

        # Add apply button
        apply_unitary_btn = QPushButton("Apply Unitary")
        apply_unitary_btn.clicked.connect(self.apply_unitary)
        unitary_layout.addWidget(apply_unitary_btn)

        # Add to main control layout
        control_layout.addWidget(unitary_group)

        # Add undo/redo buttons
        undo_redo_group = QWidget()
        undo_redo_layout = QHBoxLayout(undo_redo_group)

        # Undo button
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setShortcut("Ctrl+Z")  # Add keyboard shortcut
        self.undo_btn.clicked.connect(self.undo_last_operation)
        self.undo_btn.setEnabled(False)  # Initially disabled
        undo_redo_layout.addWidget(self.undo_btn)

        # Redo button
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setShortcut("Ctrl+Y")  # Add keyboard shortcut
        self.redo_btn.clicked.connect(self.redo_last_operation)
        self.redo_btn.setEnabled(False)  # Initially disabled
        undo_redo_layout.addWidget(self.redo_btn)

        # Add undo/redo buttons below gate controls
        control_layout.addWidget(undo_redo_group)

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
            "H": Hadamard,
            "S": SGate,
            "S†": SDaggerGate,
            "T": TGate,
            "T†": TDaggerGate
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
        # After applying gate, update undo/redo buttons
        self.update_undo_redo_buttons()

    def apply_unitary(self):
        """Apply the unitary matrix specified in the input fields"""
        try:
            # Create complex matrix from input fields
            matrix = np.zeros((2, 2), dtype=np.complex128)

            for i in range(2):
                for j in range(2):
                    real_input, imag_input = self.matrix_inputs[i][j]

                    # Evaluate real and imaginary parts as mathematical expressions
                    try:
                        real_str = real_input.text().strip()
                        imag_str = imag_input.text().strip()

                        real_part = self.evaluate_expression(
                            real_str) if real_str else 0.0
                        imag_part = self.evaluate_expression(
                            imag_str) if imag_str else 0.0

                        matrix[i, j] = real_part + 1j * imag_part

                    except ValueError as e:
                        raise ValueError(
                            f"Invalid input at position ({i},{j}): {str(e)}")

            # Create and apply the unitary gate
            gate = UnitaryGate(matrix)
            self.bloch_plot.addGate(gate)
            self.bloch_plot.figure.clear()
            self.bloch_plot.applyAddedGates()
            # After applying unitary, update undo/redo buttons
            self.update_undo_redo_buttons()

        except ValueError as e:
            # Show error message if matrix is not unitary or inputs are invalid
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText("Invalid Unitary Matrix")
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle("Error")
            error_dialog.exec_()

    def reset_simulation(self):
        """Reset the Bloch sphere to initial state"""
        self.bloch_plot.current_state_index = 0
        self.bloch_plot.state_history = [(qubit0, [])]
        self.current_state = qubit0
        self.bloch_plot.figure.clear()
        self.bloch_plot.plot_bloch_sphere()
        self.bloch_plot.setInitialState(self.current_state)
        self.update_undo_redo_buttons()

    def evaluate_expression(self, expr_str: str) -> float:
        """
        Evaluates a mathematical expression string containing pi, sqrt, and basic arithmetic.
        Supports: pi, sqrt(), +, -, *, /, (), and numerical values
        """
        if not expr_str or expr_str.isspace():
            return 0.0

        # Replace mathematical constants and functions
        expr_str = expr_str.lower().strip()
        expr_str = expr_str.replace('pi', str(np.pi))
        expr_str = expr_str.replace('π', str(np.pi))
        expr_str = expr_str.replace('sqrt', 'np.sqrt')
        expr_str = expr_str.replace('sin', 'np.sin')
        expr_str = expr_str.replace('cos', 'np.cos')

        # Create a safe dictionary of allowed functions
        safe_dict = {
            'np': np,
            'abs': abs,
            'float': float,
        }

        try:
            # Evaluate the expression in a restricted environment
            return float(eval(expr_str, {"__builtins__": {}}, safe_dict))
        except Exception as e:
            raise ValueError(
                f"Invalid expression: {expr_str}. Error: {str(e)}")

    def update_undo_redo_buttons(self):
        """Update the enabled state of undo/redo buttons"""
        self.undo_btn.setEnabled(self.bloch_plot.current_state_index > 0)
        self.redo_btn.setEnabled(
            self.bloch_plot.current_state_index < len(self.bloch_plot.state_history) - 1)

    def undo_last_operation(self):
        """Undo the last gate operation"""
        self.bloch_plot.undo()
        self.update_undo_redo_buttons()

    def redo_last_operation(self):
        """Redo the previously undone operation"""
        self.bloch_plot.redo()
        self.update_undo_redo_buttons()
