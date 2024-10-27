import unittest
import numpy as np
from qubit import Qubit, qubit0, qubit1, qubitPlus, qubitMinus
from gates import Hadamard

class TestHadamardGate(unittest.TestCase):
    def setUp(self):
        self.hadamard = Hadamard()
        self.tolerance = 1e-10 #this is the tolarence for the floating point comparisons
        
    def test_apply_to_basis_states(self):
        # Test |0⟩ -> |+⟩
        result = self.hadamard.apply(qubit0)
        expected = qubitPlus
        self.assertTrue(np.allclose(result.getState(), expected.getState(), atol=self.tolerance))

        # Test |1⟩ -> |-⟩
        result = self.hadamard.apply(qubit1)
        expected = qubitMinus
        self.assertTrue(np.allclose(result.getState(), expected.getState(), atol=self.tolerance))
    
    
if __name__ == '__main__':
    unittest.main(verbosity=2)