#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:48:44 2024

@author: lipcseymarton
"""
from PyQt5.QtWidgets import QApplication
import sys
from BlochSphereGUI import BlochSphereGUI


def main():
    app = QApplication(sys.argv)

    gui = BlochSphereGUI()

    gui.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = BlochSphereGUI()
    gui.show()
    sys.exit(app.exec_())
