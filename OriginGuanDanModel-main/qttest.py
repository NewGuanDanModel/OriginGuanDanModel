from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

import sys

if __name__ == "__main__":
    pass
    a = QApplication(sys.argv)
    w = QMainWindow()
    w.setGeometry(200, 200, 200, 200)
    
    w.show()
    sys.exit(a.exec_())