import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand as rd

# Attenzione: prima avevo usato questo snippet, che NON funziona perché la lista sys.path NON è condivisa su script diversi
# import sys
# from pathlib import Path
# _THIS_FOLDER = Path(__file__).resolve().parent
# print(_THIS_FOLDER)
# if str(_THIS_FOLDER) not in sys.path:
#     sys.path.insert(0, str(_THIS_FOLDER))