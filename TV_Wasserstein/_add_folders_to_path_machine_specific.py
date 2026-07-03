import sys

# Posso stampare i percorsi assoluti con
# from pathlib import Path
# print(Path.cwd()) # current working directory
# print(Path(__file__).parent) # parent folder from the command's script

path_0 = 'c:\\Users\\rodol\\Documents\\Codice\\PYTHON\\Libraries'
path_1 = 'c:\\Users\\rodol\\Documents\\Codice\\PYTHON\\TV_Wasserstein'

_DIRS = [path_0, path_1]

for _d in _DIRS:
    if _d not in sys.path:
        sys.path.insert(0, _d)