import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from Setting import setting

directory1 = 'D08'
directory2 = 'all'

xlsx_file = True
Graph_image = True

if __name__ == "__main__":
    current_directory = os.getcwd()
    setting(directory1, directory2, current_directory)
