import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from Setting import setting

directory0 = 'HY202103'
directory1 = 'all'              # 'D07', 'D08', 'D23', 'D24', 'all'
directory2 = 'all'              # ex) '20190715_190855' or 'all'

teststie = ['LMZO', 'LMZC']     # ['LMZO', 'LMZC']
xlsx_file = True                # True or False
graph_image = True              # True or False

if __name__ == "__main__":
    current_directory = os.getcwd()
    setting(directory0, directory1, directory2, current_directory, teststie, xlsx_file, graph_image)