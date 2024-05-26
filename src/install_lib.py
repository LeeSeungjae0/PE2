import subprocess
import sys

def install_all_library():
    # pip 설치
    subprocess.call([sys.executable, '-m', 'ensurepip', '--default-pip'])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

    # 필요한 라이브러리 설치
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'numpy'])
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'lmfit'])
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'pandas'])


    # 필요한 라이브러리 설치 확인
    import matplotlib
    import numpy
    import lmfit
    import pandas


    print(matplotlib.__version__)
    print(numpy.__version__)
    print(lmfit.__version__)
    print(pandas.__version__)
    print('install success')

install_all_library()


