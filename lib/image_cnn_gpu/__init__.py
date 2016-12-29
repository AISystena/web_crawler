import sys
import os
# __init__.pyが置いてあるフォルダをimport対象にする
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + os.pardir + '/db_manager'))
