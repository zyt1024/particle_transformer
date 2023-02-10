import sys,os
# sys.path.append('需要作为模块引入的路径')
# 添加当前路径的前一级文件作为源文件夹
path = os.path.dirname(os.path.dirname(__file__)) 
print(path)
sys.path.append(path)
