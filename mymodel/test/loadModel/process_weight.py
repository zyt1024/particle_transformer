import os, sys
import fileinput
path = "/home/atzyt/Project/myparticle/particle_transformer/mymodel/test/loadModel/weight"

file_names  = os.listdir( path )

# 输出所有文件和文件夹
for file_name in file_names :
   file_path = os.path.join(path,file_name)
   print(file_path)
   with fileinput.input(file_path, inplace=True) as file:
            for line in file:
                # 向文件中添加一行数据
                if file.isfirstline():
                    print('#include "data_type.h"')
                line = line.replace("{,","{").replace(",}","}")
                print(line, end='')