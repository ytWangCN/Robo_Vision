import os

dir = '/home/wyt/2'   #图片文件的地址
label = 2
#os.listdir的结果就是一个list集，可以使用list的sort方法来排序。如果文件名中有数字，就用数字的排序
files = os.listdir(dir)#列出dirname下的目录和文件
files.sort(key=lambda x: int(x.split('.')[0]))#排序
train = open('./TrainNum2.txt','a')
text = open('./TestNum2.txt', 'a')
i = 0
for file in files:
    if i<2000:
        fileType = os.path.split(file)#os.path.split()：按照路径将文件名和路径分割开
        if fileType[1] == '.txt':
            continue
        name =  str(dir) + '/' +  file + ' ' + str(int(label)) +'\n'
        train.write(name)
        i = i+1       
    else:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = str(dir) + '/' + file + ' ' + str(int(label)) +'\n'
        text.write(name)
        i = i+1
text.close()
train.close()
    
