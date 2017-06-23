import numpy as np
map={'a':1, 'b':2}


for a,b in map.items():
    print(a,b)

for key in map.keys():
    print(key,map[key])

for value in map.values():
    print(value)

data=[1,2,3,4,5]

#倒排
print(data[::-1])

#最后一个元素
print(data[:-1])

a = np.array([[1,2,3], [4,5,6]])

#print(np.reshape(a,(2,3)))
print(np.reshape(a,(2,3),order='C'))
print(np.reshape(a,(1,-1),order='F'))


a= [1,2,3,4]

for _ in a:
    print(.1)