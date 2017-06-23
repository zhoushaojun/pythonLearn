#查找最大递增子序列
from operator import *
mapData={}
data=[1,-1,2,-3,4,-5,6,-7]

def maxSequence(data, mapData):
    index=0
    for item in data:
        index=index+1
        if not mapData:
            list=[]
            list.append(item)
            mapData[index]= list
            continue

        indicate=False
        for (key,value) in mapData.items():
            if(item > value[-1]):
                value.append(item)
                indicate=True
                break
        if(not indicate):
            list = []
            list.append(item)
            mapData[index] = list


maxSequence(data, mapData)

for key,value in mapData.items():
    print(key,value)

result = sorted(mapData.items(), key=lambda d:len(d[1]), reverse=True)
print(result[0][1])


