# 寻找快速满足条件两个数
dataList = [5, 6, 1, 4, 7, 9, 8]


# print(dataList)


def heapSort(dataList, start, length):
    maxIndex = 0
    while start * 2 + 1 < length:
        # 查找子节点中最大值
        if start * 2 + 2 < length:
            maxIndex = start * 2 + 1 if dataList[start * 2 + 1] > dataList[start * 2 + 2] else start * 2 + 2
        if dataList[maxIndex] > dataList[start]:
            sumValue = dataList[maxIndex] + dataList[start]
            dataList[start] = sumValue - dataList[start]
            dataList[maxIndex] = sumValue - dataList[start]
            start = maxIndex;
        else:
            break


def heapAdjust(dataList):
    length = len(dataList);
    for i in reversed(range(length // 2)):
        heapSort(dataList, i, length)
    print(dataList)

    for i in reversed(range(length)):
        print(i)
        sumValue = dataList[0] + dataList[i]
        dataList[0] = sumValue - dataList[0]
        dataList[i] = sumValue - dataList[0]
        heapSort(dataList, 0, i)


# heapSort(dataList, 1, 7)
heapAdjust(dataList)
print(dataList)

def checkSum(dataList, value):
    start=0
    end=len(dataList)-1
    while start != end:
        checkValue = dataList[start] + dataList[end]
        print(start,end,checkValue)
        if checkValue == value:
            print("get it",start, end)
            break;
        elif checkValue > value:
            end-=1
        else:
            start+=1
checkSum(dataList, 13)
