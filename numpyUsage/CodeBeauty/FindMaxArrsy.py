import sys

# 寻找子数组中最大值
data = [-12, 5, 3, -6, 4, -8, 6]


def findMaxArrayLoop(data, start, end):
    if start == end:
        return (start, end, data[start])
    elif start > end:
        return (0, 0, -1000)

    mid = (start + end) // 2
    (leftStart, leftEnd, leftMax) = findMaxArrayLoop(data, start, mid)
    (rightStart, rightEnd, rightMax) = findMaxArrayLoop(data, mid + 1, end)
    maxStart =0
    maxEnd=0
    maxValue=0
    if leftMax > rightMax:
        maxStart = leftStart
        maxEnd = leftEnd
        maxValue = leftMax
    else:
        maxStart = rightStart
        maxEnd = rightEnd
        maxValue = rightMax

    if leftEnd == mid and rightStart == mid + 1:
        if maxValue < leftMax + rightMax:
            maxStart = leftStart
            maxEnd = rightEnd
            maxValue = leftMax+rightMax
    print(maxStart,maxEnd,maxValue)
    return (maxStart, maxEnd, maxValue)


result = findMaxArrayLoop(data, 0, 3)
print(result)
