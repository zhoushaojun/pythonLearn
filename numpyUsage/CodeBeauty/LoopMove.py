#循环移位
data=[1,2,3,4,"a","b","c","d"]

def move(data, start, end):
    length = len(data)
    data[start:end]=data[end-length-1:start-length-1:-1]

def moveHandler(right,data):
    length =len(data)
    right = right % length
    move(data,0,length -right)
    move(data, length-right, length)
    move(data, 0, length)

moveHandler(3,data)
print(data)


