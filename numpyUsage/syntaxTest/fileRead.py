with open('C:\\Users\\zhoushaojun\\Desktop\\txt.txt','r', encoding="gbk", errors="ignore") as f:
    #print(f.readline())
    #for lines in f.readlines():
     #   print(lines)
     print(type(f))
    TextIOWrapper
     for user in f.readlines():
         print(user.strip("\n"))

'''
with open('C:\\Users\\zhoushaojun\\Desktop\\txt.txt','a', encoding="gbk", errors="ignore") as w:
    w.write("12\n\r")


#stirngio
from io import StringIO

f=StringIO()
f.write("1")
f.write("2")
print(f.getvalue())
while True:
    s= f.readline()
    if s=='':
        break
    print(s)


#byteio
from io import BytesIO
b=BytesIO()
b.write("中文".encode("utf-8"))
print(b.getvalue())


#os
import os
print(os.name)
print(os.environ)
print(os.path.abspath("."))
path = os.path.join("/a/b","c")
print(path)
print(os.path.split("/a/b/c/d"))

ret = [x for x in os.listdir(".") if os.path.isfile(x)]
print(ret)

ret = [x for x in os.listdir(".") if os.path.isdir(x)]
print(ret)

ret = [x for x in os.listdir(".") if os.path.splitext(x)[1]==".py"]
print(ret)


'''