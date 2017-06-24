import  re
s=r'ABC-\001'
if re.match("^\d{3}-\d{3,8}","001-00"):
    print("ok")
else:
    print("fail")

s=r'a vc   d'
print(re.split(r'\s+',s))


s=r'a vc   d,d,e;f'
print(re.split(r'[\s+\,\;]+',s))

m=re.match("^(\d{3})-(\d{3,8})","001-007")
print(m.group(1))
print(m.group(2))

compile = re.compile("^(\d{3})-(\d{3,8})")
print(compile.match("001-007").group(2))