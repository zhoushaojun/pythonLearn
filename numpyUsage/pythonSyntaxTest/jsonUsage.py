import  json

data ={"a":'1', "b":'2'}
jdata = json.dumps(data)
print(jdata)
print(type(jdata))
jmap = json.loads(jdata)
print(jmap)
print(type(jmap))


class Student(object):
    val=10
    def __init__(self,name,age):
        self.__name__=name
        self.__age__=age
    @property
    def name(self):
        return self.__name__
    @property
    def age(self):
        return self.__age__

    @staticmethod
    def instance2Json(std):
        return {"name":std.name,"age":std.age}

    @staticmethod
    def json2Instance(d):
        return Student(d['name'],d['age'])

    def selfClass(self):
        print("self class")
        return {"name":std.name,"age":std.age}

    @staticmethod
    def test():
        print("self class")

    @classmethod
    def classtest(cls):
        cls.test()
        print(cls.val)

std=Student("zhou","20")
jstd= json.dumps(std,default=Student.instance2Json)
stdNew = json.loads(jstd,object_hook=Student.json2Instance)
print(stdNew.name)
print(stdNew.age)

