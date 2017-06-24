# 类的基本属性和配置
class Student(object):
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    def getAge(self):
        print(self.__age)

    def getName(self):
        print(self.__name)


student = Student('joe', '30')
# print(student.name,student.age)
student.__name = "zhou"
print(student.__name)
student.getAge()
student.getName()


# type isinstance使用
class Animal(object):
    def __init__(self, name):
        self.__name = name

    def run(self):
        print(self.__name, "is running")


class Dog(Animal):
    sex = 'man'

    def run(self):
        print("I am dog")
        super().run()
        super(Dog, self).run()
        Animal.run(self)


animal = Dog("dog")
animal.run()

print("=============20170519===============")
print(type(animal))
print(type(1))
print(type("abc"))
print(isinstance(animal, Dog))
print(isinstance([1, 2, 3], (list, tuple)))

print(hasattr(animal, "x"))
print(setattr(animal, 'x', 19))
print(getattr(animal, 'x'))

animal.sex = "win"
print(Dog.sex)
print(animal.sex)


# 多重继承 & 属性property age.setter
class UserFatherMixIn(object):
    def interface(self):
        print("I am interface")


class User(Animal, UserFatherMixIn):
    @property
    def age(self):
        return self._age;

    @age.setter
    def age(self, age):
        self._age = age


user = User('user')
user.age = 100
print(user.age)
print(user._age)
user.run()
user.interface()


# 定制类使用
class orderClass(object):
    def __init__(self, path=''):
        self.__path = path

    def __str__(self):
        return "my name is %s" % self.__path

    __repr__ = __str__

    def __getattr__(self, path):
        return orderClass('%s/%s' % (self.__path, path))

    def __call__(self, *args, **kwargs):
        print("you are calling")

myorder = orderClass('joe')
print(myorder)
print(myorder.a.b.c.d.e.f)
#调用call
myorder()


from enum import Enum,unique

@unique
class WeekDay(Enum):
    Sun=0
    Mon=1

print(WeekDay.Sun.name)
print(WeekDay.Sun.value)
for name,member in WeekDay.__members__.items():
    print(name,member.value)


import  logging
logging.basicConfig(level=logging.INFO)
#Exception
def bar(s):
    logging.info("here value %s" %s)
    print("cal value", s/0)

def main():
    try:
        bar(10)
    except Exception as e:
        print("Error",e)
        logging.exception(e)
        raise ValueError("zhou error")
    else:
        print("no error")
    finally:
        print("finally")

print("-------")
try:
    main()
except Exception as e:
    print("main exception",e)