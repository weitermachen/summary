# 1.可变不可变类型
# 可变类型：值改变，ID不变，证明改的是原值，证明原值是可以改变的
# 不可变类型：值改变，ID也变了，证明是产生新的值，压根没有改变原值，证明原值是不可以被修改的
# 验证：所有的赋值操作都是产生新的空间绑定
# 1.1 int是不可变类型
# x = 10
# print(id(x))
# x = 11  # 产生新值
# print(id(x))

# 1.2 float是不可变类型
# x = 3.1
# print(id(x))
# x = 3.2  # 产生新值
# print(id(x))

# 1.3 str是不可变类型
# x = 'abc'
# print(id(x))
# x = 'ggg'  # 产生新值
# print(id(x))

# 小结：int、float、str都被设计成了不可分割的整体，不能够被改变

# 1.4 list是可变类型
# l = ['aaa', 'bbb', 'ccc']
# print(id(l))
# print(l)
# l[0] = 'AAA'
# print(l)
# print(id(l))

# 1.5 dict是可变类型
# dic = {'k1':111, 'k2':222}
# print(id(dic))
# print(dic)
# dic['k1'] = 333
# print(dic)
# print(id(dic))

# 1.6 bool是不可变类型

# 1.7 关于字典补充：定义：{}内用逗号隔开多个key：value，其中value可以是任意类型但是key必须是不可变类型
# dic={'k1':111,
#      'k2':3.1,
#      'k3':[333,],
#      'k4':{'name':'sugar'}
# }

# 2.什么是条件？什么可以当作条件 ？为何要用条件？
#   显式布尔值：True、False
#   隐式布尔值：所有数据类型，其中0、None、空为假
#
# 3.逻辑运算符:用来
# 3.1 not、and、or的基本使用
#not:就是把紧跟其后的那个条件结果取反，注意not与紧跟其后的那个条件是一个不可分割的整体
# print(not 16>13)
# print(not 10)
# and:逻辑与，用来连接左右两个条件，两个条件同时为True，最终结果才为True
# print(True and 10>3)#条件全为真，最终结果才为True（偷懒原则）
# or:逻辑或，用来连接左右两个条件，两个条件但凡有一个为True，最终结果为True，两个条件都为False的情况下，最终结果才为Fals
# print(3>2 or 0)
#3.2 区分优先级：not>and>or
# 了解：
# 短路运算（判断真假时偷懒使用）
# 4.成员运算符
# print("sugar" in "hello sugar")#判断一个字符串是否存在于一个大字符串中
# print(111 in [111,222,333])#判断元素是否存在列表中
# print('k1'in {'k1':111,'k2':222})#判断k是否存在于字典

# not in #不在里面
# 5.身份运算符
# is 判断id是否相等
#
# 6.流程控制之if判断
'''
语法1：
if 条件：
    代码1
    代码2
    代码3 #代码1，代码2，代码3由于缩进对的空格数一样，代表他们属于同一级别的代码，称为一组代码块会按照自上而下的特点依次运行
语法2：
if 条件：
    代码1
    代码2
    代码3
else:
    代码1
    代码2
    代码3
语法3：
if 条件：
    代码1
    代码2
    代码3
elif 条件2:
    代码1
    代码2
    代码3
'''

# score = input('请输入您的成绩：')#score=“18”
# score = int(score)
# if score >= 90:
#     print("优秀")
# elif score >= 80:
#     print("良好")
# elif score >=70 :
#     print("普通")
# else:
#     print("很差，小垃圾")

# 7、短路运算:偷懒原则，偷懒到哪个位置，就把当前位置的值返回
# 8、深浅copy
# list1 = [
#     "egon",
#     "lxx",
#     [1,2]
# ]
# list2 = list1 #list1和list2相关，即list1变list2也变，list1和list2相关
# 8.1 需求：拷贝一下原列表产生一个新的列表，想让两个列表完全独立开来，独立开是针对改操作
# 8.2浅copy：将原列表第一层的内存地址不加区分完全copy一份给新列表
# list1 = [
#     "egon",
#     "lxx",
#     [1,2]
# ]
# list2=list1.copy()
#由上两种情况可知，要想copy得到的新列表与原列表的改操作完全独立开必须有一种可以区分开可变类型和不可变类型的copy机制，这就是深copy
# 8.3深copy
# import copy
# list1 = [
#     "egon",
#     "lxx",
#     [1,2]
# ]

# list3 = copy.deepcopy(list1)