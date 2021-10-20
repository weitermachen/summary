# 一、垃圾回收机制
# 1、引用计数
# x = 10
# l = ['a', 'b', x]  # l=['a'的内存地址，'b'的内存地址，10的内存地址]
# x = 123
# print(l[2])
# 2、标记清除
# 循环引用=>导致内存泄漏问题=>解决方案：标记清除
# l1 = [111, ]
# l2 = [222, ]
#
# l1.append(l2)  # l1=[值111的内存地址，l2列表的内存地址]
# l2.append(l1)  # l2=[值222的内存地址，l1列表的内存地址]

# print(id(l1[1]))
# print(id(l2))
#
# print(id(l2[1]))
# print(id(l1))
# del l1  # 此时l1的直接引用被剪掉，但还存在间接引用，不会被当作垃圾清除，但我们同时也无法访问到
# del l2  # 此时l2的直接引用被剪掉，但还存在间接引用，不会被当作垃圾清除，但我们同时也无法访问到
# print(l1)
# 二、与用户交互
# 1、接收用户的输入
# python3:input会将用户输入的所有内容都存成字符串类型
# username = input("请输入您的账号：")
# print(username, type(username))

# age = input("请输入你的年龄：")
# print(age, type(age))
# age = int(age)  # int只能将纯数字的字符串转成整型
# print(age > 16)

# 在python2中：
# raw_input():用法与python3的input一模一样
# input():要求用户必须输入一个明确的数据类型，输入的是什么类型，就存成什么类型

# 2、格式化输出
# 1）%
# 值按照位置与%s一一对应，少一个不行，多一个也不行
# res="my name is %s my age is %s" %('sugar','21')
# print(res)

# 以字典形式传值，打破位置限制
# res='我的名字是 %(name)s 我的年龄是 %(age)s' %{'age':'21','name':'sugar'}
# print(res)

# 2）str.format
# 按照位置传值
# res='我的名字是{}我的年龄是{}' .format('sugar',21)
# print(res)

# res='我的名字是{0}{0}{0}我的年龄是{1}{1}' .format('sugar',21)
# print(res)

# 打破位置的限制，按照key=value传值
# res='我的名字是{name}我的年龄是{age}'.format(age=21,name='sugar')
# print(res)

# 3) f
# x = input('your name:')
# y = input('your age:')
# res = f'我的名字是{x}我的年龄是{y}'
# print(res)

# 三、基本运算符
# 1、算数运算符
# print(3.1 + 10)
# print(10 / 3)  # 结果带小数
# print(10 // 3)  # 只保留整数部分
# print(10 % 3)  # 取模，取余数
# print(10 ** 3)  # 10的3次方
# 2、比较运算
# print(10 > 3)
# print(10 == 10)
# print(10 >= 10)
# print(10 >= 3)


