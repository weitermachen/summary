# 1、循环之while循环
# 1.1循环的语法与基本使用
'''
while 条件：
    代码1
    代码2
    代码3
'''

# while count <5:
#     print(count)
#     count+=1
# 1.2死循环与效率问题
#不会结束的循环称为死循环，即条件永远为真
# 死循环本身无害，纯计算无io的死循环会导致致命的效率问题
# 1.3循环的应用
# username = 'sugar'
# password = '123'
#
# inp_name = input("请输入你的账号:")
# inp_pwd = input("请输入你的密码:")
# if inp_name == username and inp_pwd == password:
#     print('登录成功')
# else:
#     print('账号名或密码错误')
# 1.4 退出循环的两种方式
#     条件
'''
username = 'sugar'
password = '123'
tag = True
while tag:
    inp_name = input("请输入你的账号:")
    inp_pwd = input("请输入你的密码:")
    if inp_name == username and inp_pwd == password:
        print('登录成功')
        tag = False
    else:
        print('账号名或密码错误')

    print("====end====")
'''
#     while+break
# break ,只要运行到break就会立刻终止本层循环
'''
username = 'sugar'
password = '123'
while True:
    inp_name = input("请输入你的账号:")
    inp_pwd = input("请输入你的密码:")
    if inp_name == username and inp_pwd == password:
        print('登录成功')
        break #立刻终止本层循环
    else:
        print('账号名或密码错误')

    print("====end====")
'''
# 1.5while循环嵌套

# 1.6 while+continue:结束本次循环，直接进入下一次(结束本层循环中continue之后的程序直接进入下一次循环)
# 在continue之后添加同级代码毫无意义，因为永远无法运行
'''
count=0
while count < 6:
    if count == 4:
        count+=1
        continue

    print(count)
    count+=1
'''

# 1.7 while+else
'''
while count < 6:
    if count == 4:
        count+=1
        continue
    print(count)
    count+=1
else:
    print('else包含的代码会在while循环结束后，并且while循环是在没有被break打断的情况下运行')
'''
# 1.8 应用案例
'''
# 版本1,简易的密码登录系统
username = 'sugar'
password = '123'
count = 0
tag = True
while tag:
    if count == 3:
        print('输错超过3次,退出')
        break
    inp_name=input('请输入您的账号:')
    inp_pwd=input('请输入您的密码:')

    if inp_name == username and inp_pwd ==password:
        print('登录成功')
        while tag:
            cmd=input("输入命令>:")
            if cmd == 'q':
                tag=False
            else:
                print("命令{x}正在运行".format(x=cmd))
    else:
        print('账号名或密码错误')
        count+=1
'''

'''
# 版本2:优化
username = 'sugar'
password = '123'
count = 0
tag = True
while count < 3:
    inp_name=input('请输入您的账号:')
    inp_pwd=input('请输入您的密码:')

    if inp_name == username and inp_pwd ==password:
        print('登录成功')
        while tag:
            cmd=input("输入命令>:")
            if cmd == 'q':
                tag=False
            else:
                print("命令{x}正在运行".format(x=cmd))
        break
    else:
        print('账号名或密码错误')
        count+=1
else:
    print('输错超过3次,退出')
'''

# 2、循环之for循环
# 2.1 for循环的语法与基本使用
'''
语法:
for 变量名 in 可迭代对象:#可迭代对象可以是列表\字典\字符串\元组\集合
    代码1
    代码2
    代码3
    ...
'''
# 案例一:循环取值
# 简单版
# for x in ['sugar','xia','ze']:
#     print(x)

# 复杂版
# l=['sugar','xia','ze']
# i=0
# while i<3:
#     print(l[i])
#     i+=1

# 案例二:字典循环取值
# 简单版
# dic={'k1':111,'k2':222,'k3':333}
# for x in dic:
#     print(x,dic[x])

# 复杂版:while循环遍历字典,太麻烦

# 案例二:字符串循环取值
# 简单版
# msg='you can you up,no can no bb'
# for x in msg:
#     print(x)

# 2.2 总结for循环与while循环的异同
# 相同:都是循环,for循环可以干的事,while循环也可以干
# 不同:while循环称之为条件循环,循环次数取决于条件何时变为假
    # for循环称之为"取值循环",循环次数取决于in后包含的值的个数

# 2.3for循环控制循环次数:range()
# range功能介绍
'''
>>> range(10)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> range(1,9)
[1, 2, 3, 4, 5, 6, 7, 8]
>>> range(1,9,1)
[1, 2, 3, 4, 5, 6, 7, 8]
>>> range(1,9,2)
[1, 3, 5, 7]
'''

# l=['aaa','bbb','ccc']#len(l)来查询列表l的长度
# for i in range(len(l)):
#     print(i,l[i])


# for i in range(30):
    # print('====>')

# 应用案例
'''
username = 'sugar'
password = '123'
count = 0
tag = True
for i in range(3):
    inp_name=input('请输入您的账号:')
    inp_pwd=input('请输入您的密码:')

    if inp_name == username and inp_pwd ==password:
        print('登录成功')
        while tag:
            cmd=input("输入命令>:")
            if cmd == 'q':
                tag=False
            else:
                print("命令{x}正在运行".format(x=cmd))
        break
    else:
        print('账号名或密码错误')
else:
    print('输错超过3次,退出')
'''

# 3.基本数据类型
# 3.1 int类型
# 3.1.1作用
# 3.1.2定义
# age = 10#等同于age=int(10)
# 3.1.3类型转换
# res=int('1253486')#纯数字字符串转成int
# print(res,type(res))
# print(bin(11))#bin()在Py中为10进制转二进制函数
# print(oct(11))#oct()在Py中为10进制转八进制函数
# print(hex(11))#hex()在Py中为10进制转十六进制函数
# print(int('0b1011',2))#表示二进制转十进制
# 3.1.4使用

# 3.2 float类型
# 3.2.1作用
# 3.2.2定义
# salary=3.1#salary=float(3.1)
# 3.2.3类型转换
# res=float('3.1')
# print(res,type(res))
# 3.2.4使用
# int与float没有需要掌握的内置方法
# 他们的使用就是数学运算+比较运算

# 3.3 字符串类型
# 3.3.1作用
# 3.3.2定义
# msg='hello'#msg=str('hello')
# print(type(msg))
# 3.3.3类型转换
#str可以把任意其他类型都转成字符串
# res=str({'a':1})
# print(res,type(res))
# 3.3.4使用:内置方法
# 1)优先掌握
#     一)按索引取值(正向取+反向取):只能取
'''
msg='hello world'
# 正向取
print(msg[0])
print(msg[5])
# 反向取
print(msg[-1])
'''

#     二)切片(和range类似)(顾头不顾尾,步长):索引的拓展应用,从一个大字符串中拷贝出一个子字符串
'''
# 顾头不顾尾
msg='hello world'
res=msg[0:4]
print(res)
# 步长
msg='hello world'
res=msg[0:5:2]
print(res)
# 反向步长
msg='hello world'
res=msg[5:0:-1]
print(res)
'''
#     三)长度len
#     四)成员运算in和not in
'''
# 判断一个子字符串是否存在于一个大字符串中
print('sugar'in'sugar is handsome')
print('sugar'not in'sugar is handsome')
'''
#     五)移除字符串左右两侧的符号strip
'''
# 去掉两侧的值,只去两边不去中间
msg='    sugar     '
res=msg.strip()#默认去掉空格,会产生一个新的值,不会改变原值,即字符串为不可变类型
print(res)
print(msg)

# 去掉多种符号
msg='****/-=()**sugar****-/=***'
res=msg.strip('*-/=()')
print(res)
print(msg)

# 应用
inp_user=input('your name >>:').strip()#.strip()用来去掉输入的内容中的空格
inp_pwd=input('your password >>:').strip()
if inp_user == 'sugar'and inp_pwd == '123':
    print('登录成功')
else:
    print('账号密码错误')
'''
#     六)切分split:把一个字符串按照某种分隔符进行切分,得到一个列表
'''
# 指定分隔符:默认分隔符是空格
info='sugar 21 male'
res=info.split()#split()的括号中不传表示默认按照空格来切分
print(res)

# 指定分隔次数
info='sugar:21:male'
res=info.split(':',1)
print(res)
'''
#     七)循环

#2)需要掌握
#     一)strip,lstrip,rstrip
'''
msg='****sugar****'
print(msg.strip('*'))
print(msg.lstrip('*'))#表示只去左边
print(msg.rstrip('*'))#表示只去右边
'''

#     二)lower,upper
'''
msg='AbbbCCCC'
print(msg.lower())#产生一个新的字符全都变为小写
print(msg.upper())#产生一个新的字符全都变为大写
'''

#     三)startswith,endswith
'''
print('sugar is handsome'.startswith('sugar'))#以括号里面的内容开头
print('sugar is handsome'.endswith('handsome'))#以括号里面的内容结尾
'''

#     四)format的三种玩法
#     五)split,rsplit(split表示从左往右开始切,rsplit表示从右往左开始切)
#     六)join:把列表拼接成字符串,和split的功能正好相反
'''
l=['sugar','21','male']
res=':'.join(l)#按照某个分割符号,把元素全为字符串的列表拼接成一个字符串
print(res)
'''

#     七)replace
'''
msg='you can you up,no can no bb'
print(msg.replace('you','YOU'))
print(msg.replace('you','YOU',1))
'''
#     八)isdigit#判断字符是否由纯数字组成
'''
print('123'.isdigit())
print('12.3'.isdigit())

# 应用
age=input('请输入你的年龄:').strip()
if age.isdigit():
    age=int(age)
    if age>18:
        print('猜大了')
    elif age<18:
        print('猜小了')
    else:
        print('猜对了')
else:
    print('必须输入数字')
'''

# 3)了解
#      一)find,rfind,index,rindex,count
'''
#找到返回起始索引
msg='hello sugar hahaha'
print(msg.find('e'))#返回要查找的字符串在大字符串中的起始索引
print(msg.find('sugar'))
print(msg.index('e'))
print(msg.index('sugar'))
#找不到
print(msg.find('xxx'))#返回-1
print(msg.index('xxx'))#抛出异常,程序停止
'''

#      二)center,ljust,rjust,zfill
'''
#控制打印格式
print('sugar'.center(50,'*'))
print('sugar'.ljust(50,'*'))
print('sugar'.rjust(50,'*'))
print('sugar'.zfill(10))

'''
#      三)expandtabs
'''
msg='hello\tworld'#\t默认4个宽度
print(msg.expandtabs(2))#设置制表符代表的空格数为2
'''
#      四)capitalize,swapcase,title
'''
print('hello world sugar'.capitalize())#表示首字母大写
print('Hello world SUGAR'.swapcase())#表示大写变小写,小写变大写
print('hello world sugar'.title())#表示每个单词首字母大写
'''

#      五)isdigit,isnumberic,isdecimal
'''
num1=b'4'
num2=u'4'
num3='四'
num4='Ⅳ'

'''

#isdigit只能识别num1,num2
#isnumberic可以识别num2,num3,num4
#isdecimal只能识别num2

# 3.4 list
# 3.4.1 作用:按位置存放多个值
# 3.4.2 定义
'''
l=[1,1.2,'a']
print(type(l))
'''

# 3.4.3 类型转换:但凡能够被for循环遍历的类型都可以当作参数传给list()转成列表
'''
res=list('hello')
print(res)

res=list({'k1':111,'k2':222,'k3':333})
print(res)
'''

# 3.4.4 内置方法
# 1)优先掌握的操作:
#     一)按索引存取值(正向存取+反向存取):即可取也可以改
'''
l=[111,'sugar','hello']
# 正向取
print(l[0])
# 反向取
print(l[-1])
# 可以取也可以改:索引存在则修改对应值,索引不存在 则报错
l[0]=222
print(l)
'''

#     二)切片(顾头不顾尾,步长),切片等同于浅拷贝行为
'''
l=[111,'sugar','hello','a','b','c']
print(l[0:3])
print(l[0:5:2])
'''

#     三)长度
#     四)成员运算in和not in
#     五)追加append在列表的最后面添加
'''
l=[111,'sugar','hello']
l.append(333)
print(l)
'''

#     六)插入值insert
'''
l=[111,'sugar','hello']
l.insert(1,'xia')#在1号索引前面插入xia
print(l)
'''

#     七)连接两个列表
'''
new_l=[1,2,3]
l=[111,'sugar','hello']
print(l)
# 代码实现
# for item in new_l:
#     l.append(item)
# print(l)

#extend实现上述代码
l.extend(new_l)
print(l)
'''

#     八)删除
'''
# 方式一:del为通用的删除方法,只是单纯的删除,没有返回值,不支持赋值语法
l=[111,'sugar','hello']
del l[1]
print(l)

# 方式二:l.pop()根据索引删除,不指定索引默认删除最后一个,会返回值,返回他删除的值
l=[111,'sugar','hello']
res =l.pop(1)
print(l)
print(res)

# 方式三:l.remove()根据指定元素删除,返回None
l=[111,'sugar','hello']
res=l.remove('sugar')
print(l)
print(res)
'''
# 2)需要掌握的操作
#     一)l.count(),统计括号内元素出现的次数
'''
l=[111,'sugar','hello','aaa','aaa','aaa']
print(l.count('aaa'))
'''

#     二)l.index(),返回找到的第一个索引值
'''
l=[111,'sugar','hello','aaa','aaa','aaa']
print(l.index('aaa'))
'''

#     三)l.clear(),清空列表元素
'''
l=[111,'sugar','hello','aaa','aaa','aaa']
l.clear()
print(l)
'''

#     四)l.reverse(),将列表反转,不是排序就是将列表到过来
'''
l=[111,'sugar','hello','aaa','aaa','aaa']
l.reverse()
print(l)
'''

#     五)l.sort(),排序列表,默认从小到大排序,列表内元素必须是同种类型才能排序
'''
l=[11,-3,9,2]
l.sort()#默认从小到大排序
print(l)
l.sort(reverse=True)#从大到小排序
print(l)

# 字符串可以比大小,按照对应的位置的字符依次比较
# 字符的大小是按照ASCII码表的先后顺序加以区别,表中排在后面的字符大于前面
#列表也可以比大小,原理同字符串,对应位置的元素必须是同种类型
'''
# 3.5元组,元组就是一个不可变的列表
# 3.5.1 作用:按照索引/位置存放多个值,只用于读不用于改
# 3.5.2 定义:()内用逗号分隔开多个任意类型的元素
'''
t=(1,1.3,'aa')#t=tuple((1,1.3,'aa'))
print(t,type(t))

x=(10)#单独一个括号代表包含的意思
print(x,type(x))

t=(10,)#如果元组中只有一个元素,必须加逗号
print(t,type(t))

#元组中存放的是每个索引的内存地址,元组不可改指的是存放的索引的内存地址不可变
t=(1,1.3,'aa')#t=(0->值1的内存地址,1->值1.3的内存地址,2->值'aa'的内存地址)
t[0]=111111#报错

t=(1,[11,22])#t=(0->值1的内存地址,1->列表[11,22]的内存地址)
print(id(t[0]),id(t[1]))
t[0]=11111111#不能改
t[1]=22222222#不能改

t[1][0]=11111#可以改

'''

# 3.5.3 类型转换
'''
print(tuple('hello'))
print(tuple([1,2,3]))
print(tuple({'a1':111,'a2':333}))
'''

# 3.5.4 内置方法
# 1)优先掌握的操作:
#     一)按索引存取值(正向存取+反向存取):即可存也可取
#     二)切片(顾头不顾尾,步长)
#     三)长度
#     四)成员运算in和not in
# 2)需要掌握的操作
#     一)index
#     二)count

# 3.5字典
# 3.5.1 作用
# 3.5.2 定义:在{}内用逗号分隔开多个key:value,其中value可以是任意类型,但是key必须是不可变类型且不能重复
'''
# 方式一
d={'k1':111,(1,2,3):222}
print(d['k1'])
print(d[1,2,3])

# 方式二
d={}#默认定义出空字典
print(d,type(d))

# 方式三
d=dict(x=1,y=2,z=3)
print(d,type(d))
'''

# 3.5.3 类型转换
'''
# 方式一
info=[
    ['name','sugar'],
    ['age',21],
    ['gender','male']
]

d={}
for item in info:
    d[item[0]]=item[1]
print(d)

# 方式二
info=[
    ['name','sugar'],
    ['age',21],
    ['gender','male']
]

d={}
for k,v in info:#k,v=['name','sugar']
    d[k]=v
print(d)

# 方式三
info=[
    ['name','sugar'],
    ['age',21],
    ['gender','male']
]
res=dict(info)
print(res)

#方式四,快速初始化一个字典
keys=['name','age','gender']
d={}.fromkeys(keys,None)
print(d)
'''

# 3.5.4 内置方法
# 1)优先掌握的操作:
#     一)按key存取值:即可存也可取
'''
d={'k1':111}
#针对赋值操作:key存在,则修改
d['k2']=222
#针对赋值操作:key不存在,则创建新值
d['k2']=333
print(d)
'''

#     二)长度len
#     三)成员运算in和not in
#     四)del,通用删除
#     五)pop删除,根据key删除,返回删除key对应的value值
'''
d={'k1':111,'k2':222}
res=d.pop('k2')
print(d)
print(res)
'''

#     六)popitem删除:随机删除,返回元组(删除的key,删除的value)
'''
d={'k1':111,'k2':222}
res=d.popitem()
print(d)
print(res)
'''


#     七)键值keys,values(),items()
'''
d={'k1':111,'k2':222}
print(d.keys())
print(d.values())
print(d.items())
'''

# 2)需要掌握的操作
#     一)clear(),清空字典
#     二)get(),按照key取值,相比于直接取值容错性更好
'''
dic={'k1':111,'k2':222}
print(dic['k3'])#k不存在报错
print(dic.get('k3'))#key不存在不报错返回None
'''

#     三)setdefault()
'''
info={}
if 'name'in info:
    ...#等同于pass
else:
    info['name']='sugar'
print(info)

#如果key有则不添加
info={'name':'sugar'}
info.setdefault('name','sugar')
print(info)

#如果key没有则添加,返回字典中执行完毕后key对应的value
info={}
res=info.setdefault('name','sugar')
print(info)
print(res)
'''

#     四)update(),用()里面的字典更新update之前的字典,返回None
'''
dic={'k1':111,'k2':222}
d={'k1':111,'k2':222,'k3':333}
res=dic.update(d)
print(dic)
print(res)
'''

# 3.6集合
# 3.6.1 作用
# 3.6.2 定义
# 3.6.3 类型转换
# 3.6.4 内置方法
# 1)优先掌握的操作:
#     一)按key存取值:即可存也可取
#     二)长度len
#     三)成员运算in和not in
#     四)del,通用删除
#     五)pop删除,根据key删除,返回删除key对应的value值
# 2)需要掌握的操作
#     一)clear(),清空字典
#     二)get(),按照key取值,相比于直接取值容错性更好



# 3.6集合
# 3.6.1 作用:关系运算和去重
# 关系运算:找到两者共同的好友
'''
friends1={'sugar','xia','xiao','egon'}
friends2={'sugar','ze','heng','egon'}

l=[]
for x in friends1:
    if x in friends2:
        l.append(x)
print(l)
'''


# 3.6.2 定义:在{}内用逗号分隔开多个元素,多个元素满足一下三个条件
            # 1.集合内元素必须为不可变类型
            # 2.集合内元素无序
            # 3.集合内元素没有重复
# s={1,2}#s=set({1,2})
# s={1,[1,2]}#集合内必须为不可变类型
# s={1,'a','z','b',4,7}#集合内元素无序
# s={1,1,1,1,1,'a','b'}#集合内元素没有重复
# print(s)

# s={}#默认是空字典
# print(type(s))
# s=set()#定义空集合
# print(s,type(s))

# 3.6.3 类型转换
# set({1,2,3})
# res=set('hellolllll')
# print(res)

# print(set([1,1,1,1,1]))

# 3.6.4 内置方法
friends1={'sugar','xia','xiao','egon'}
friends2={'sugar','ze','heng','egon'}
# =====================关系运算==============================
# 1)取交集:两者共同的好友:

'''
res=friends1 & friends2
print(res)

print(friends1.intersection(friends2))
'''
# 2)取并集:两者所有同的好友:

'''
print(friends1 | friends2)

print(friends1.union(friends2))
'''
# 3)取差集:取friends1独有好友:
'''
print(friends1-friends2)

print(friends1.difference(friends2))
'''
# 4)取对称差集:取两个用户独有的好友(即去掉共同的好友)
'''
print((friends1-friends2)|(friends2-friends1))
print(friends1^friends2)

print(friends1.symmetric_difference(friends2))
'''
# 5)父子集:包含的关系即一个集合为另一个集合的子集
'''
print(friends1.issuperset(friends2))#friends1是friends2的爹
print(friends1.issubset(friends2))#friends1是friends2的儿子
'''

# ========================去重========================
# 局限性:
# 1.只能针对不可变类型
# print(set([1,1,1,1,1,2,2]))

# 2.无法保证原来的顺序
# l=[1,'a','b','z',1,1,1,2]
# l=list(set(l))
# print(l)

#     一)discard(),remove()
'''
s={1,2,3}
s.discard(4)#删除集合中元素,不存在不报错
print(s)
s.remove(4)#删除集合中元素,不存在则报错
print(s)
'''

#     二)update()
'''
s={1,2,3}
s.update({1,3,5})
print(s)

s={1,2,3}
s.difference_update({1,3,5})#等价于s=s.difference({1.3,5})
print(s)
'''

#     三)isdisjoint()#判断两个集合是不是完全独立,没有共同部分,若是返回True
