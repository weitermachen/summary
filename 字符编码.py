'''
结论:
1.内存固定使用unicode,我们可以改变的是存入硬盘采用格式
    英文+汉字->unicode->gbk
    英文+日文->unicode->shift-jis
    万国字符->unicode->uft-8
2.文本文件存取乱码问题
    存乱了:解决方法是编码格式应该设置成支持文件内字符串的格式
    取乱了:解决方法是文件以什么编码格式存入硬盘就以什么编码格式读入内存
3.python解释器默认读文件的编码
    python3默认:uft-8
    python2默认:ASCII

    指定文件头修改默认的编码:
    在py文件的首行写:
        #coding:gbk
4.保证运行python程序前两阶段不乱码的核心法则:指定文件头
    #coding:文件当初存入硬盘时所采用的编码格式
5.python3的str类型默认直接存成Unicode格式,无论如何都不会乱码
保证python2的str类型不乱码:x=u'上'
6.了解
python2解释器有两种字符串类型:str,unicode
    #str类型
    x='上'#字符串值会按照文件头指定的编码格式存入变量值的内存空间
    #Unicode类型
    x=u'上'#强制存成unicode

# 编码和解码
x='上'
res=x.encode('gbk')#编码,unicode--->gbk
print(res,type(res))

print(res.decode('gbk'))
'''


