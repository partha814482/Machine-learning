import mysql.connector

conn = mysql.connector.connect(host = 'localhost', user = 'root', password = '8144', database = 'internshipdb')

mycursor = conn.cursor()

sql = 'insert into student (name,branch,id)  values(%s,%s,%s)'

# if user want to crearte multiple values then you can create list
# val = ('jhon','cse','56')

val = [('jhon','cse','56'),('mike','IT','78'),('tyson','me','80')]

# mycursor.execute(sql,val)

mycursor.executemany(sql,val)
conn.commit()
print(mycursor.rowcount,'record inserted')



