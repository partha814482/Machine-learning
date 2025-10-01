import mysql.connector

conn = mysql.connector.connect(host = 'localhost',user = 'root', password = '8144')

if conn.is_connected():
    print('connection established')
    
mycursor = conn.cursor()
mycursor.execute('create database internshipdb')
print(mycursor)

