import MySQLdb
import processplate
processplate.processplatefunc()
#print(processplate.final_valid_plates)
plates = processplate.final_valid_plates
#plates = ['苏ADK6318', '苏A00CP1', '苏A8R87G']
#print(plates)

def readDB():
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='root',
        password='',
        db='user',
        charset='utf8'
    )
    cur = conn.cursor()
    cur.execute('select * from carplate')
    print(cur.fetchall())
    cur.close()
    conn.close()


def writeDB():
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='root',
        password='',
        db='user',
        charset='utf8'
    )
    cur = conn.cursor()
    for plate in plates:
        try:
            # 执行插入操作
            cur.execute("INSERT INTO carplate (plate) VALUES (%s)", (plate,))
        except Exception as e:
            print("Error inserting plate:", e)
    conn.commit()
    cur.execute('SELECT * FROM carplate')
    print(cur.fetchall())
    cur.close()
    conn.close()

writeDB()

#readDB()

