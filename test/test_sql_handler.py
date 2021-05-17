import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mysql.connector.connection import MySQLConnection
from mysql.connector.cursor import MySQLCursor, MySQLCursorBufferedDict
from sql_handler import *
from contextlib import closing


def test_create_connection():
    with closing(create_connection()) as db:
        assert isinstance(db, MySQLConnection)
        with closing(db.cursor()) as cursor:
            assert isinstance(cursor, MySQLCursor)
        
        with closing(db.cursor(buffered=True, dictionary=True)) as cursor:
            assert isinstance(cursor, MySQLCursorBufferedDict)
            sql = 'select * from dutModels'
            cursor.execute(sql)
            assert isinstance(cursor.fetchall(), list)