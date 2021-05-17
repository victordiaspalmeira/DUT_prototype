import mysql
import dotenv
from mysql.connector.connection import MySQLConnection
dotenv.load_dotenv('./.env')
import os


from mysql import connector

def create_connection():
    db = connector.connect(
    host= os.getenv("DB_HOST"),
    user= os.getenv("DB_USER"),
    password= os.getenv("DB_PASSWORD"),
    database= os.getenv("DB"),
    )

    assert isinstance(db, MySQLConnection)
    return db