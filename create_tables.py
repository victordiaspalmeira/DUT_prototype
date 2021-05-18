from mysql.connector.cursor import MySQLCursorBufferedDict
import sql_handler

from contextlib import closing


def create_tables():
    tables_sql = [
        '''CREATE TABLE `duts` (`dev_id` varchar(20) NOT NULL,`model_id` int NOT NULL, PRIMARY KEY (`dev_id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci''',
        '''CREATE TABLE `dutmodels` (`ID` int NOT NULL AUTO_INCREMENT, `dev_id` varchar(20) DEFAULT NULL, `evaluate` float DEFAULT NULL, `train_timestamp` datetime DEFAULT NULL, `start_timestamp` datetime DEFAULT NULL, `end_timestamp` datetime DEFAULT NULL, PRIMARY KEY (`ID`)) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci''',        
    ]

    with closing(sql_handler.create_connection()) as db:
        with closing(db.cursor(buffered=True, dictionary=True)) as cursor:
            assert isinstance(cursor, MySQLCursorBufferedDict)
            map(cursor.execute, tables_sql)


if __name__ == '__main__':
    create_tables()