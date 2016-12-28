from DbManager import DbManager

class DbCommiter:
    def __init__(self):
        self.dbm = DbManager()

    def insert(self, sql, params=()):
        result = self.dbm.exe_query_commit(sql, params)
        return result

    def update (self, sql, params=()):
        result = self.dbm.exe_query_commit(sql, params)
        return result

    def find_all(self, sql, params=()):
        result = self.dbm.exe_query_fetchall(sql, params)
        return result

    def find_one(self, sql, params=()):
        result = self.dbm.exe_query_fetchone(sql, params)
        return result
