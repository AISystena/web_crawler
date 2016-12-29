from db_committer.DbCommitter import DbCommitter


class Categories(DbCommitter):
    def __init__(self):
        super().__init__()
        self.tbl_name = 'categories'

    def find_all(self):
        sql = 'SELECT * FROM ' + self.tbl_name
        result = self.dbc.exe_query_fetchall(sql)
        for data in result:
            print(data)

    def insert(self, params=()):
        # @TODO creating
        pass
        # sql = 'INSERT INTO~~'
        # result = self.dbc.exe_query_commit(sql, params)
        # return result

    def update(self, params=()):
        # @TODO creating
        pass
        # sql = 'UPDATE ~~'
        # result = self.dbc.exe_query_commit(sql, params)
        # return result

if __name__ == '__main__':
    cat = Categories()
    cat.find_all()
