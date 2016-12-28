from DbCommiter import DbCommiter

class Categories(DbCommiter):
    def __init__(self):
        super().__init__()
        self.tbl_name = 'categories'


    def get_categories(self):
        sql = 'SELECT * FROM '+ self.tbl_name
        result = super().find_all(sql)
        for data in result:
            print(data)


if __name__ == '__main__':
    cat = Categories()
    cat.get_categories()
