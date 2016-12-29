from abc import ABCMeta, abstractmethod
import MySQLdb
 
class DbConnector:
    def __init__(self):
        """ DBのコネクションを生成 """

        # 接続する 
        self.con = MySQLdb.connect(
                user='root',
                passwd='systena!',
                host='localhost',
                db='web_crawler')

    def __check_params(self, sql, params):
        """ パラメータチェック """
        if not isinstance(sql, str):
            return False
        if not isinstance(params, tuple):
            return False
        return True


    def __exe_query(self, sql, params):
        """ queryを実行 """
        is_executable = self.__check_params(sql, params)
        if not is_executable:
            return None

        # カーソルを取得する
        cur= self.con.cursor()

        # クエリを実行する
        if len(params) > 0:
            cur.execute(sql, params)
        else:
            cur.execute(sql)

        return cur


    def exe_query_fetchone(self, sql, params=()):
        """ tableからデータを1行取得する """
        cur = self.__exe_query(sql, params)
        if cur is None:
            return None

        # 実行結果をすべて取得する
        row = cur.fetchone()
        print(row)

        cur.close
        return row

    def exe_query_fetchall(self, sql, params=()):
        """ tableからデータを全て取得する """
        cur = self.__exe_query(sql, params)
        if cur is None:
            return None

        # 実行結果をすべて取得する
        rows = cur.fetchall()
        for row in rows:
            print(row)

        cur.close
        return rows

    def exe_query_commit(self, sql, params=()):
        """ tableへデータをコミットする """
        cur = self.__exe_query(sql, params)
        if cur is None:
            return False

        cur.close
        return True

    def __del__(self):
        """ DBのコネクションを破棄 """
        self.con.close


class DbCommitter(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self.dbc = DbConnector()

    @abstractmethod
    def insert(self, params=()):
        pass

    @abstractmethod
    def update(self, params=()):
        pass

    @abstractmethod
    def find_all(self):
        pass
