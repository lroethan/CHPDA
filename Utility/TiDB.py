import os
from configparser import ConfigParser
from typing import List
import pandas as pd
import time
import pymysql


class TiDBHypo:
    def __init__(self):
        config_raw = ConfigParser()
        config_raw.read(os.path.abspath('..') + '/configure.ini')
        defaults = config_raw.defaults()
        self.host = defaults.get('tidb_ip')
        self.port = defaults.get('tidb_port')
        self.user = defaults.get('tidb_user')
        self.password = defaults.get('tidb_password')
        self.database = defaults.get('tidb_database')
        self.conn = pymysql.connect(host=self.host, port=int(self.port), user=self.user, password=self.password,
                                    database=self.database, charset='utf8')

    def close(self):
        self.conn.close()

    def execute_create_hypo(self, index):
        schema = index.split("#")
        sql = "CREATE INDEX START_X_IDx ON " + schema[0] + "(" + schema[1] + ");"
        cur = self.conn.cursor()
        cur.execute(sql)
        return cur.lastrowid

    def execute_delete_hypo(self, oid):
        sql = "DROP INDEX START_X_IDx;"
        cur = self.conn.cursor()
        cur.execute(sql)
        return True

    def get_queries_cost(self, query_list):
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        for i, query in enumerate(query_list):
            query = "explain " + query
            cur.execute(query)
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            cost_info = str(df[0][0])
            cost_list.append(float(cost_info[cost_info.index("..") + 2:cost_info.index(" rows=")]))
        return cost_list

    def get_storage_cost(self, oid_list):
        costs = list()
        cur = self.conn.cursor()
        for i, oid in enumerate(oid_list):
            cost_info = 0
            if oid == 0:
                continue
            sql = "select sum(data_length+index_length) from information_schema.tables where table_schema='" + self.database + "' and table_name='START_TABLE';"
            cur.execute(sql)
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            cost_info = int(df[0][0])
            costs.append(cost_info)
        return costs

    def execute_sql(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()

    def delete_indexes(self):
        sql = 'DROP INDEX START_X_IDx;'
        self.execute_sql(sql)

    def get_sel(self, table_name, condition):
        cur = self.conn.cursor()
        totalQuery = "select * from " + table_name + ";"
        cur.execute("EXPLAIN " + totalQuery)
        rows = cur.fetchall()[0][0]
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from " + table_name + " Where " + condition + ";"
        cur.execute("EXPLAIN  " + resQuery)
        rows = cur.fetchall()[0][0]
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        return select_rows/total_rows

    def get_rel_cost(self, query_list):
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        for i, query in enumerate(query_list):
            _start = time.time()
            query = "explain analyse" + query
            cur.execute(query)
            _end = time.time()
            cost_list.append(_end-_start)
        return cost_list

    def create_indexes(self, indexes):
        for index in indexes:
            schema = index.split("#")
            sql = 'CREATE INDEX START_X_IDx ON ' + schema[0] + "(" + schema[1] + ');'
            self.execute_sql(sql)

    def delete_t_indexes(self):
        sql = "DROP INDEX START_X_IDx;"
        self.execute_sql(sql)

    def get_tables(self, schema):
        tables_sql = 'select table_name from information_schema.tables where table_schema=\''+schema+'\';'
        cur = self.conn.cursor()
        cur.execute(tables_sql)
        rows = cur.fetchall()
        table_names = list()
        for i, table_name in enumerate(rows):
            table_names.append(table_name[0])
        return table_names

    def get_attributes(self, table_name, schema):
        attrs_sql = 'select column_name, column_type from information_schema.columns where table_schema=\''+schema+'\' and table_name=\''+table_name+'\''
        cur = self.conn.cursor()
        cur.execute(attrs_sql)
        rows = cur.fetchall()
        attrs = list()
        for i, attr in enumerate(rows):
            info = str(attr[0]) + "#" + str(attr[1]) + ")"
            attrs.append(info)
        return attrs