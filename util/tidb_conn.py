import os
from configparser import ConfigParser
from typing import List
import pandas as pd
import time
import pymysql
import logging


class TiDBHypo:
    def __init__(self, db_name):
        config_raw = ConfigParser()
        config_raw.read('configure.ini')
        defaults = config_raw.defaults()
        if db_name is None:
            db_name = 'test'
        self.host = defaults.get('tidb_ip')
        self.port = defaults.get('tidb_port')
        self.user = defaults.get('tidb_user')
        self.password = defaults.get('tidb_password')
        self.database = defaults.get('tidb_database')
        self.conn = None
        self.create_connection()
        

    def close(self):
        self.conn.close()


    def create_connection(self):
        if self.conn:
            self.close()
        self.conn = pymysql.connect(host='127.0.0.1',
                     port=4000,
                     user='root',
                     password='',
                     database="{}".format(self.db_name),
                     local_infile=True)


    def execute_create_hypo(self, index):
        schema = index.split("#")
        table_name = schema[0]
        idx_cols = schema[1]
        idx_name = f"hypo_{table_name}_{idx_cols}_idx" 
        statement = (
            f"create index {idx_name} type hypo "
            f"on {table_name} ({idx_cols})"
        )
        cur = self.conn.cursor()
        cur.execute(statement)
        return idx_name


    def get_hypo_indexes_from_table(self, table_name):
        statement = f"show create table {table_name}"
        result = self.exec_fetch(statement)
        hypo_indexes = []
        for line in result[1].split("\n"):
            if "HYPO INDEX" in line:
                tmp = line.split("`")
                hypo_indexes.append(tmp[1])
        return hypo_indexes
    
    def execute_delete_hypo(self, table, idx_name):
        statement = f"drop index {idx_name} on {table}"
        cur = self.conn.cursor()
        cur.execute(statement)
        return cur.fetchall()
    
    
    def delete_all_hypo_indexes_table(self, table_name):
        indexes = self.get_hypo_indexes_from_table(table_name)
        for idx in indexes:
            self.execute_delete_hypo(table_name, idx)
            
    def delete_all_hypo_indexes_database(self): 
        tables = self.get_tables(self.database)
        for table in tables:
            self.delete_all_hypo_indexes_table(table)
            

    def _cleanup_query(self, query):
        for query_statement in query.text.split(";"):
            if "drop view" in query_statement:
                self.exec_only(query_statement)
                
    def _prepare_query(self, query):
        for query_statement in query.text.split(";"):
            if "create view" in query_statement:
                try:
                    self.exec_only(query_statement)
                except Exception as e:
                    logging.error(e)
            elif "select" in query_statement or "SELECT" in query_statement:
                return query_statement
    
    def _get_plan(self, query):
        query_text = self._prepare_query(query)
        statement = f"explain format='verbose' {query_text}"
        query_plan = self.exec_fetch(statement, False)
        for line in query_plan:
            if "stats:pseudo" in line[5]:
                raise Exception("plan with pseudo stats " + str(query_plan))
        self._cleanup_query(query)
        return query_plan

    def get_queries_cost(self, query_list):
        cost_list: List[float] = list()
        for i, query in enumerate(query_list):
            query_plan = self._get_plan(query)
            cost = query_plan[0][2]
            cost_list.append(float(cost))
        return cost_list

    def execute_sql(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()

    def get_tables(self, schema):
        tables_sql = 'select tablename from pg_tables where schemaname=\''+schema+'\';'
        cur = self.conn.cursor()
        cur.execute(tables_sql)
        rows = cur.fetchall()
        table_names = list()
        for i, table_name in enumerate(rows):
            table_names.append(table_name[0])
        return table_names
