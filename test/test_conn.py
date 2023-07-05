import sys
sys.path.append("/home/ubuntu/CODE/CHPDA/")

import util.tidb_connector as connector


def main():
    conn = connector.TiDBDatabaseConnector("tpch")
    o_id = conn.execute_create_hypo("customer#c_custkey,c_mktsegment")
    print(o_id)
    conn.close()

if __name__ == '__main__':
    main()