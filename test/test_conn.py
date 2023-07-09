import sys

sys.path.append("/home/ubuntu/CODE/CHPDA/")
import unittest

import util.tidb_connector as connector


def hist_tuple_to_dict(hist_tuple):
    result = {}
    for tpl in hist_tuple:
        key = tpl[1] + "#" + tpl[3]
        value = [tpl[6], tpl[7], tpl[8]]
        result[key] = value
    return result


def main():
    conn = connector.TiDBDatabaseConnector("tpch")

    print(conn.get_indexe_size("customer#tiflash"))

    # o_stats = conn.exec_fetch("show stats_meta;", one=False)

    # print(o_stats)

    # sql = "show stats_histograms;"
    # res = conn.exec_fetch(sql, one=False)
    # res2 = hist_tuple_to_dict(res)
    # print(res2)

    # json = conn.get_histogram()
    # print(json)
    # o_id = conn.execute_create_hypo("customer#c_custkey,c_mktsegment")
    # print(o_id)
    conn.close()


if __name__ == "__main__":
    main()
