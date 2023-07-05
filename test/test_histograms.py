import json

def is_separate_line(line):
    line = line.strip()
    if len(line) == 0:
        return False
    for c in line:
        if c != "+" and c != "-":
            return False
    return True


def trim_and_split_explain_result(explain_result):
    lines = explain_result.split("\n")
    idx = [0, 0, 0]
    p = 0
    for i in range(len(lines)):
        if is_separate_line(lines[i]):
            idx[p] = i
            p += 1
            if p == 3:
                break
    if p != 3:
        raise Exception("invalid explain result")

    return lines[idx[0] : idx[2] + 1]


def split_rows(rows):
    results = []
    for row in rows:
        cols = row.split("|")
        cols = [c.strip() for c in cols[1:-1]]
        results.append(cols)
    return results


def parse_text(explain_text):
    explain_lines = trim_and_split_explain_result(explain_text)
    print(explain_lines)
    rows = split_rows(explain_lines[3 : len(explain_lines) - 1])
    result = {}
    for row in rows:
        db_name = row[0]
        table_name = row[1]
        column_name = row[3]
        num_dv = row[6]
        num_nv = row[7]
        column_size = row[8]
        result[table_name + "#" + column_name] = {
            "num_dv": num_dv,
            "num_nv": num_nv,
            "column_size": column_size,
        }

    return json.dumps(result)



# def extract_element(table_string, row_index, column_name):
#     rows = table_string.strip().split('\n')
#     column_names = rows[1].strip().split('|')
#     print(column_names)
#     column_index = column_names.index(column_name.strip())
#     data_row = rows[row_index + 2].strip().split('|')
#     element = data_row[column_index].strip()
    
#     return element

histograms = '''
+---------+------------+----------------+-----------------+----------+---------------------+----------------+------------+--------------+-------------------------+-------------+-----------------+----------------+----------------+---------------+
| Db_name | Table_name | Partition_name | Column_name     | Is_index | Update_time         | Distinct_count | Null_count | Avg_col_size | Correlation             | Load_status | Total_mem_usage | Hist_mem_usage | Topn_mem_usage | Cms_mem_usage |
+---------+------------+----------------+-----------------+----------+---------------------+----------------+------------+--------------+-------------------------+-------------+-----------------+----------------+----------------+---------------+
| tpch    | customer   |                | C_CUSTKEY       |        0 | 2023-06-29 15:31:34 |         149568 |          0 |            8 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | customer   |                | C_NAME          |        0 | 2023-06-29 15:31:34 |         148224 |          0 |           19 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | customer   |                | C_ADDRESS       |        0 | 2023-06-29 15:31:34 |         148176 |          0 |        26.05 |   0.0015099963487022997 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | customer   |                | C_NATIONKEY     |        0 | 2023-06-29 15:31:34 |             25 |          0 |            8 |     0.04263135530917918 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | customer   |                | C_PHONE         |        0 | 2023-06-29 15:31:34 |         144672 |          0 |           16 |   0.0026251416983150227 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | customer   |                | C_ACCTBAL       |        0 | 2023-06-29 15:31:34 |         139872 |          0 |            9 |    0.007551065809589437 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | customer   |                | C_MKTSEGMENT    |        0 | 2023-06-29 15:31:33 |              5 |          0 |           10 |     0.20113716736359355 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | customer   |                | C_COMMENT       |        0 | 2023-06-29 15:31:34 |         149536 |          0 |        74.11 |  -0.0013968027027810798 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_ORDERKEY      |        0 | 2023-06-29 15:31:34 |        1487616 |          0 |         6.95 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_PARTKEY       |        0 | 2023-06-29 15:31:34 |         196960 |          0 |         6.67 |  -0.0005779877836632638 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_SUPPKEY       |        0 | 2023-06-29 15:31:34 |          10000 |          0 |         6.47 |   -0.001327361863265675 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_LINENUMBER    |        0 | 2023-06-29 15:31:34 |              7 |          0 |         6.16 |     0.17925914003230342 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_QUANTITY      |        0 | 2023-06-29 15:31:34 |             50 |          0 |            9 |    0.025891376469877266 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_EXTENDEDPRICE |        0 | 2023-06-29 15:31:34 |         941184 |          0 |            9 |    0.005679244994703122 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_DISCOUNT      |        0 | 2023-06-29 15:31:34 |             11 |          0 |            9 |     0.09535172850377327 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_TAX           |        0 | 2023-06-29 15:31:34 |              9 |          0 |            9 |     0.10987582141931113 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_RETURNFLAG    |        0 | 2023-06-29 15:31:34 |              3 |          0 |            2 |      0.3761196918555518 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_LINESTATUS    |        0 | 2023-06-29 15:31:34 |              2 |          0 |            2 |      0.4975922802407373 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_SHIPDATE      |        0 | 2023-06-29 15:31:34 |           2526 |          0 |            8 |    -0.00064493239807099 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_COMMITDATE    |        0 | 2023-06-29 15:31:34 |           2466 |          0 |            8 |  -0.0008091469186916645 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_RECEIPTDATE   |        0 | 2023-06-29 15:31:34 |           2554 |          0 |            8 |  -0.0006497527189496226 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_SHIPINSTRUCT  |        0 | 2023-06-29 15:31:34 |              4 |          0 |           13 |      0.2476051179397787 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_SHIPMODE      |        0 | 2023-06-29 15:31:34 |              7 |          0 |         5.29 |      0.1433823818924952 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | L_COMMENT       |        0 | 2023-06-29 15:31:34 |        4508672 |          0 |        27.49 |    0.002223634998529881 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | lineitem   |                | PRIMARY         |        1 | 2023-06-29 15:31:34 |        5960704 |          0 |            0 |                       0 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | nation     |                | N_NATIONKEY     |        0 | 2023-06-29 15:31:34 |             25 |          0 |            8 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | nation     |                | N_NAME          |        0 | 2023-06-29 15:31:34 |             25 |          0 |         8.08 |      0.9130769230769231 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | nation     |                | N_REGIONKEY     |        0 | 2023-06-29 15:31:34 |              5 |          0 |            8 |      0.3476923076923077 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | nation     |                | N_COMMENT       |        0 | 2023-06-29 15:31:34 |             25 |          0 |        75.88 |     0.04692307692307692 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | orders     |                | O_ORDERKEY      |        0 | 2023-06-29 15:31:35 |        1487616 |          0 |            8 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | orders     |                | O_CUSTKEY       |        0 | 2023-06-29 15:31:35 |          99248 |          0 |            8 | -0.00019094390964022743 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | orders     |                | O_ORDERSTATUS   |        0 | 2023-06-29 15:31:35 |              3 |          0 |            2 |     0.47400549128297814 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | orders     |                | O_TOTALPRICE    |        0 | 2023-06-29 15:31:34 |        1500000 |          0 |            9 |  -0.0007488532030123949 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | orders     |                | O_ORDERDATE     |        0 | 2023-06-29 15:31:35 |           2406 |          0 |            8 | -0.00019817306578388436 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | orders     |                | O_ORDERPRIORITY |        0 | 2023-06-29 15:31:35 |              5 |          0 |          9.4 |       0.203544452597151 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | orders     |                | O_CLERK         |        0 | 2023-06-29 15:31:35 |           1000 |          0 |           16 |   0.0039012046719564043 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | orders     |                | O_SHIPPRIORITY  |        0 | 2023-06-29 15:31:34 |              1 |          0 |            8 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | orders     |                | O_COMMENT       |        0 | 2023-06-29 15:31:35 |        1489408 |          0 |        49.76 |   0.0033236463230936334 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | part       |                | P_PARTKEY       |        0 | 2023-06-29 15:31:35 |         196960 |          0 |            8 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | part       |                | P_NAME          |        0 | 2023-06-29 15:31:35 |         198848 |          0 |        33.75 |   -0.001833420074338086 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | part       |                | P_MFGR          |        0 | 2023-06-29 15:31:35 |              5 |          0 |           15 |     0.20275777841067186 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | part       |                | P_BRAND         |        0 | 2023-06-29 15:31:35 |             25 |          0 |            9 |      0.0429150290271412 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | part       |                | P_TYPE          |        0 | 2023-06-29 15:31:35 |            150 |          0 |         21.6 |    0.009414809430815557 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | part       |                | P_SIZE          |        0 | 2023-06-29 15:31:35 |             50 |          0 |            8 |    0.026207841871047584 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | part       |                | P_CONTAINER     |        0 | 2023-06-29 15:31:35 |             40 |          0 |         8.57 |     0.02625445731635424 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | part       |                | P_RETAILPRICE   |        0 | 2023-06-29 15:31:35 |          21588 |          0 |            9 |     0.19158460235345143 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | part       |                | P_COMMENT       |        0 | 2023-06-29 15:31:35 |         126992 |          0 |        14.51 |   -0.002066606682605673 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | partsupp   |                | PS_PARTKEY      |        0 | 2023-06-29 15:31:35 |         196960 |          0 |            8 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | partsupp   |                | PS_SUPPKEY      |        0 | 2023-06-29 15:31:35 |          10000 |          0 |            8 |   0.0023089805006221175 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | partsupp   |                | PS_AVAILQTY     |        0 | 2023-06-29 15:31:35 |           9999 |          0 |            8 |   0.0009900264907642192 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | partsupp   |                | PS_SUPPLYCOST   |        0 | 2023-06-29 15:31:35 |          98032 |          0 |            9 |  -0.0012195581771791244 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | partsupp   |                | PS_COMMENT      |        0 | 2023-06-29 15:31:35 |         792192 |          0 |       125.52 |  -0.0018664672457200451 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | partsupp   |                | PRIMARY         |        1 | 2023-06-29 15:31:35 |         795904 |          0 |            0 |                       0 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | region     |                | R_REGIONKEY     |        0 | 2023-06-29 15:31:35 |              5 |          0 |            8 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | region     |                | R_NAME          |        0 | 2023-06-29 15:31:35 |              5 |          0 |          7.8 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | region     |                | R_COMMENT       |        0 | 2023-06-29 15:31:35 |              5 |          0 |         67.4 |                     0.6 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | supplier   |                | S_SUPPKEY       |        0 | 2023-06-29 15:31:36 |          10000 |          0 |            8 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | supplier   |                | S_NAME          |        0 | 2023-06-29 15:31:36 |          10000 |          0 |           19 |                       1 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | supplier   |                | S_ADDRESS       |        0 | 2023-06-29 15:31:36 |          10000 |          0 |        25.98 |    0.001058393914583939 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | supplier   |                | S_NATIONKEY     |        0 | 2023-06-29 15:31:36 |             25 |          0 |            8 |     0.04594194903941949 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | supplier   |                | S_PHONE         |        0 | 2023-06-29 15:31:36 |          10000 |          0 |           16 |   0.0064639693486396935 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | supplier   |                | S_ACCTBAL       |        0 | 2023-06-29 15:31:36 |           9955 |          0 |            9 |    0.015707805673078057 | allEvicted  |               0 |              0 |              0 |             0 |
| tpch    | supplier   |                | S_COMMENT       |        0 | 2023-06-29 15:31:35 |          10000 |          0 |        64.06 |   -0.008585964205859642 | allEvicted  |               0 |              0 |              0 |             0 |
+---------+------------+----------------+-----------------+----------+---------------------+----------------+------------+--------------+-------------------------+-------------+-----------------+----------------+----------------+---------------+
'''

result = parse_text(histograms)
print(result)  

