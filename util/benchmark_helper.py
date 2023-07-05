import os
import pymysql

def setup_database(db_name):
    # 连接到 TiDB
    conn = pymysql.connect(
        host='localhost',
        port=4000,
        user='root',
        password='',
        local_infile=True,
        charset='utf8mb4'
    )

    # # 检查是否存在 db_name 数据库，如果不存在则创建
    cursor = conn.cursor()
    # cursor.execute(f"SHOW DATABASES LIKE '{db_name}'")
    # result = cursor.fetchone()

    # if not result:
    #     cursor.execute(f"CREATE DATABASE {db_name}")
    #     print(f"Database '{db_name}' created.")

    # # 执行 schema.sql 文件建立数据表
    # schema_file = os.path.join('../workload', db_name, 'schema.sql')
    # print(schema_file)
    # if os.path.exists(schema_file):
    #     with open(schema_file, 'r') as file:
    #         schema_sql = file.read()
    #     try:
    #         cursor.execute(schema_sql)
    #         print("Table creation successful.")
    #     except Exception as e:
    #         print(f"Table creation failed: {e}")

    # 加载 stats 下的所有 json 文件
    stats_dir = os.path.join('/home/ubuntu/CODE/CHPDA/workload', db_name, 'stats')
    stats_files = os.listdir(stats_dir)
    for file in stats_files:
        if file.endswith('.json'):
            stats_file_path = os.path.join(stats_dir, file)
            load_stats_query = f"LOAD STATS '{stats_file_path}'"
            try:
                cursor.execute(load_stats_query)
                print(f"Loaded stats from '{stats_file_path}'.")
            except Exception as e:
                print(f"Failed to load stats from '{stats_file_path}': {e}")

    # 执行 show stats_meta
    cursor.execute("SHOW STATS_META")
    stats_meta_results = cursor.fetchall()
    for result in stats_meta_results:
        print(result)

    # 关闭连接
    cursor.close()
    conn.close()

# 调用函数并传入 db_name 参数
db_name = 'tpcds'
setup_database(db_name)