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


def hist_tuple_to_dict(hist_tuple):
    '''
    {'customer#C_CUSTKEY': [149568, 0, 8.0], 'customer#C_NAME': [148224, 0, 19.0]}
    '''
    result = {}
    for tpl in hist_tuple:
        key = tpl[1] + '#' + tpl[3]
        value = [tpl[6], tpl[7], tpl[8]]
        result[key] = value
    return result


def meta_tuple_to_dict(hist_tuple):
    '''
    {'customer': 150000, 'lineitem': 8143998, 'nation': 25, 'orders': 1500000, 'part': 200000, 'partsupp': 800000, 'region': 5, 'supplier': 10000}
    '''
    result = {}
    for tpl in hist_tuple:
        key = tpl[1]
        value = tpl[-1]
        result[key] = value
    return result