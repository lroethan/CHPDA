import numpy as np
import pickle

import Model.Model3DQNFixCount as dqn_fc
import Model.Model3DQNFixStorage as dqn_fs
import Utility.TiDB as tihypo


def run_dqn(is_fix_count, conf, x, is_dnn, is_ps, is_double, a):
    conf['NAME'] = f"MA_9{x}"
    print("Loading workload...")
    with open("workload.pickle", "rb") as wf:
        workload = pickle.load(wf)
    print("Loading candidate...")
    with open("cands.pickle", "rb") as cf:
        index_candidates = pickle.load(cf)
    print(workload)
    if is_fix_count:
        agent = dqn_fc.DQN(workload[:], index_candidates, "hypo", conf, is_dnn, is_ps, is_double, a)
    else:
        agent = dqn_fs.DQN(workload, index_candidates, "hypo", conf)
    indexes, storages = agent.train(False, x)
    selected_indexes = [index_candidates[i] for i, idx in enumerate(indexes) if idx == 1.0]
    return selected_indexes


def get_performance(selected_indexes, frequencies):
    print(frequencies)
    frequencies = np.array(frequencies) / np.array(frequencies).sum()
    with open("workload.pickle", "rb") as wf:
        workload = pickle.load(wf)
    tidb_client = tihypo.TiDBHypo()
    tidb_client.delete_indexes()
    cost1 = (np.array(tidb_client.get_queries_cost(workload)) * frequencies).sum()
    print(cost1)
    for index in selected_indexes:
        tidb_client.execute_create_hypo(index)
    cost2 = (np.array(tidb_client.get_queries_cost(workload)) * frequencies).sum()
    print(cost2)
    tidb_client.delete_indexes()
    print((cost1 - cost2) / cost1)


conf21 = {
    "LR": 0.002,
    "EPISILO": 0.97,
    "Q_ITERATION": 200,
    "U_ITERATION": 5,
    "BATCH_SIZE": 64,
    "GAMMA": 0.95,
    "EPISODES": 800,
    "LEARNING_START": 1000,
    "MEMORY_CAPACITY": 20000,
}

conf = {
    "LR": 0.1,
    "EPISILO": 0.9,
    "Q_ITERATION": 9,
    "U_ITERATION": 3,
    "BATCH_SIZE": 8,
    "GAMMA": 0.9,
    "EPISODES": 800,
    "LEARNING_START": 400,
    "MEMORY_CAPACITY": 800,
}


# is_fixcount == True, constraint is the index number
# is_fixcount == False, constraint is the index storage unit
def entry(is_fix_count, constraint):
    if is_fix_count:
        selected_indexes = run_dqn(is_fix_count, conf21, constraint, False, True, True, 0)
    else:
        selected_indexes = run_dqn(is_fix_count, conf, constraint, False, False, False, 0)
    frequencies = [1659, 1301, 1190, 1741, 1688, 1242, 1999, 1808, 1433, 1083, 1796, 1266, 1046, 1353]
    get_performance(selected_indexes, frequencies)


entry(True, 4)