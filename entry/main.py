import numpy as np
import pickle
import sys
sys.path.append("/home/ubuntu/CODE/CHPDA/")
import model.model_count_constraint as dqn_fc
import model.model_storage_constraint as dqn_fs
import util.tidb_connector as ti_conn


def run_dqn(is_fix_count, hyperparameter, x, is_dnn, is_ps, is_double, a, workload_path, cands_path):
    # Set name for experiment
    hyperparameter['NAME'] = f"TEST_{x}"

    # Load workload and candidate indexes
    print("Loading workload...")
    with open(workload_path, "rb") as workload_file:
#         workload2 = """select
#   l_returnflag,
#   l_linestatus,
#   sum(l_quantity) as sum_qty,
#   sum(l_extendedprice) as sum_base_price,
#   sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
#   sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
#   avg(l_quantity) as avg_qty,
#   avg(l_extendedprice) as avg_price,
#   avg(l_discount) as avg_disc,
#   count(*) as count_order
# from
#   lineitem
# where
#   l_shipdate <= '1998-09-02'
# group by
#   l_returnflag,
#   l_linestatus
# order by
#   l_returnflag,
#   l_linestatus;"""
#         workload = [workload2]
        workload = pickle.load(workload_file)
        
        print(workload)
    print("Loading candidate indexes...")
    with open(cands_path, "rb") as candidate_file:
        static_candidates = pickle.load(candidate_file)

    # Choose the appropriate DQN agent
    if is_fix_count:
        agent = dqn_fc.DQN(workload[:], static_candidates, "hypo", hyperparameter, is_dnn, is_ps, is_double, a)
    else:
        agent = dqn_fs.DQN(workload, static_candidates, "hypo", hyperparameter)

    # Train the agent and select the indexes
    one_hot_indexes = agent.train(False, x)
    selected_indexes = [static_candidates[i] for i, flag in enumerate(one_hot_indexes) if flag == 1.0]

    return selected_indexes


def get_performance(selected_indexes, frequencies, workload_path):
    # Normalize the frequencies
    frequencies = np.array(frequencies) / np.array(frequencies).sum()

    # Load the workload and connect to TiDB
    with open(workload_path, "rb") as wf:
        workload = pickle.load(wf)
    tidb_client = ti_conn.TiDBDatabaseConnector("tpch")

    # Calculate the original cost
    original_cost = (np.array(tidb_client.get_queries_cost(workload)) * frequencies).sum()
    # original_storage_cost = [tidb_client.get_indexe_size(index) for index in selected_indexes].sum()
    print(f"Original cost: {original_cost}")

    # Create the hypothetical indexes and calculate the new cost
    for index in selected_indexes:
        tidb_client.execute_create_hypo(index)
    new_cost = (np.array(tidb_client.get_queries_cost(workload)) * frequencies).sum()
    new_storage_cost = sum([tidb_client.get_indexe_size(index) for index in selected_indexes])
    print(f"New cost: {new_cost}")
    print(f"Index storage cost: {new_storage_cost}")

    # Calculate the cost reduction and return it
    cost_reduction = (original_cost - new_cost) / original_cost
    print("===============COST REDUCTION=================")
    return cost_reduction



# conf21 = {
#     "LR": 0.002,
#     "EPISILO": 0.97,
#     "Q_ITERATION": 200,
#     "U_ITERATION": 5,
#     "BATCH_SIZE": 64,
#     "GAMMA": 0.95,
#     "EPISODES": 800,
#     "LEARNING_START": 1000,
#     "MEMORY_CAPACITY": 20000,
# }

test_count_conf = {
    "LR": 0.1,
    "EPISILO": 0.9,
    "Q_ITERATION": 200,
    "U_ITERATION": 5,
    "BATCH_SIZE": 64,
    "GAMMA": 0.95,
    "EPISODES": 50,
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


def entry(is_fix_count: bool, constraint, workload_path, cands_path):
    """
    Run the DQN algorithm with the specified constraint.

    Args:
        is_fix_count (bool): True if the constraint is the index number, False if the constraint is the index storage budget.
        constraint: The constraint value.
        workload_path: The path to the workload file.
        cands_path: The path to the candidate indexes file.

    Returns:
        selected_indexes: The selected indexes.

    """
    if is_fix_count:
        selected_indexes = run_dqn(is_fix_count, test_count_conf, constraint, False, True, True, 0, workload_path, cands_path)
    else:
        selected_indexes = run_dqn(is_fix_count, conf, constraint, False, False, False, 0, workload_path, cands_path)

    # frequencies = [1659, 1301, 1190, 1741, 1688, 1242, 1999, 1808, 1433, 1083, 1796, 1266, 1046, 1353]
    frequencies = [1] * 14
    print("===============SELECTED INDEXES=================")
    print(selected_indexes)
    print(get_performance(selected_indexes, frequencies, workload_path))


if __name__ == '__main__':
    
    WORKLOAD_PATH = "entry/workload.pickle"
    CANDS_PATH = "entry/cands.pickle"
    # CANDS_PATH = "entry/cands2.pickle"
    
    entry(True, 4, WORKLOAD_PATH, CANDS_PATH)