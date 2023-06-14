import numpy as np
import pickle
import sys
sys.path.append("/home/ubuntu/CODE/CHPDA/")
import model.model_count_constraint as dqn_fc
import model.model_storage_constraint as dqn_fs
import util.tidb_connector as ti_conn


def run_dqn(is_fix_count, hyperparameter, x, is_dnn, is_ps, is_double, a):
    # Set name for experiment
    hyperparameter['NAME'] = f"TEST_{x}"

    # Load workload and candidate indexes
    print("Loading workload...")
    with open("entry/workload.pickle", "rb") as wf:
        workload = pickle.load(wf)
    print("Loading candidate indexes...")
    with open("entry/cands.pickle", "rb") as cf:
        index_candidates = pickle.load(cf)

    # Choose the appropriate DQN agent
    if is_fix_count:
        agent = dqn_fc.DQN(workload[:], index_candidates, "hypo", hyperparameter, is_dnn, is_ps, is_double, a)
    else:
        agent = dqn_fs.DQN(workload, index_candidates, "hypo", hyperparameter)

    # Train the agent and select the indexes
    indexes = agent.train(False, x)
    selected_indexes = [index_candidates[i] for i, idx in enumerate(indexes) if idx == 1.0]

    return selected_indexes


def get_performance(selected_indexes, frequencies):
    # Normalize the frequencies
    frequencies = np.array(frequencies) / np.array(frequencies).sum()

    # Load the workload and connect to TiDB
    with open("entry/workload.pickle", "rb") as wf:
        workload = pickle.load(wf)
    tidb_client = ti_conn.TiDBDatabaseConnector("tpch")

    # Calculate the original cost
    original_cost = (np.array(tidb_client.get_queries_cost(workload)) * frequencies).sum()
    print(f"Original cost: {original_cost}")

    # Create the hypothetical indexes and calculate the new cost
    for index in selected_indexes:
        tidb_client.execute_create_hypo(index)
    new_cost = (np.array(tidb_client.get_queries_cost(workload)) * frequencies).sum()
    print(f"New cost: {new_cost}")

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

conf21 = {
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


def entry(is_fix_count: bool, constraint):
    """
    Run the DQN algorithm with the specified constraint.

    Args:
        is_fix_count (bool): True if the constraint is the index number, False if the constraint is the index storage budget.
        constraint: The constraint value.

    Returns:
        selected_indexes: The selected indexes.

    """
    if is_fix_count:
        selected_indexes = run_dqn(is_fix_count, conf21, constraint, False, True, True, 0)
    else:
        selected_indexes = run_dqn(is_fix_count, conf, constraint, False, False, False, 0)

    frequencies = [1659, 1301, 1190, 1741, 1688, 1242, 1999, 1808, 1433, 1083, 1796, 1266, 1046, 1353]
    print(get_performance(selected_indexes, frequencies))

entry(True, 3)