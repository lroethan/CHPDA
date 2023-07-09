import numpy as np
import pickle
import sys

sys.path.append("/home/ubuntu/CODE/CHPDA/")
import model.model_count_constraint as dqn_fc
import model.model_storage_constraint as dqn_fs
import util.tidb_connector as ti_conn
from argparse import ArgumentParser


N_WORKLOAD = 60


def get_args():
    parser = ArgumentParser(description=None)
    parser.add_argument("--database", default="tpcds", type=str, help="database name")
    parser.add_argument("--seed", default=1, type=int, help="seed random # generators")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--epsilon", default=0.9, type=float, help="epsilon")
    parser.add_argument("--q_iterations", default=9, type=int, help="Q-iteration")
    parser.add_argument("--u_iterations", default=3, type=int, help="U-iteration")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--discount_factor", default=0.9, type=float, help="discount factor")
    parser.add_argument("--num_episodes", default=800, type=int, help="number of episodes")
    parser.add_argument("--replay_start", default=400, type=int, help="when to start experience replay")
    parser.add_argument("--memory_capacity", default=800, type=int, help="replay buffer size")
    parser.add_argument("--w_path", default="entry/tpcds.pickle", type=str, help="workload path")
    parser.add_argument("--c_path", default="entry/hand_cands.pickle", type=str, help="candidate path")
    parser.add_argument("--count", default=5, type=int, help="count constraint")

    return parser.parse_args()


def run_dqn(args, is_dnn, is_ps, is_double, a):
    count_constraint = args.count
    args.exp_name = f"c_constraint_{count_constraint}"

    print("Loading workload...")
    with open(args.w_path, "rb") as f:
        workload = pickle.load(f)
    print("Loading candidate indexes...")
    with open(args.c_path, "rb") as f:
        candidate = pickle.load(f)

    agent = dqn_fc.DQN(workload[:], candidate, "hypo", args, is_dnn, is_ps, is_double, a)

    one_hot_indexes = agent.train(False, count_constraint)
    selected_indexes = [candidate[i] for i, flag in enumerate(one_hot_indexes) if flag == 1.0]

    return selected_indexes


def get_performance(selected_indexes, frequencies, workload_path, database):
    frequencies = np.array(frequencies) / np.array(frequencies).sum()

    with open(workload_path, "rb") as wf:
        workload = pickle.load(wf)

    tidb_client = ti_conn.TiDBDatabaseConnector(database)

    original_cost = (np.array(tidb_client.get_queries_cost(workload)) * frequencies).sum()
    print(f"Original cost: {original_cost}")

    for index in selected_indexes:
        tidb_client.execute_create_hypo(index)
    new_cost = (np.array(tidb_client.get_queries_cost(workload)) * frequencies).sum()
    new_storage_cost = sum([tidb_client.get_index_size(index) for index in selected_indexes])
    print(f"New cost: {new_cost}")
    print(f"Index storage cost: {new_storage_cost}")

    cost_reduction = (original_cost - new_cost) / original_cost
    print("===============COST REDUCTION=================")
    return cost_reduction


def main(args):
    selected_indexes = run_dqn(args, False, True, True, 0)
    frequencies = [1] * args.num_workload
    print("===============SELECTED INDEXES=================")
    print(selected_indexes)
    print(get_performance(selected_indexes, frequencies, args.w_path, args.database))


if __name__ == "__main__":
    args = get_args()
    
    if args.database == "tpcds":
        args.num_workload = 60
    elif args.database == "tpch":
        args.num_workload = 14

    main(args)
