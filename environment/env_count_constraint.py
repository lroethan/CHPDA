from typing import List

import numpy as np

import util.tidb_connector as tidb


class Env:
    def __init__(self, workload, candidates, mode, a, database, num_workload):
        self.workload = workload
        self.candidates = candidates

        # Create real/hypothetical index
        self.mode = mode
        self.conn = tidb.TiDBDatabaseConnector(db_name=database)
        self._frequencies = [1] * num_workload
        self.frequencies = np.array(self._frequencies) / np.array(self._frequencies).sum()

        # Initial state
        self.init_cost = np.array(self.conn.get_queries_cost(workload)) * self.frequencies
        self.init_cost_sum = self.init_cost.sum()
        self.init_state = np.append(self.frequencies, np.zeros(len(candidates)))  # 全是 0 的 candidates 0-1 向量

        # Final state
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum

        # Utility info
        self.index_oids = np.zeros(len(candidates))
        self.performance_gain = np.zeros(len(candidates))
        self.n_exist_idx = 0
        self.current_index = np.zeros(len(candidates))
        self.current_index_storage = np.zeros(len(candidates))

        # Monitor info
        self.cost_trace_overall = list()
        self.index_trace_overall = list()
        self.storage_trace_overall = list()
        self.min_cost_overall = list()
        self.min_indexes_overall = list()
        self.current_min_cost = (np.array(self.conn.get_queries_cost(workload)) * 0.1 * self.frequencies).sum()
        self.current_min_index = np.zeros(len(candidates), dtype=np.cfloat)

        self.current_storage_sum = 0
        self.max_count = 0
        self.alpha = a

        self.pre_create = []


    def step(self, action):
        action = action[0]

        # recommended index is existed
        if self.current_index[action] != 0.0:
            return self.last_state, 0, False

        idx_id, idx_oid = self.conn.execute_create_hypo(self.candidates[action])
        self.current_index[action] = 1.0

        # this action influence the envrionment
        self.current_storage_sum = self.conn.get_index_size(idx_oid)
        self.n_exist_idx += 1
        l_current_cost = np.array(self.conn.get_queries_cost(self.workload)) * self.frequencies
        current_cost_sum = l_current_cost.sum()

        self.last_cost = l_current_cost
        self.last_state = np.append(self.frequencies, self.current_index)

        # reward is cost reduction
        reward = (self.init_cost_sum - current_cost_sum) / self.init_cost_sum
        self.last_cost_sum = current_cost_sum

        # constraint
        done = False
        if self.n_exist_idx >= self.max_count or self.current_storage_sum >= 284808301.72:
            done = True

        self.cost_trace_overall.append(current_cost_sum)
        self.index_trace_overall.append(self.current_index)
        self.storage_trace_overall.append(self.current_storage_sum)

        return self.last_state, reward, done

    def reset(self):
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum
        self.current_storage_sum = 0
        # self.index_trace_overall.append(self.currenct_index)
        self.index_oids = np.zeros(len(self.candidates))
        self.performance_gain = np.zeros(len(self.candidates))
        self.n_exist_idx = 0
        self.current_min_cost = np.array(self.conn.get_queries_cost(self.workload)).sum()
        self.current_min_index = np.zeros(len(self.candidates))
        self.current_index = np.zeros(len(self.candidates))
        self.current_index_storage = np.zeros(len(self.candidates))
        self.conn.delete_indexes()
        # self.cost_trace_overall.append(self.last_cost_sum)
        if len(self.pre_create) > 0:
            print("x")
            for i in self.pre_create:
                self.conn.execute_create_hypo(i)
            self.init_cost_sum = (np.array(self.conn.get_queries_cost(self.workload)) * self.frequencies).sum()
            self.last_cost_sum = self.init_cost_sum
        return self.last_state
