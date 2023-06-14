import math
import sys

import numpy as np

import util.tidb_connector as ti_conn


class Env:
    def __init__(self, workload, candidates, mode):
        self.workload = workload
        self.candidates = candidates

        # Create real/hypothetical index
        self.db_client1 = ti_conn.TiDBDatabaseConnector(db_name='tpch')
        self.db_client1 = ti_conn.TiDBDatabaseConnector(db_name='tpch')

        self._frequencies = [1659, 1301, 1190, 1741, 1688, 1242, 1999, 1808, 1433, 1083, 1796, 1266, 1046, 1353]
        self.frequencies = np.array(self._frequencies) / np.array(self._frequencies).sum()

        # State info
        self.init_cost = np.array(self.db_client1.get_queries_cost(workload)) * self.frequencies
        self.init_cost_sum = self.init_cost.sum()
        self.init_state = np.append(self.frequencies, np.zeros((len(candidates),), dtype=np.float))
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum

        # util info
        self.index_oids = np.zeros((len(candidates),), dtype=np.int)
        self.performance_gain = np.zeros((len(candidates),), dtype=np.float)
        self.current_index_count = 0
        self.current_index = np.zeros((len(candidates),), dtype=np.float)
        self.current_index_storage = np.zeros((len(candidates),), dtype=np.float)

        # monitor info
        self.cost_trace_overall = list()
        self.index_trace_overall = list()
        self.storage_trace_overall = list()
        self.min_cost_overall = list()
        self.min_indexes_overall = list()
        self.current_min_cost = (np.array(self.db_client1.get_queries_cost(workload)) * 0.1 * self.frequencies).sum()
        self.current_min_index = np.zeros((len(candidates),), dtype=np.float)

        self.current_storage_sum = 0
        self.last_reward = 0
        self.last_perf_gain = sys.maxsize
        self.max_size = 0
        self.imp_count = 0

    def step(self, action):
        action = action[0]

        # Check if the current index is already chosen
        if self.current_index[action] != 0.0:
            return self.last_state, self.last_reward, False

        # Execute create_hypo to get the oid
        oid = self.db_client1.execute_create_hypo(self.candidates[action])
        oids = [oid]

        # Calculate storage cost
        storage_cost = self.db_client1.get_storage_cost(oids)[0]
        if len(self.candidates[action].split('#')[1].split(',')) == 3:
            storage_cost = storage_cost / 2

        # Calculate current cost and check for improvement
        current_cost_info = np.array(self.db_client1.get_queries_cost(self.workload)) * self.frequencies
        current_cost_sum = current_cost_info.sum()
        if current_cost_sum < self.last_cost_sum:
            self.imp_count += 1

        # Calculate reward and perform necessary updates
        perf_gain = self.init_cost_sum - current_cost_sum
        deltac0 = perf_gain / self.init_cost_sum
        deltac0 = max(0.00001, deltac0)  # 0.9 100
        if deltac0 == 0.00001:
            reward = -1.5
        else:
            reward = math.log(0.00001, deltac0)
        if reward > 0:
            reward += self.imp_count
        print(reward)

        # Update storage sum and check if maximum size reached
        self.current_storage_sum += storage_cost
        if self.current_storage_sum >= self.max_size:
            self.cost_trace_overall.append(self.last_cost_sum)
            self.index_trace_overall.append(self.current_index)
            self.storage_trace_overall.append(self.current_index_storage)
            return self.last_state, reward, True

        # Update index and other related variables
        self.index_oids[action] = oid
        self.current_index[action] = 1.0
        self.current_index_storage[action] = storage_cost
        self.current_index_count += 1
        self.last_cost = current_cost_info
        self.last_state = np.append(self.frequencies, self.current_index)
        self.last_cost_sum = current_cost_sum
        self.last_reward = reward

        return self.last_state, reward, False

    def reset(self):
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum
        self.index_oids = np.zeros((len(self.candidates),), dtype=np.int)
        self.performance_gain = np.zeros((len(self.candidates),), dtype=np.float)
        self.current_index_count = 0
        self.current_min_cost = np.array(self.db_client1.get_queries_cost(self.workload)).sum()
        self.current_min_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.current_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.current_index_storage = np.zeros((len(self.candidates),), dtype=np.float)
        self.db_client1.delete_indexes()
        self.last_reward = 0
        self.current_storage_sum = 0
        self.last_perf_gain = sys.maxsize
        self.imp_count = 0

        return self.last_state
