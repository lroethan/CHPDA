from typing import List

import numpy as np

import util.tidb_connector as tidb

max_index_size = 4


class Env:
    def __init__(self, workload, candidates, mode, a):
        self.workload = workload
        self.candidates = candidates

        # Create real/hypothetical index
        self.mode = mode
        self.conn = tidb.TiDBDatabaseConnector(db_name="tpch")
        self.conn_for_checkout = tidb.TiDBDatabaseConnector(db_name="tpch")
        self._frequencies = [1265, 897, 643, 1190, 521, 1688, 778, 1999, 1690, 1433, 1796, 1266, 1046, 1353]
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
        self.current_index_count = 0
        self.current_index = np.zeros(len(candidates))
        self.current_index_storage = np.zeros(len(candidates))

        # Monitor info
        self.cost_trace_overall = list()
        self.index_trace_overall = list()
        self.min_cost_overall = list()
        self.min_indexes_overall = list()
        self.current_min_cost = (np.array(self.conn.get_queries_cost(workload)) * 0.1 * self.frequencies).sum()
        self.current_min_index = np.zeros(len(candidates), dtype=np.cfloat)

        self.current_storage_sum = 0
        self.max_count = 0
        self.alpha = a

        self.pre_create = []

    @property
    def checkout(self):
        pre_index_set = []
        while True:
            current_max_benefit = 0
            current_index = None
            current_index_len = 0
            original_workload_cost = (
                np.array(self.conn_for_checkout.get_queries_cost(self.workload)) * self.frequencies
            ).sum()

            for index in self.candidates:
                ident, _ = self.conn_for_checkout.execute_create_hypo(index)
                current_workload_cost = (
                    np.array(self.conn_for_checkout.get_queries_cost(self.workload)) * self.frequencies
                ).sum()
                benefit = (original_workload_cost - current_workload_cost) / original_workload_cost
                if benefit > 0.4 and current_max_benefit < benefit:
                    current_max_benefit = benefit
                    current_index = index
                    current_index_len = current_index_len  # 这个暂时没用上
                self.conn_for_checkout.execute_delete_hypo(ident)

            if current_index is None:
                break
            pre_index_set.append(current_index)
            self.conn_for_checkout.execute_create_hypo(current_index)
        # pre_index_set = ['lineitem#l_orderkey,l_shipdate', 'lineitem#l_partkey,l_orderkey', 'lineitem#l_receiptdate',
        # 'lineitem#l_shipdate,l_partkey', 'lineitem#l_suppkey,l_commitdate'] pre_index_set = ['lineitem#l_orderkey,
        # l_suppkey', 'lineitem#l_partkey,l_suppkey', 'lineitem#l_receiptdate', 'lineitem#l_shipdate,l_discount',
        # 'lineitem#l_suppkey,l_commitdate'] pre_index_set.append('lineitem#l_orderkey')
        self.pre_create = pre_index_set
        # self.conn_for_checkout.delete_indexes()
        self.max_count -= len(self.pre_create)

    def step(self, action):
        action = action[0]  # 这里的 action 是一个 list，里面只有一个元素，所以取第一个元素，且为整数

        if self.current_index[action] != 0.0:
            # self.cost_trace_overall.append(self.last_cost_sum)
            # self.index_trace_overall.append(self.currenct_index)
            return self.last_state, 0, False

        # print("====================")
        # print(self.candidates)
        # print("====================")
        idx_oid, _ = self.conn.execute_create_hypo(self.candidates[action])

        self.current_index[action] = 1.0
        oids: List[str] = list()
        oids.append(idx_oid)
        storage_cost = self.conn.get_storage_cost(oids)[0]  # TODO
        # print(storage_cost)
        self.current_storage_sum += storage_cost
        self.current_index_storage[action] = storage_cost
        self.current_index_count += 1

        # reward & performance gain
        current_cost_info = np.array(self.conn.get_queries_cost(self.workload)) * self.frequencies
        current_cost_sum = current_cost_info.sum()
        # performance_gain_current = self.init_cost_sum - current_cost_sum
        # performance_gain_current = (self.last_cost_sum - current_cost_sum)/self.last_cost_sum
        # performance_gain_avg = performance_gain_current.round(1)
        # self.performance_gain[action] = performance_gain_avg
        # monitor info
        # self.cost_trace_overall.append(current_cost_sum)

        # update
        self.last_cost = current_cost_info
        # state = (self.init_cost - current_cost_info)/self.init_cost
        self.last_state = np.append(self.frequencies, self.current_index)
        deltac0 = (self.init_cost_sum - current_cost_sum) / self.init_cost_sum
        deltac1 = (self.last_cost_sum - current_cost_sum) / self.init_cost_sum
        # print(deltac0)
        """deltac0 = max(0.000003, deltac0)
        if deltac0 == 0.000003:
            reward = -10
        else:
            reward = math.log(0.0003, deltac0)"""
        b = 1 - self.alpha
        # reward = deltac0
        print(deltac0)
        reward = self.alpha * deltac0 * 100 + b * deltac1 * 100
        # reward = deltac0
        # reward = math.log(0.99, deltac0)
        """deltac0 = self.init_cost_sum/current_cost_sum
        deltac1 = self.last_cost_sum/current_cost_sum
        reward = math.log(deltac0,10)"""
        self.last_cost_sum = current_cost_sum
        if self.current_index_count >= self.max_count:  # TODO
            self.cost_trace_overall.append(current_cost_sum)
            self.index_trace_overall.append(self.current_index)
            return self.last_state, reward, True
        else:
            return self.last_state, reward, False
            # re5 return self.last_state, reward, False

    def reset(self):
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum
        # self.index_trace_overall.append(self.currenct_index)
        self.index_oids = np.zeros(len(self.candidates))
        self.performance_gain = np.zeros(len(self.candidates))
        self.current_index_count = 0
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
