# Cross-granularity Hybrid Physical Design Advisor with Deep Reinforcement Learning

### 1. Preparation

1. Refer to the TiDB Quickstart guide located at [TiDB Quickstart](https://docs.pingcap.com/tidb/dev/quick-start-with-tidb) for the necessary pre-requisites before installing and running TiUP.
2. Run the following command to launch a TiUP playground environment with one instance of TiDB (--db 1), one instance of PD (--pd 1), and the latest nightly version of TiKV (--kv nightly):
```shell
tiup playground v7.1.0 --db 1 --pd 1 --kv nightly
```