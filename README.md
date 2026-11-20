# Hybrid power system analysis

从 hybrid_power_system_analysis/src/hybrid_power_system_analysis 迁移的独立程序代码，包含：

- AC/DC/混合电力潮流：lfcore
- AC/DC/混合状态估计：secore
- 多能流模型：model
- 事故分析：cacore
- 拓扑 SVG 绘图：drawcore
- 频率响应仿真：simu
- E-file 读取与单位系统：根目录模块

代码保留原有的顶层模块接口（例如 from lfcore.ac_lf import ...），在仓库根目录运行脚本或将仓库根目录加入 PYTHONPATH 即可使用。

## 数据

案例数据放在独立工作区。仓库的 data/lf/lf.para、data/se/se.para 提供算法默认参数，默认参数文件缺失时使用同值内置参数；显式指定的其他参数文件缺失仍报错。

## 工作区 SQLite 入口

在 D:\codex\analysis 执行：

```powershell
python workspace_analysis.py --root D:\codex\workspace --runtime realtime --scenario qinling --task all
```

Python 调用方式：

```python
from pathlib import Path
from workspace import Workspace
from workspace_analysis import run_workspace

workspace = Workspace(Path(r"D:\codex\workspace"), runtime="realtime", scenario="qinling")
outcomes = run_workspace(workspace, tasks=("se", "lf", "ca"))
```

工作目录由 root / runtime / scenario 组成，默认 D:\codex\workspace\realtime\qinling。
每次调用独立解析路径，不改变全局当前目录，多个场景互不影响。

| 用途 | 相对工作目录路径 |
| --- | --- |
| 模型（只读） | sqldb/model/devices.sqlite3 |
| SE 实测来源（只读） | sqldb/scada/scdfe_c.sqlite3 |
| SE 测点映射（只读） | sqldb/scada/point_catalog.json |
| SE 结果 | sqldb/pas/se.sqlite3 |
| LF 结果 | sqldb/pas/lf.sqlite3 |
| CA 结果 | sqldb/pas/ca.sqlite3 |

--task 支持 se、lf、ca、all；--state 是 --runtime 的别名。
--scada 和 --point-map 可覆盖 SCADA 库和映射文件。
不自动把 qingling 改写为 qinling，目录名称严格使用输入参数。

SE 按 point_catalog 的 model_table + model_idx 关联设备，rtu_id + pnt_no 读取
fes_yc_define.value；核对设备名称和 dev_id，并按模型 Model 的 u_unit/p_unit/i_unit 换算。
默认 weight=1（原测点单位的逆方差），换算后同步调整权重，映射项可指定 weight。
质量码必须为 0，refresh_time 为 Unix 毫秒，默认不超过 120 秒（--max-age 可调）。
空时间戳、未来时间、非有限数、无效质量、过期测点、关联不符和不支持的类型均记录拒用原因。
不把 seed_value、设定值回读、SOC、频率等作为电气状态量测。
平衡三相模型使用 uab/ia（以及端口对应字段）作为电压/电流代表量，不重复纳入其余两相。
ACRealBs/DCRealBs 的电压映射到对应电气节点。当前 SCADA 适配器只接入电气量测；
氢气等流体实测尚未映射，不能将流体模型的约束或伪量测当作现场实测。

没有有效量测时，SE 写入 input_unavailable，不用模型值冒充实测。
LF/CA 使用模型库的运行状态和设定值；本入口不向模型库回写实测或计算值。
CA 使用 --limit 指定原格式 .calmt（物理单位），--no-n1 可只检查基态。
缺少限值文件时仍执行基态收敛和 N-1 失负荷分析，但标记 partial，不能据此认定全部设备未越限。

模型读取支持普通命名列 SQLite 和 RTDB 内嵌 schema 的 c0/c1 列映射。
每次调用通过 SQLite 在线备份获取包含已提交 WAL 的一致模型快照，运行结束删除临时快照；
无跨次网络缓存。原有 AC/DC/混合 LF、SE 的参数签名不变，模型/量测文件参数可直接传 .sqlite3。

每个结果库包含 analysis_run（场景、运行态、UTC 起止时间、状态、输入哈希、摘要）、
analysis_row（run_id、category、row_key、data_json）、latest_run 视图和 se_result/lf_result/ca_result 视图。
LF 保存节点、支路、电源、负荷、转换器及流体结果；SE 保存状态向量、统计量、量测分类和 SCADA 映射审计；
CA 保存 N-0/N-1 风险和扫描统计。输入模型元数据及单位随结果保存，电气结果使用内核标幺值。
每次结果和运行记录在同一事务中追加，保留历史；latest_run 始终反映最新尝试，失败不会冒用上次成功结果。
三个数据库各自原子提交。任一任务失败仍继续其他任务，全部 success 才返回进程退出码 0。

```sql
SELECT status, message, summary_json FROM latest_run;
SELECT category, row_key, data_json FROM lf_result;
SELECT * FROM analysis_row WHERE run_id = 1;
```

## 验证

    python -m py_compile (Get-ChildItem -Recurse -Filter *.py)
    python -c "from lfcore.ac_lf import ACPowerFlowCalc; from lfcore.dc_lf import DCPowerFlowCalc; from lfcore.hybrid_lf import HybridPowerFlowCalc; from secore.ac_se import ACStateEstimator; from secore.dc_se import DCStateEstimator; from secore.hybrid_se import HybridStateEstimator; print('interfaces import ok')"

工作区入口测试：python -m pytest tests/test_workspace_analysis.py -q。
SQLite 接入仅使用 Python 标准库，数值依赖仍为 numpy/scipy；可选稀疏求解器沿用内核配置。
