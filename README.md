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

本次只迁移源程序目录中的 Python 代码，没有复制原工程的 data/、测试、缓存和运行输出。需要运行具体案例时，请通过 --model/--meas 指定 E-file，或将对应数据放在 data/ 目录下。

## 验证

    python -m py_compile (Get-ChildItem -Recurse -Filter *.py)
    python -c "from lfcore.ac_lf import ACPowerFlowCalc; from lfcore.dc_lf import DCPowerFlowCalc; from lfcore.hybrid_lf import HybridPowerFlowCalc; from secore.ac_se import ACStateEstimator; from secore.dc_se import DCStateEstimator; from secore.hybrid_se import HybridStateEstimator; print('interfaces import ok')"

依赖和具体案例数据按原工程的 pyproject.toml 与 data/ 配置准备。
