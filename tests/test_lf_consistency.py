"""Interface consistency, solver accuracy, and parser tests for AC, DC, and Hybrid LF."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest

import algorithm_parameters
from efile_read import _read_efile_rows
from lfcore.ac_lf import ACPowerFlowCalc
from lfcore.dc_lf import DCPowerFlowCalc
from lfcore.hybrid_lf import HybridPowerFlowCalc
from lfcore.solver_common import AdaptiveNewtonFactor


LF_PARAMS = algorithm_parameters.PowerFlowParameters(
    tol=1e-8,
    max_iter=50,
    min_voltage=0.1,
    divergence_threshold=1e6,
)
NL = chr(10)


def _gen_ac_e(n_bus: int, out_dir: Path, *, seed: int = 301, load_scale: float = 1.0) -> Path:
    rng = np.random.default_rng(seed)
    e_path = out_dir / f"ac_{n_bus}_{load_scale:.1f}.e"
    side = int(np.ceil(np.sqrt(n_bus)))
    lines = [
        "<Model>",
        "@ p_base u_unit p_unit i_unit",
        "# 100.0 kV MW kA",
        "</Model>",
        "<ACNode>",
        "@ idx name vbase voltage angle isl run_stat",
    ]
    for i in range(1, n_bus + 1):
        lines.append(f"# {i} bus_{i} 110.0 110.0 0.0 1 1")
    lines.append("</ACNode>")
    edges = []
    for i in range(1, n_bus + 1):
        if (i - 1) % side + 1 < side and i + 1 <= n_bus:
            edges.append((i, i + 1))
        if i + side <= n_bus:
            edges.append((i, i + side))
    lines += ["<ACBranch>", "@ idx name i_node j_node r x b run_stat"]
    for idx, (u, v) in enumerate(edges, 1):
        lines.append(
            f"# {idx} br_{idx} {u} {v} {0.005 + 0.005 * rng.random():.6f} "
            f"{0.02 + 0.03 * rng.random():.6f} {0.005 + 0.01 * rng.random():.6f} 1"
        )
    lines += ["</ACBranch>", "<ACGenerator>", "@ idx name node control_type p_set q_set v_set alpha run_stat"]
    lines.append("# 1 gen_1 1 SLACK 0.0 0.0 112.0 1.0 1")
    for g_idx, node in enumerate(range(20, n_bus + 1, 20), 2):
        lines.append(f"# {g_idx} gen_{g_idx} {node} PV {3.0 * load_scale:.2f} 0.0 110.5 1.0 1")
    lines += ["</ACGenerator>", "<ACLoad>", "@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat"]
    l_idx = 0
    for node in range(2, n_bus + 1):
        if node % 4 != 0:
            l_idx += 1
            lines.append(
                f"# {l_idx} load_{l_idx} {node} {0.25 * load_scale:.4f} 1.0 0.0 0.0 "
                f"{0.06 * load_scale:.4f} 1.0 0.0 0.0 1"
            )
    lines.append("</ACLoad>")
    e_path.write_text(NL.join(lines) + NL, encoding="utf8")
    return e_path


def _gen_dc_e(n_bus: int, out_dir: Path, *, seed: int = 302) -> Path:
    rng = np.random.default_rng(seed)
    e_path = out_dir / f"dc_{n_bus}.e"
    side = int(np.ceil(np.sqrt(n_bus)))
    lines = [
        "<Model>",
        "@ p_base u_unit p_unit i_unit",
        "# 100.0 kV MW kA",
        "</Model>",
        "<DCNode>",
        "@ idx name vbase voltage isl run_stat",
    ]
    for i in range(1, n_bus + 1):
        lines.append(f"# {i} dcbus_{i} 100.0 100.0 1 1")
    lines.append("</DCNode>")
    edges = []
    for i in range(1, n_bus + 1):
        if (i - 1) % side + 1 < side and i + 1 <= n_bus:
            edges.append((i, i + 1))
        if i + side <= n_bus:
            edges.append((i, i + side))
    lines += ["<DCBranch>", "@ idx name i_node j_node r run_stat"]
    for idx, (u, v) in enumerate(edges, 1):
        lines.append(f"# {idx} dcbr_{idx} {u} {v} {0.01 + 0.02 * rng.random():.6f} 1")
    lines += ["</DCBranch>", "<DCGenerator>", "@ idx name node control_type p_set v_set run_stat"]
    lines.append("# 1 dcgen_1 1 V 0.0 102.0 1")
    lines += ["</DCGenerator>", "<DCLoad>", "@ idx name node pbase pv0 pv1 pv2 run_stat"]
    l_idx = 0
    for node in range(2, n_bus + 1):
        if node % 4 != 0:
            l_idx += 1
            lines.append(f"# {l_idx} dcload_{l_idx} {node} 0.15 1.0 0.0 0.0 1")
    lines.append("</DCLoad>")
    e_path.write_text(NL.join(lines) + NL, encoding="utf8")
    return e_path


def _gen_hybrid_e(n_ac: int, n_dc: int, out_dir: Path) -> Path:
    ac_e = _gen_ac_e(n_ac, out_dir, seed=401)
    dc_e = _gen_dc_e(n_dc, out_dir, seed=402)
    out_dc = []
    in_m = False
    for line in dc_e.read_text(encoding="utf8").splitlines():
        if line.startswith("<Model>"):
            in_m = True
        elif line.startswith("</Model>"):
            in_m = False
        elif not in_m:
            out_dc.append(line)
    conv_lines = [
        "<DCACConverter>",
        "@ idx name ac_node dc_node r1 r2 ac_control_type dc_control_type p_ac_set p_dc_set q_ac_set v_ac_set v_dc_set run_stat",
        f"# 1 dcac_1 {min(n_ac, 4)} {min(n_dc, 4)} 0.005 0.005 PQ NONE 0.5 0.0 0.1 0.0 0.0 1",
        "</DCACConverter>",
    ]
    hyb_e = out_dir / f"hyb_{n_ac}_{n_dc}.e"
    hyb_e.write_text(ac_e.read_text(encoding="utf8") + NL.join(out_dc) + NL + NL.join(conv_lines) + NL, encoding="utf8")
    return hyb_e


def test_lf_init_and_public_api_consistency():
    """ACPowerFlowCalc, DCPowerFlowCalc, and HybridPowerFlowCalc must share identical core signatures."""
    classes = (ACPowerFlowCalc, DCPowerFlowCalc, HybridPowerFlowCalc)
    init_params = [
        [(p.name, p.kind) for p in inspect.signature(cls.__init__).parameters.values()]
        for cls in classes
    ]
    assert init_params[0] == init_params[1] == init_params[2]

    shared_methods = (
        "from_file_fast",
        "prepare",
        "run",
        "get_f",
        "get_jacobi",
        "write_array_result_to_network",
    )
    for method_name in shared_methods:
        param_lists = []
        for cls in classes:
            assert hasattr(cls, method_name), f"{cls.__name__} missing {method_name}"
            sig = inspect.signature(getattr(cls, method_name))
            param_lists.append([(p.name, p.kind) for p in sig.parameters.values()])
        assert param_lists[0] == param_lists[1] == param_lists[2], (
            f"Signature mismatch for {method_name}: {param_lists}"
        )


@pytest.mark.parametrize("kind", ["ac", "dc", "hybrid"])
def test_lf_result_modes_attributes_and_adaptive_lu(tmp_path: Path, kind: str):
    """Verify all 4 result_modes, consistent attributes, and convergence >=128 vars (LU reuse active)."""
    if kind == "ac":
        ef = _gen_ac_e(200, tmp_path, load_scale=1.5)
        cls = ACPowerFlowCalc
    elif kind == "dc":
        ef = _gen_dc_e(200, tmp_path)
        cls = DCPowerFlowCalc
    else:
        ef = _gen_hybrid_e(150, 80, tmp_path)
        cls = HybridPowerFlowCalc

    required_attrs = (
        "converged",
        "iterations",
        "normF",
        "x",
        "result",
        "lf_result",
        "failure_reason",
        "result_mode",
        "linear_solver",
        "tol",
        "max_iter",
        "min_voltage",
    )
    ref_x = None
    for mode in ("none", "summary", "array", "full"):
        calc = cls.from_file_fast(ef, parameters=LF_PARAMS, result_mode=mode, verbose=False)
        calc.prepare()
        rc = calc.run(result_mode=mode)
        assert rc == 0 and calc.converged
        assert calc.normF < LF_PARAMS.tol
        assert calc.failure_reason == ""
        for attr in required_attrs:
            assert hasattr(calc, attr), f"{cls.__name__} missing attribute {attr}"
        if ref_x is None:
            ref_x = calc.x.copy()
        else:
            assert np.allclose(calc.x, ref_x, atol=1e-9, rtol=1e-9)

        if mode == "none":
            assert calc.lf_result is None
        elif mode == "summary":
            assert calc.lf_result is None and isinstance(calc.result, dict) and len(calc.result) > 0
        elif mode == "array":
            assert calc.lf_result is None and isinstance(calc.result, dict) and len(calc.result) > 0
            calc.write_array_result_to_network()
        else:
            assert calc.lf_result is not None
            if kind == "hybrid":
                assert len(calc.lf_result.ac.nodes) > 0 and len(calc.lf_result.dc.nodes) > 0
                node_sample = next(iter(calc.lf_result.ac.nodes.values()))
                assert hasattr(node_sample, "volt") and hasattr(node_sample, "angle")
            elif kind == "ac":
                node_sample = next(iter(calc.lf_result.nodes.values()))
                br_sample = next(iter(calc.lf_result.branches.values()))
                assert hasattr(node_sample, "volt") and hasattr(node_sample, "angle")
                assert hasattr(br_sample, "i_p") and hasattr(br_sample, "j_q")
            else:
                node_sample = next(iter(calc.lf_result.nodes.values()))
                br_sample = next(iter(calc.lf_result.branches.values()))
                assert hasattr(node_sample, "volt")
                assert hasattr(br_sample, "i_p") and hasattr(br_sample, "j_c")


def test_adaptive_newton_factor_rollback_safeguard():
    """AdaptiveNewtonFactor must roll back x when a reused step increases residual norm."""
    class DummyFactor:
        def solve(self, rhs):
            return rhs * 0.5

    mgr = AdaptiveNewtonFactor(reuse_threshold=0.1, contract_ratio=0.5, max_reuses=2, min_size=2)
    mgr.update_factor(DummyFactor(), 1.0)
    assert mgr.can_reuse(0.01, 4)
    x_orig = np.array([1.0, 2.0, 3.0, 4.0])
    delta = mgr.solve_reused(np.ones(4), x_orig, 0.01)
    assert np.allclose(delta, 0.5)
    # Simulate a bad reused step that increases normF from 0.01 to 0.05
    restored = mgr.check_rollback(0.05)
    assert restored is not None and np.array_equal(restored, x_orig)
    assert not mgr.can_reuse(0.001, 4)


def test_efile_flat_rows_mutation_and_fallback(tmp_path: Path):
    """Bulk E-file parser must support lazy row mutation and fall back cleanly on quotes/ragged rows."""
    ef_std = _gen_dc_e(6, tmp_path)
    tables = _read_efile_rows(ef_std)
    rows = tables["DCBranch"]["rows"]
    assert len(rows) > 0
    # Mutating a materialized row (as hybrid_ca does) must persist and disable stale flat slices
    old_val = rows[0][-1]
    rows[0][-1] = "0"
    assert rows[0][-1] == "0"
    assert rows.column_tokens(len(tables["DCBranch"]["header_list"]) - 1) is None
    rows[0][-1] = old_val

    # Quoted / ragged E-file triggers fallback parser
    ef_q = tmp_path / "quoted.e"
    ef_q.write_text(
        "<Model>" + NL + "@ p_base u_unit p_unit i_unit" + NL + "# 100.0 kV MW kA" + NL + "</Model>" + NL
        + "<ACNode>" + NL + "@ idx name vbase voltage angle isl run_stat" + NL
        + "# 1 'bus one' 110.0 110.0 0.0 1 1" + NL + "</ACNode>" + NL,
        encoding="utf8",
    )
    q_tables = _read_efile_rows(ef_q)
    assert q_tables["ACNode"]["rows"][0][1] == "'bus one'"


@pytest.mark.parametrize("kind", ["ac", "dc", "hybrid"])
def test_lf_init_attrs_failure_reason_summary_and_lazy_csr(tmp_path: Path, kind: str):
    """验证三套求解器初始化属性、未收敛/发散原因、summary 键与懒构建 CSR 雅可比一致性。"""
    if kind == "ac":
        ef = _gen_ac_e(30, tmp_path)
        cls = ACPowerFlowCalc
    elif kind == "dc":
        ef = _gen_dc_e(30, tmp_path)
        cls = DCPowerFlowCalc
    else:
        ef = _gen_hybrid_e(20, 12, tmp_path)
        cls = HybridPowerFlowCalc

    # 1. 初始化后即具备统一核心公共属性
    calc = cls.from_file_fast(ef, parameters=LF_PARAMS, result_mode="array", verbose=False)
    for attr in (
        "converged",
        "iterations",
        "normF",
        "failure_reason",
        "x",
        "result",
        "lf_result",
        "total_vars",
        "total_eq",
        "target_island",
        "tol",
        "max_iter",
        "min_voltage",
        "linear_solver",
        "result_mode",
        "verbose",
        "ppc",
    ):
        assert hasattr(calc, attr), f"{cls.__name__} 初始化后缺少属性 {attr}"
    assert isinstance(calc.ppc, dict)
    assert calc.lf_result is None
    assert calc.total_vars == 0 and calc.total_eq == 0

    # 2. array 模式下调用 get_jacobi(x) 懒构建 CSR 稀疏矩阵，且与数值中心差分一致
    calc.prepare()
    x0 = calc.x.copy()
    f0 = calc.get_f(x0).copy()
    j_csr = calc.get_jacobi(x0)
    assert j_csr.format == "csr"
    assert j_csr.shape == (calc.total_eq, calc.total_vars)
    # 抽样校验若干列的解析雅可比与中心差分一致性
    eps = 1e-7
    j_dense = j_csr.toarray()
    sample_cols = np.linspace(0, calc.total_vars - 1, min(8, calc.total_vars), dtype=int)
    for col in sample_cols:
        xp = x0.copy()
        xm = x0.copy()
        xp[col] += eps
        xm[col] -= eps
        fp = calc.get_f(xp).copy()
        fm = calc.get_f(xm).copy()
        fd_col = (fp - fm) / (2.0 * eps)
        assert np.allclose(j_dense[:, col], fd_col, atol=1e-5, rtol=1e-5)

    # 3. summary 模式下均提供顶层 "summary" 键
    rc = calc.run(result_mode="summary")
    assert rc == 0 and calc.converged
    assert "summary" in calc.result
    assert set(calc.result["summary"].keys()) >= {"converged", "iterations", "normF"}

    # 4. 达到 max_iter 未收敛时均填写非空 failure_reason
    p_nonconv = algorithm_parameters.PowerFlowParameters(
        tol=1e-12, max_iter=1, min_voltage=0.1, divergence_threshold=1e6
    )
    calc_nc = cls.from_file_fast(ef, parameters=p_nonconv, result_mode="array", verbose=False)
    calc_nc.prepare()
    assert calc_nc.run() == -1
    assert not calc_nc.converged
    assert "未在最大迭代次数" in calc_nc.failure_reason

    # 5. 残差超过 divergence_threshold 时提前终止并填写发散原因
    p_div = algorithm_parameters.PowerFlowParameters(
        tol=1e-12, max_iter=20, min_voltage=0.1, divergence_threshold=1e-6
    )
    calc_div = cls.from_file_fast(ef, parameters=p_div, result_mode="array", verbose=False)
    calc_div.prepare()
    assert calc_div.run() == -1
    assert not calc_div.converged
    assert calc_div.iterations == 1
    assert "发散" in calc_div.failure_reason


def test_build_group_share_metadata_and_zero_branch_components_fast():
    """验证向量化分组共享元数据与含环/无环零阻抗连通分量算法的精确等价性。"""
    from lfcore.common import build_group_share_metadata, build_zero_branch_components_fast

    positions = np.array([5, 2, 5, 8, 2, 5, 9], dtype=np.int32)
    setpoints = np.array([1.02, 1.00, 1.06, 0.99, 1.04, 1.01, 1.05], dtype=np.float64)
    active_mask = np.array([True, True, True, False, True, True, True], dtype=bool)
    avg_set, rep_mask, share_mask, ref_idx = build_group_share_metadata(positions, setpoints, active_mask)

    assert np.isclose(avg_set[0], (1.02 + 1.06 + 1.01) / 3.0)
    assert np.isclose(avg_set[2], avg_set[0]) and np.isclose(avg_set[5], avg_set[0])
    assert np.isclose(avg_set[1], 1.02) and np.isclose(avg_set[4], 1.02)
    assert np.isclose(avg_set[3], 0.99)
    assert np.isclose(avg_set[6], 1.05)
    assert rep_mask.tolist() == [True, True, False, False, False, False, True]
    assert share_mask.tolist() == [False, False, True, False, True, True, False]
    assert ref_idx.tolist() == [0, 1, 0, 3, 1, 0, 6]

    # 含环连通分量（节点 0-1-2 构成三角形 + 节点 4-5 单边树分量）
    edge_a = np.array([0, 1, 2, 4], dtype=np.int32)
    edge_b = np.array([1, 2, 0, 5], dtype=np.int32)
    res = build_zero_branch_components_fast(edge_a, edge_b, build_lists=True)
    assert res["n_comp"] == 2
    assert res["phi_node"].size == 5
    # 三角形选 2 条生成树边 + 单边选 1 条 = 共 3 条树边
    assert res["tree_edge_indices"].size == 3
    assert len(res["comp_nodes"]) == 2
    assert sum(len(edges) for edges in res["comp_tree_edges"]) == 3


def test_hybrid_deferred_subsolver_csc_and_acac_zero_branch_convergence(tmp_path: Path):
    """混合耦合模式跳过子求解器重复 CSC 构建时，求解结果、子求解器懒构建 get_jacobi 与含零阻抗/ACAC 网络均应正确。"""
    ef = _gen_hybrid_e(16, 12, tmp_path)
    # 追加 ACACConverter、ACZeroBranch 与 DCZeroBranch 以覆盖全部向量化预处理分支
    extra = NL.join(
        [
            "<ACACConverter>",
            "@ idx name i_node j_node r1 r2 i_control_type j_control_type p_set i_q_set j_q_set i_v_set j_v_set run_stat",
            "# 1 acac_1 2 6 0.004 0.004 PQ PQ 0.1 0.02 -0.02 110.0 110.0 1",
            "</ACACConverter>",
            "<ACZeroBranch>",
            "@ idx name i_node j_node run_stat",
            "# 1 acz_1 7 8 1",
            "</ACZeroBranch>",
            "<DCZeroBranch>",
            "@ idx name i_node j_node run_stat",
            "# 1 dcz_1 5 6 1",
            "</DCZeroBranch>",
        ]
    )
    ef_ext = tmp_path / "hyb_ext.e"
    ef_ext.write_text(ef.read_text(encoding="utf8") + NL + extra + NL, encoding="utf8")

    calc = HybridPowerFlowCalc.from_file_fast(ef_ext, parameters=LF_PARAMS, result_mode="array", verbose=False)
    calc.prepare()
    # 验证混合耦合模式下子求解器确实跳过了冗余子块 CSC 构建
    assert calc.ac_calc.full_jac_raw_to_csc_pos.size == 0
    assert calc.dc_calc._dc_jac_raw_to_csc_pos.size == 0
    assert calc.run() == 0 and calc.converged
    assert calc.normF < LF_PARAMS.tol

    # 外部显式调用子求解器 get_jacobi() 仍能按需懒构建并返回正确 CSR 矩阵
    ac_x, dc_x, _ = calc._split_x(calc.x)
    j_ac = calc.ac_calc.get_jacobi(ac_x)
    j_dc = calc.dc_calc.get_jacobi(dc_x)
    assert j_ac.format == "csr" and j_ac.shape == (calc.ac_eq, calc.ac_size)
    assert j_dc.format == "csr" and j_dc.shape == (calc.dc_eq, calc.dc_size)

