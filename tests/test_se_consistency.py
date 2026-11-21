"""Interface consistency and accuracy tests for AC, DC, and Hybrid state estimators."""

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
from lfcore.ac_lf import ACPowerFlowCalc
from lfcore.dc_lf import DCPowerFlowCalc
from lfcore.hybrid_lf import HybridPowerFlowCalc
from model.ac_array_model import build_ac_ppc_from_e_file
from model.dc_array_model import build_dc_ppc_from_e_file
from secore.ac_se import ACStateEstimator
from secore.dc_se import DCStateEstimator
from secore.hybrid_se import HybridStateEstimator
from secore.se_math import (
    build_normal_equations,
    inverse_gain_for_bad_data,
    measurement_leverage,
    measurement_leverage_for_bad_data,
)
from secore.se_result import SEResult


LF_PARAMS = algorithm_parameters.PowerFlowParameters(
    tol=1e-8,
    max_iter=50,
    min_voltage=0.1,
    divergence_threshold=1e6,
)
SE_PARAMS = algorithm_parameters.StateEstimationParameters(
    tol=1e-6,
    max_iter=30,
    diff_step=1e-6,
    flat_start=True,
    bad_threshold=3.0,
    max_remove=10,
    pseudo_measurement_weight=1e-3,
    targeted_pseudo_measurement_max=200,
    targeted_pseudo_measurement_redundancy_ratio=0.0,
    targeted_pseudo_measurement_step=10,
    voltage_floor=0.1,
    min_current_voltage=1e-3,
    power_flow_tol=1e-8,
    power_flow_max_iter=50,
    power_flow_min_voltage=0.1,
)


def _generate_ac_case(n_bus: int, out_dir: Path, *, seed: int = 101, inject_bad: bool = False):
    rng = np.random.default_rng(seed)
    e_path = out_dir / f"ac_{n_bus}.e"
    m_path = out_dir / f"ac_{n_bus}.meas"
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
    lines += ["</ACGenerator>", "<ACLoad>", "@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat"]
    l_idx = 0
    for node in range(2, n_bus + 1):
        if node % 3 != 0:
            l_idx += 1
            lines.append(f"# {l_idx} load_{l_idx} {node} 0.3 1.0 0.0 0.0 0.08 1.0 0.0 0.0 1")
    lines.append("</ACLoad>")
    e_path.write_text("\n".join(lines) + "\n", encoding="utf8")

    ppc = build_ac_ppc_from_e_file(e_path)
    calc = ACPowerFlowCalc(ppc, parameters=LF_PARAMS, keep_node_objects=False, verbose=False)
    calc.prepare()
    assert calc.run(result_mode="array") == 0
    bus_res, br_res = calc.ppc["bus"], calc.ppc["branch"]
    p_base = float(calc.ppc["base"]["p_base"])

    mlines = ["<Measurement>", "@ idx name dev_type dev_name meas_type weight valid value"]
    m_idx = 1
    for i in range(n_bus):
        err = 35.0 if (inject_bad and i == n_bus // 2) else 0.0
        mlines.append(f"# {m_idx} m_{m_idx} ACNode bus_{i + 1} V 100.0 1 {bus_res[i, 2] * bus_res[i, 1] + err:.6f}")
        m_idx += 1
    mlines.append(f"# {m_idx} m_{m_idx} ACNode bus_1 ANGLE 1000.0 1 0.0")
    m_idx += 1
    for i in range(br_res.shape[0]):
        for mtype, col in (("P_FROM", 7), ("Q_FROM", 8), ("P_TO", 10), ("Q_TO", 11)):
            mlines.append(f"# {m_idx} m_{m_idx} ACBranch br_{i + 1} {mtype} 25.0 1 {br_res[i, col] * p_base:.6f}")
            m_idx += 1
    mlines.append("</Measurement>")
    m_path.write_text("\n".join(mlines) + "\n", encoding="utf8")
    return e_path, m_path


def _generate_dc_case(n_bus: int, out_dir: Path, *, seed: int = 102, inject_bad: bool = False):
    rng = np.random.default_rng(seed)
    e_path = out_dir / f"dc_{n_bus}.e"
    m_path = out_dir / f"dc_{n_bus}.meas"
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
        if node % 3 != 0:
            l_idx += 1
            lines.append(f"# {l_idx} dcload_{l_idx} {node} 0.2 1.0 0.0 0.0 1")
    lines.append("</DCLoad>")
    e_path.write_text("\n".join(lines) + "\n", encoding="utf8")

    ppc = build_dc_ppc_from_e_file(e_path)
    calc = DCPowerFlowCalc(ppc, parameters=LF_PARAMS, keep_node_objects=False, verbose=False)
    calc.prepare()
    assert calc.run(result_mode="array") == 0
    bus_res, br_res = calc.ppc["bus"], calc.ppc["branch"]
    p_base = float(calc.ppc["base"]["p_base"])

    mlines = ["<Measurement>", "@ idx name dev_type dev_name meas_type weight valid value"]
    m_idx = 1
    for i in range(n_bus):
        err = 35.0 if (inject_bad and i == n_bus // 2) else 0.0
        mlines.append(f"# {m_idx} m_{m_idx} DCNode dcbus_{i + 1} V 100.0 1 {bus_res[i, 2] * bus_res[i, 1] + err:.6f}")
        m_idx += 1
    for i in range(br_res.shape[0]):
        for mtype, col in (("P_FROM", 5), ("P_TO", 6)):
            mlines.append(f"# {m_idx} m_{m_idx} DCBranch dcbr_{i + 1} {mtype} 25.0 1 {br_res[i, col] * p_base:.6f}")
            m_idx += 1
    mlines.append("</Measurement>")
    m_path.write_text("\n".join(mlines) + "\n", encoding="utf8")
    return e_path, m_path


def _generate_hybrid_case(n_ac: int, n_dc: int, out_dir: Path, *, inject_bad: bool = False):
    ac_e, _ = _generate_ac_case(n_ac, out_dir, seed=201)
    dc_e, _ = _generate_dc_case(n_dc, out_dir, seed=202)
    ac_text = ac_e.read_text(encoding="utf8")
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
        f"# 1 dcac_1 {min(n_ac, 3)} {min(n_dc, 3)} 0.005 0.005 PQ NONE 0.5 0.0 0.1 0.0 0.0 1",
        "</DCACConverter>",
    ]
    hyb_e = out_dir / f"hyb_{n_ac}_{n_dc}.e"
    hyb_e.write_text(ac_text + "\n".join(out_dc) + "\n" + "\n".join(conv_lines) + "\n", encoding="utf8")

    hcalc = HybridPowerFlowCalc.from_file_fast(hyb_e, parameters=LF_PARAMS, verbose=False)
    hcalc.prepare()
    assert hcalc.run(result_mode="array") == 0
    ac_ppc, dc_ppc = hcalc.ac_calc.ppc, hcalc.dc_calc.ppc
    p_base = float(ac_ppc["base"]["p_base"])

    mlines = ["<Measurement>", "@ idx name dev_type dev_name meas_type weight valid value"]
    m_idx = 1
    ab, abr = ac_ppc["bus"], ac_ppc["branch"]
    for i in range(n_ac):
        err = 35.0 if (inject_bad and i == n_ac // 2) else 0.0
        mlines.append(f"# {m_idx} m_{m_idx} ACNode bus_{i + 1} V 100.0 1 {ab[i, 2] * ab[i, 1] + err:.6f}")
        m_idx += 1
    mlines.append(f"# {m_idx} m_{m_idx} ACNode bus_1 ANGLE 1000.0 1 0.0")
    m_idx += 1
    for i in range(abr.shape[0]):
        for mtype, col in (("P_FROM", 7), ("Q_FROM", 8), ("P_TO", 10), ("Q_TO", 11)):
            mlines.append(f"# {m_idx} m_{m_idx} ACBranch br_{i + 1} {mtype} 25.0 1 {abr[i, col] * p_base:.6f}")
            m_idx += 1
    db, dbr = dc_ppc["bus"], dc_ppc["branch"]
    for i in range(n_dc):
        mlines.append(f"# {m_idx} m_{m_idx} DCNode dcbus_{i + 1} V 100.0 1 {db[i, 2] * db[i, 1]:.6f}")
        m_idx += 1
    for i in range(dbr.shape[0]):
        for mtype, col in (("P_FROM", 5), ("P_TO", 6)):
            mlines.append(f"# {m_idx} m_{m_idx} DCBranch dcbr_{i + 1} {mtype} 25.0 1 {dbr[i, col] * p_base:.6f}")
            m_idx += 1
    mlines.append(f"# {m_idx} m_{m_idx} DCACConverter dcac_1 P_AC 25.0 1 0.5")
    m_idx += 1
    mlines.append(f"# {m_idx} m_{m_idx} DCACConverter dcac_1 Q_AC 25.0 1 0.1")
    mlines.append("</Measurement>")
    hyb_m = out_dir / f"hyb_{n_ac}_{n_dc}.meas"
    hyb_m.write_text("\n".join(mlines) + "\n", encoding="utf8")
    return hyb_e, hyb_m


def test_se_init_and_public_api_consistency():
    """ACStateEstimator, DCStateEstimator, and HybridStateEstimator must share identical core signatures."""
    classes = (ACStateEstimator, DCStateEstimator, HybridStateEstimator)
    init_params = [
        [(p.name, p.kind) for p in inspect.signature(cls.__init__).parameters.values()]
        for cls in classes
    ]
    assert init_params[0] == init_params[1] == init_params[2]

    shared_methods = (
        "prepare",
        "finalize_prepare",
        "run",
        "estimate",
        "observability_analysis",
        "identify_bad_data",
        "estimate_with_bad_data_removal",
        "build_se_result",
        "write_back",
        "apply_state",
        "initial_state",
        "evaluate",
        "jacobian",
        "jacobian_sparse",
        "measurement_device_counts",
        "measurement_device_names",
        "materialize_measurements",
        "voltage_measurement_node_idx",
        "state_layout",
        "print_state",
    )
    for method_name in shared_methods:
        param_lists = []
        default_lists = []
        for cls in classes:
            assert hasattr(cls, method_name), f"{cls.__name__} missing {method_name}"
            sig = inspect.signature(getattr(cls, method_name))
            param_lists.append([(p.name, p.kind) for p in sig.parameters.values()])
            default_lists.append([(p.name, p.default) for p in sig.parameters.values()])
        assert param_lists[0] == param_lists[1] == param_lists[2], (
            f"Signature mismatch for {method_name}: {param_lists}"
        )
        assert default_lists[0] == default_lists[1] == default_lists[2], (
            f"Default value mismatch for {method_name}: {default_lists}"
        )

    for prop_name in ("state_labels", "state_meta"):
        for cls in classes:
            assert isinstance(getattr(cls, prop_name, None), property), (
                f"{cls.__name__} missing @property {prop_name}"
            )


@pytest.mark.parametrize("kind", ["ac", "dc", "hybrid"])
def test_se_modes_bad_data_and_apply_state_consistency(tmp_path: Path, kind: str):
    if kind == "ac":
        ef, mf = _generate_ac_case(16, tmp_path, inject_bad=True)
        cls = ACStateEstimator
    elif kind == "dc":
        ef, mf = _generate_dc_case(16, tmp_path, inject_bad=True)
        cls = DCStateEstimator
    else:
        ef, mf = _generate_hybrid_case(12, 8, tmp_path, inject_bad=True)
        cls = HybridStateEstimator

    for mode in ("none", "array", "summary", "full"):
        est = cls(
            e_file=ef,
            meas_file=mf,
            parameters=SE_PARAMS,
            flat_start=True,
            profile=True,
            auto_prepare=False,
        )
        assert est.diff_step == SE_PARAMS.diff_step
        assert list(est.state_labels) == []
        assert list(est.state_meta) == []
        est.prepare()
        assert len(est.state_labels) == est.n_state
        assert len(est.state_meta) == est.n_state
        se_res = est.run(result_mode=mode, skip_bad_data=False, verbose=False)
        assert est.estimate_result is not None and est.estimate_result.converged
        assert len(est.bad_items) >= 1
        assert "bad_data.total" in est.profile_times
        assert est.normalized_residual.size == est.estimate_result.residual.size
        if mode == "full":
            J_dense = est.jacobian(est.estimate_result.x)
            J_sparse = est.jacobian_sparse(est.estimate_result.x).toarray()
            assert np.allclose(J_dense, J_sparse, rtol=1e-12, atol=1e-12)
            if kind == "hybrid":
                # 验证兼容设备视图按需懒加载可用
                assert not est._compatibility_device_views_ready
                assert len(est.ac_nodes) > 0 and len(est.dc_nodes) > 0
                assert "dcac_1" in est.dcac_by_name
                assert est._compatibility_device_views_ready
        counts = est.measurement_device_counts()
        assert isinstance(counts, dict) and len(counts) > 0
        names = est.measurement_device_names()
        assert isinstance(names, dict) and len(names) > 0
        est.apply_state(est.estimate_result.x)

        if mode in ("none", "array"):
            assert se_res is None
        else:
            assert isinstance(se_res, SEResult)
            assert se_res.statistics.converged
            assert se_res.statistics.bad_data_count == len(est.bad_items)
            if mode == "full":
                assert len(se_res.bad_data) == len(est.bad_items)
                assert len(se_res.normal_measurements) > 0


def test_sparse_selected_inversion_leverage_matches_dense_exact(tmp_path: Path):
    """Supernodal Takahashi selected inversion must match dense diag(H G^-1 H^T) to machine precision."""
    # n_bus = 350 -> n_state = 350 (> sparse_threshold=250, exercises sparse selected inversion)
    ef, mf = _generate_dc_case(350, tmp_path, inject_bad=True)
    est = DCStateEstimator(e_file=ef, meas_file=mf, parameters=SE_PARAMS, flat_start=True)
    res = est.estimate(verbose=False, final_diagnostics=True)
    tabs = est._active_measurement_plan_tables_ref()
    _z, w = est._measurement_vectors(tabs)
    H = est.jacobian_sparse(res.x, tabs)
    G, _ = build_normal_equations(H, res.residual, w)

    lev_sparse = measurement_leverage_for_bad_data(H, G, dense_state_limit=5000, sparse_threshold=100)
    G_inv_dense = inverse_gain_for_bad_data(G, dense_state_limit=5000)
    lev_dense = measurement_leverage(H, G_inv_dense)

    assert lev_sparse is not None
    assert np.allclose(lev_sparse, lev_dense, rtol=1e-10, atol=1e-11)
    assert measurement_leverage_for_bad_data(H, G, dense_state_limit=100) is None


@pytest.mark.parametrize("kind", ["ac", "dc", "hybrid"])
def test_se_bad_data_removal_end_to_end(tmp_path: Path, kind: str):
    """AC / DC / Hybrid SE 在平启动下均应正确完成端到端坏数据剔除。"""
    if kind == "ac":
        ef, mf = _generate_ac_case(16, tmp_path, inject_bad=True)
        cls = ACStateEstimator
    elif kind == "dc":
        ef, mf = _generate_dc_case(16, tmp_path, inject_bad=True)
        cls = DCStateEstimator
    else:
        ef, mf = _generate_hybrid_case(12, 8, tmp_path, inject_bad=True)
        cls = HybridStateEstimator

    est = cls(e_file=ef, meas_file=mf, parameters=SE_PARAMS, flat_start=True)
    se_res = est.run(result_mode="full", remove_bad_data=True, verbose=False)
    assert est.estimate_result is not None and est.estimate_result.converged
    assert len(est.removed_bad_data) >= 1
    assert len(est.bad_items) == 0
    assert isinstance(se_res, SEResult)
    assert se_res.statistics.bad_data_count == 0


def test_hybrid_selected_inversion_with_cholmod_backfill_matches_dense(tmp_path: Path):
    """交直流混合网络稀疏选择逆（含 CHOLMOD 因子缓存与缺失对回代）应与稠密求逆机器精度一致。"""
    from secore.se_math import CHOLMOD_CHOLESKY

    ef, mf = _generate_hybrid_case(120, 80, tmp_path, inject_bad=True)
    est = HybridStateEstimator(e_file=ef, meas_file=mf, parameters=SE_PARAMS, flat_start=True)
    res = est.estimate(verbose=False, final_diagnostics=True)
    assert res.converged and res.H is not None
    H = res.H
    _z, w = est._measurement_vectors(est.active_measurements)
    G, _ = build_normal_equations(H, res.residual, w, dense_gain_limit=0)
    factor = CHOLMOD_CHOLESKY(G.tocsc())

    lev_sparse = measurement_leverage_for_bad_data(
        H,
        None,
        dense_state_limit=5000,
        sparse_threshold=10,
        cholmod_factor=factor,
    )
    G_inv_dense = inverse_gain_for_bad_data(G, dense_state_limit=5000)
    lev_dense = measurement_leverage(H, G_inv_dense)

    assert lev_sparse is not None
    assert np.allclose(lev_sparse, lev_dense, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("kind", ["ac", "dc", "hybrid"])
def test_se_materialize_measurements_and_voltage_node_idx(tmp_path: Path, kind: str):
    """验证三大 SE 统一补齐的 materialize_measurements 与 voltage_measurement_node_idx 接口行为。"""
    from model.meas_model import MeasurementList
    from model.meas_type import DEVICE_TYPE_ACNode, DEVICE_TYPE_DCNode, MEAS_TYPE_V

    if kind == "ac":
        ef, mf = _generate_ac_case(10, tmp_path)
        cls = ACStateEstimator
    elif kind == "dc":
        ef, mf = _generate_dc_case(10, tmp_path)
        cls = DCStateEstimator
    else:
        ef, mf = _generate_hybrid_case(8, 6, tmp_path)
        cls = HybridStateEstimator

    est = cls(e_file=ef, meas_file=mf, parameters=SE_PARAMS, flat_start=True)
    mat = est.materialize_measurements()
    assert isinstance(mat, MeasurementList)
    assert len(mat) == len(est.active_measurements) > 0
    assert all(m.device_type and m.meas_type and m.device_name for m in mat[:10])

    if kind in ("ac", "hybrid"):
        ac_node = est.voltage_measurement_node_idx(DEVICE_TYPE_ACNode, 0, MEAS_TYPE_V)
        assert ac_node is not None and ac_node >= 1
    if kind in ("dc", "hybrid"):
        dc_node = est.voltage_measurement_node_idx(DEVICE_TYPE_DCNode, 0, MEAS_TYPE_V)
        assert dc_node is not None and dc_node >= 1


def test_multi_island_reference_selection_ac_and_dc(tmp_path: Path):
    """多岛 AC / DC 网络向量化参考节点选择应覆盖全部电气岛且能稳定求解。"""
    n_islands = 6
    ac_lines = [
        "<Model>",
        "@ p_base u_unit p_unit i_unit",
        "# 100.0 kV MW kA",
        "</Model>",
        "<ACNode>",
        "@ idx name vbase voltage angle isl run_stat",
    ]
    ac_br = ["<ACBranch>", "@ idx name i_node j_node r x b run_stat"]
    ac_gen = ["<ACGenerator>", "@ idx name node control_type p_set q_set v_set alpha run_stat"]
    ac_load = ["<ACLoad>", "@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat"]
    ac_meas = ["<Measurement>", "@ idx name dev_type dev_name meas_type weight valid value"]
    m_id = 1
    for isl in range(n_islands):
        base = isl * 2 + 1
        ac_lines.append(f"# {base} bus_{base} 110.0 110.0 0.0 {isl + 1} 1")
        ac_lines.append(f"# {base + 1} bus_{base + 1} 110.0 110.0 0.0 {isl + 1} 1")
        ac_br.append(f"# {isl + 1} br_{isl + 1} {base} {base + 1} 0.01 0.05 0.0 1")
        ac_gen.append(f"# {isl + 1} gen_{isl + 1} {base} SLACK 0.0 0.0 110.0 1.0 1")
        ac_load.append(f"# {isl + 1} load_{isl + 1} {base + 1} 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1")
        ac_meas.append(f"# {m_id} m_{m_id} ACNode bus_{base} V 100.0 1 110.0")
        m_id += 1
        ac_meas.append(f"# {m_id} m_{m_id} ACNode bus_{base + 1} V 100.0 1 110.0")
        m_id += 1
        if isl % 3 == 1:
            ac_meas.append(f"# {m_id} m_{m_id} ACNode bus_{base + 1} ANGLE 1000.0 1 0.0")
            m_id += 1
        ac_meas.append(f"# {m_id} m_{m_id} ACBranch br_{isl + 1} P_FROM 25.0 1 0.0")
        m_id += 1
        ac_meas.append(f"# {m_id} m_{m_id} ACBranch br_{isl + 1} Q_FROM 25.0 1 0.0")
        m_id += 1
    ac_lines.append("</ACNode>")
    ac_br.append("</ACBranch>")
    ac_gen.append("</ACGenerator>")
    ac_load.append("</ACLoad>")
    ac_meas.append("</Measurement>")
    ac_e = tmp_path / "ac_multi_isl.e"
    ac_m = tmp_path / "ac_multi_isl.meas"
    ac_e.write_text("\n".join(ac_lines + ac_br + ac_gen + ac_load) + "\n", encoding="utf8")
    ac_m.write_text("\n".join(ac_meas) + "\n", encoding="utf8")

    ac_est = ACStateEstimator(e_file=ac_e, meas_file=ac_m, parameters=SE_PARAMS, flat_start=True)
    assert len(ac_est.references) == n_islands
    ac_res = ac_est.estimate(verbose=False)
    assert ac_res.converged

    dc_lines = [
        "<Model>",
        "@ p_base u_unit p_unit i_unit",
        "# 100.0 kV MW kA",
        "</Model>",
        "<DCNode>",
        "@ idx name vbase voltage isl run_stat",
    ]
    dc_br = ["<DCBranch>", "@ idx name i_node j_node r run_stat"]
    dc_gen = ["<DCGenerator>", "@ idx name node control_type p_set v_set run_stat"]
    dc_load = ["<DCLoad>", "@ idx name node pbase pv0 pv1 pv2 run_stat"]
    dc_meas = ["<Measurement>", "@ idx name dev_type dev_name meas_type weight valid value"]
    m_id = 1
    for isl in range(n_islands):
        base = isl * 2 + 1
        dc_lines.append(f"# {base} dcbus_{base} 100.0 100.0 {isl + 1} 1")
        dc_lines.append(f"# {base + 1} dcbus_{base + 1} 100.0 100.0 {isl + 1} 1")
        dc_br.append(f"# {isl + 1} dcbr_{isl + 1} {base} {base + 1} 0.02 1")
        dc_gen.append(f"# {isl + 1} dcgen_{isl + 1} {base} V 0.0 100.0 1")
        dc_load.append(f"# {isl + 1} dcload_{isl + 1} {base + 1} 0.0 1.0 0.0 0.0 1")
        dc_meas.append(f"# {m_id} m_{m_id} DCNode dcbus_{base} V 100.0 1 100.0")
        m_id += 1
        dc_meas.append(f"# {m_id} m_{m_id} DCNode dcbus_{base + 1} V 100.0 1 100.0")
        m_id += 1
        dc_meas.append(f"# {m_id} m_{m_id} DCBranch dcbr_{isl + 1} P_FROM 25.0 1 0.0")
        m_id += 1
    dc_lines.append("</DCNode>")
    dc_br.append("</DCBranch>")
    dc_gen.append("</DCGenerator>")
    dc_load.append("</DCLoad>")
    dc_meas.append("</Measurement>")
    dc_e = tmp_path / "dc_multi_isl.e"
    dc_m = tmp_path / "dc_multi_isl.meas"
    dc_e.write_text("\n".join(dc_lines + dc_br + dc_gen + dc_load) + "\n", encoding="utf8")
    dc_m.write_text("\n".join(dc_meas) + "\n", encoding="utf8")

    dc_est = DCStateEstimator(e_file=dc_e, meas_file=dc_m, parameters=SE_PARAMS, flat_start=True)
    dc_res = dc_est.estimate(verbose=False)
    assert dc_res.converged


def test_multi_energy_context_electric_only_short_circuit():
    """纯电 E 文件行应触发多能上下文短路，避免无意义的对象物化；含流体表时正常构建。"""
    from model.multi_energy_model import build_multi_energy_context_from_rows

    electric_only_rows = {
        "Model": {"header_list": ["p_base", "u_unit", "p_unit", "i_unit"], "rows": [["100.0", "kV", "MW", "kA"]]},
        "ACNode": {
            "header_list": ["idx", "name", "vbase", "voltage", "angle", "isl", "run_stat"],
            "rows": [["1", "b1", "110.0", "110.0", "0.0", "1", "1"]],
        },
        "DCNode": {
            "header_list": ["idx", "name", "vbase", "voltage", "isl", "run_stat"],
            "rows": [["1", "d1", "100.0", "100.0", "1", "1"]],
        },
    }
    ctx_elec = build_multi_energy_context_from_rows(electric_only_rows, source="electric_only")
    assert not ctx_elec.fluid_networks
    assert not ctx_elec.couplings
    assert ctx_elec.total_fluid_nodes == 0
    assert ctx_elec.source == Path("electric_only")

    fluid_rows = {
        "Model": {"header_list": ["p_base", "u_unit", "p_unit", "i_unit"], "rows": [["100.0", "kV", "MW", "kA"]]},
        "HeatNode": {
            "header_list": ["idx", "name", "p", "t", "run_stat"],
            "rows": [["1", "h1", "0.5", "80.0", "1"]],
        },
    }
    ctx_fluid = build_multi_energy_context_from_rows(fluid_rows, source="with_heat")
    assert "heat" in ctx_fluid.fluid_networks
    assert ctx_fluid.total_fluid_nodes == 1


def test_sparse_jacobian_builder_fixed_pattern_add_many_fast_path():
    """验证 SparseJacobianBuilder 固定模式 add_many 恒等快路径及回退分支与动态构建完全等价。"""
    from secore.se_math import SparseJacobianBuilder, _normal_equation_structural_pattern

    rows_full = np.array([0, 1, 2, 3], dtype=np.int32)
    cols_full = np.array([0, 1, 2, 3], dtype=np.int32)
    rows_part = np.array([0, 1, 2, 3], dtype=np.int32)
    cols_part = np.array([1, -1, 3, 0], dtype=np.int32)
    mask_part = np.array([True, True, False, True], dtype=bool)
    mask_empty = np.array([False, False, False, False], dtype=bool)

    fixed = SparseJacobianBuilder((4, 4))
    fixed._assume_fixed_pattern = True

    for step in range(3):
        v1 = np.array([1.0 + step, 2.0 + step, 3.0 + step, 4.0 + step], dtype=np.float64)
        v2 = np.array([0.5 + step, 99.0, 88.0, -1.5 - step], dtype=np.float64)
        v3 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

        fixed.reset()
        fixed.add_many(rows_full, cols_full, v1)
        fixed.add_many(rows_part, cols_part, v2, mask_part)
        fixed.add_many(rows_full, cols_full, v3, mask_empty)
        H_fixed = fixed.to_csr()

        dyn = SparseJacobianBuilder((4, 4))
        dyn.add_many(rows_full, cols_full, v1)
        dyn.add_many(rows_part, cols_part, v2, mask_part)
        dyn.add_many(rows_full, cols_full, v3, mask_empty)
        H_dyn = dyn.to_csr()

        assert np.allclose(H_fixed.toarray(), H_dyn.toarray(), rtol=0.0, atol=0.0)
        pat_fixed = _normal_equation_structural_pattern(H_fixed, triangular="lower")
        pat_dyn = _normal_equation_structural_pattern(H_dyn.copy(), triangular="lower")
        assert np.array_equal(pat_fixed.indptr, pat_dyn.indptr)
        assert np.array_equal(pat_fixed.indices, pat_dyn.indices)

    # 动态替换新 mask 对象时应安全回退而不出错
    alt_mask = mask_part.copy()
    v1 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    v2 = np.array([1.25, 0.0, 0.0, -2.5], dtype=np.float64)
    fixed.reset()
    fixed.add_many(rows_full, cols_full, v1)
    fixed.add_many(rows_part, cols_part, v2, alt_mask)
    fixed.add_many(rows_full, cols_full, v1, mask_empty)
    H_fallback = fixed.to_csr()
    dyn = SparseJacobianBuilder((4, 4))
    dyn.add_many(rows_full, cols_full, v1)
    dyn.add_many(rows_part, cols_part, v2, alt_mask)
    dyn.add_many(rows_full, cols_full, v1, mask_empty)
    assert np.allclose(H_fallback.toarray(), dyn.to_csr().toarray(), rtol=0.0, atol=0.0)


