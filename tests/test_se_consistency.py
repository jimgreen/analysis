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
        "install_measurement_runtime",
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
        assert est.converged is True
        assert est.iterations == est.estimate_result.iterations and est.iterations >= 1
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


@pytest.mark.parametrize("kind", ["ac", "dc", "hybrid"])
def test_se_array_mode_bad_data_removal_with_blind_nodes_flat_start(tmp_path: Path, kind: str):
    """验证平启动 + 盲区缺测 + 大偏差坏数据在 array 模式下的可观测恢复、坏数据剔除及统一接口。"""
    if kind == "ac":
        ef, mf = _generate_ac_case(20, tmp_path, inject_bad=True)
        cls = ACStateEstimator
        blind_dev = "bus_19"
    elif kind == "dc":
        ef, mf = _generate_dc_case(20, tmp_path, inject_bad=True)
        cls = DCStateEstimator
        blind_dev = "dcbus_19"
    else:
        ef, mf = _generate_hybrid_case(12, 8, tmp_path, inject_bad=True)
        cls = HybridStateEstimator
        blind_dev = "acbus_11"

    # 将指定盲区节点的电压量测标记为不可用 (valid=0)，模拟局部量测缺失
    raw_lines = mf.read_text(encoding="utf8").splitlines()
    patched_lines = []
    for line in raw_lines:
        if line.startswith("# ") and f" {blind_dev} V " in line:
            parts = line.split()
            parts[6] = "0"
            line = " ".join(parts)
        patched_lines.append(line)
    mf.write_text("\n".join(patched_lines) + "\n", encoding="utf8")

    est = cls(
        e_file=ef,
        meas_file=mf,
        parameters=SE_PARAMS,
        flat_start=True,
        profile=True,
        auto_prepare=False,
    )
    est.prepare()
    # 验证三家统一提供 install_measurement_runtime 接口
    runtime = est.install_measurement_runtime(est.active_measurements)
    assert runtime is not None

    # 验证三家统一支持 result_mode='array' 与 remove_bad_data=True 组合
    se_res = est.run(result_mode="array", remove_bad_data=True, verbose=False)
    assert se_res is None
    assert est.se_result is None
    assert est.converged is True
    assert est.iterations >= 1
    assert est.observability_result is not None and est.observability_result.observable is True
    assert len(est.removed_bad_data) >= 1
    assert len(est.bad_items) == 0


def test_se_sparse_observability_woodbury_and_critical_protection():
    """验证稀疏二分图匹配可观测性诊断、临界量测保护及 Woodbury 秩 1 杠杆更新的数学精度。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        compute_normalized_residuals_for_bad_data,
        exact_leverage_for_candidate_rows,
        observability_rank_details,
        update_leverage_after_row_removal,
    )

    # 1. 构造 320 维稀疏雅可比（超过 dense_svd_limit=256）：
    #    - 第 100 列为零覆盖盲区；
    #    - 第 200 与 201 列仅共享一条量测行，导致结构缺秩 1 维；
    n = 320
    rows = []
    cols = []
    vals = []
    r_id = 0
    for c in range(n):
        if c in (100, 200, 201):
            continue
        rows.append(r_id)
        cols.append(c)
        vals.append(1.0)
        r_id += 1
    # 共享行覆盖 200 和 201 两列
    rows.extend([r_id, r_id])
    cols.extend([200, 201])
    vals.extend([1.0, -1.0])
    r_id += 1
    # 额外冗余行（覆盖其余列）确保 m >= n
    for c in range(0, 20):
        rows.append(r_id)
        cols.append(c)
        vals.append(0.5)
        r_id += 1
    H_def = csr_matrix((vals, (rows, cols)), shape=(r_id, n), dtype=np.float64)
    rank, deficiency, _sv, weak_states = observability_rank_details(H_def, n, dense_svd_limit=256)
    assert deficiency >= 2
    assert rank <= n - 2
    weak_cols = {col for col, _score in weak_states}
    assert 100 in weak_cols
    assert (200 in weak_cols) or (201 in weak_cols)

    # 2. 临界量测与伪量测归一化残差保护
    weights = np.array([10000.0, 10000.0, 1e-3, 10000.0], dtype=np.float64)
    R_diag = 1.0 / weights
    # 第 0 条为普通冗余坏数据；第 1 条为收敛噪声级临界量测；第 2 条为临界伪量测；第 3 条为普通正常量测
    leverage = np.array([0.4 * R_diag[0], R_diag[1], R_diag[2], 0.3 * R_diag[3]], dtype=np.float64)
    residual = np.array([0.15, 1e-7, 0.05, 1e-4], dtype=np.float64)
    is_pseudo = np.array([False, False, True, False], dtype=bool)
    norm, crit = compute_normalized_residuals_for_bad_data(residual, weights, leverage, is_pseudo=is_pseudo)
    assert crit.tolist() == [False, True, True, False]
    assert norm[0] > 10.0
    assert norm[1] == 0.0
    assert norm[2] == 0.0
    assert norm[3] < 1.0

    # 3. Woodbury 秩 1 杠杆传播 + 多右端精确回代与稠密求逆精确一致
    if CHOLMOD_CHOLESKY is not None:
        rng = np.random.default_rng(77)
        n_s = 60
        m_s = 140
        H_dense = np.zeros((m_s, n_s), dtype=np.float64)
        for i in range(n_s):
            H_dense[i, i] = 1.0
        for i in range(n_s, m_s):
            a, b = rng.choice(n_s, size=2, replace=False)
            H_dense[i, a] = float(rng.uniform(0.5, 2.0))
            H_dense[i, b] = float(rng.uniform(-2.0, -0.5))
        H_sp = csr_matrix(H_dense)
        w = rng.uniform(10.0, 1000.0, size=m_s)
        G_old, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
        fac_old = CHOLMOD_CHOLESKY(G_old.tocsc())
        lev_old = measurement_leverage(H_sp, inverse_gain_for_bad_data(G_old))

        remove_pos = 75
        lev_woodbury = update_leverage_after_row_removal(lev_old, H_sp, w, remove_pos, fac_old)
        keep = np.delete(np.arange(m_s), remove_pos)
        H_new = H_sp[keep]
        w_new = w[keep]
        G_new, _ = build_normal_equations(H_new, np.zeros(m_s - 1), w_new, dense_gain_limit=0)
        fac_new = CHOLMOD_CHOLESKY(G_new.tocsc())
        lev_new_exact = measurement_leverage(H_new, inverse_gain_for_bad_data(G_new))
        assert lev_woodbury is not None
        assert np.allclose(lev_woodbury, lev_new_exact, rtol=1e-10, atol=1e-11)

        cand = np.array([0, 15, 59, 90, 120], dtype=np.intp)
        lev_cand = exact_leverage_for_candidate_rows(H_new, cand, fac_new)
        assert lev_cand is not None
        assert np.allclose(lev_cand, lev_new_exact[cand], rtol=1e-10, atol=1e-11)

        # keep_shape=True 零权重掩码模式与物理删行模式逐项等价
        lev_woodbury_masked = update_leverage_after_row_removal(
            lev_old, H_sp, w, remove_pos, fac_old, keep_shape=True
        )
        assert lev_woodbury_masked is not None
        assert lev_woodbury_masked.size == m_s
        assert lev_woodbury_masked[remove_pos] == 0.0
        assert np.allclose(lev_woodbury_masked[keep], lev_new_exact, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("kind", ["ac", "dc", "hybrid"])
def test_se_masked_weight_bad_data_matches_full_and_array_modes(tmp_path: Path, kind: str):
    """验证零权重掩码与自适应冻结因子在 array 与 full 模式下状态估计与坏数据剔除结果完全一致。"""
    if kind == "ac":
        ef, mf = _generate_ac_case(24, tmp_path, inject_bad=True)
        cls = ACStateEstimator
    elif kind == "dc":
        ef, mf = _generate_dc_case(24, tmp_path, inject_bad=True)
        cls = DCStateEstimator
    else:
        ef, mf = _generate_hybrid_case(16, 12, tmp_path, inject_bad=True)
        cls = HybridStateEstimator

    est_array = cls(e_file=ef, meas_file=mf, parameters=SE_PARAMS, flat_start=True)
    res_array = est_array.run(result_mode="array", remove_bad_data=True, verbose=False)
    est_full = cls(e_file=ef, meas_file=mf, parameters=SE_PARAMS, flat_start=True)
    res_full = est_full.run(result_mode="full", remove_bad_data=True, verbose=False)

    assert res_array is None
    assert isinstance(res_full, SEResult)
    assert est_array.converged is True and est_full.converged is True
    assert est_array.observability_result.observable is True
    assert est_full.observability_result.observable is True
    assert [int(b.measurement.idx) for b in est_array.removed_bad_data] == [
        int(b.measurement.idx) for b in est_full.removed_bad_data
    ]
    assert len(est_array.removed_bad_data) >= 1
    assert len(est_array.bad_items) == 0 and len(est_full.bad_items) == 0
    assert np.allclose(est_array.estimate_result.x, est_full.estimate_result.x, rtol=1e-7, atol=1e-7)
    assert est_array.estimate_result.residual.size == est_full.estimate_result.residual.size
    assert est_array.normalized_residual.size == est_full.normalized_residual.size


def test_se_independent_batch_selection_and_block_woodbury():
    """验证拓扑隔离批量坏数据选择与 Block Woodbury 秩 k 杠杆批量更新的精确性。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        select_independent_bad_data_batch,
        update_leverage_after_batch_removal,
    )

    # 1. 链式图 20 个状态，相邻状态有支路量测相连
    n = 20
    rows = []
    cols = []
    vals = []
    r_id = 0
    # 0..19: 节点电压量测
    for c in range(n):
        rows.append(r_id)
        cols.append(c)
        vals.append(1.0)
        r_id += 1
    # 20..38: 相邻支路量测 (c, c+1)
    for c in range(n - 1):
        rows.extend([r_id, r_id])
        cols.extend([c, c + 1])
        vals.extend([1.0, -1.0])
        r_id += 1
    H_chain = csr_matrix((vals, (rows, cols)), shape=(r_id, n), dtype=np.float64)

    # 候选：节点 0 (rn=20), 节点 1 (1跳冲突, rn=18), 节点 2 (2跳冲突, rn=17),
    #       节点 3 (3跳隔离, rn=16), 节点 10 (远距隔离, rn=15), 节点 16 (低置信度 rn=3.5 < 5.0)
    cand_rows = [0, 1, 2, 3, 10, 16]
    cand_norms = [20.0, 18.0, 17.0, 16.0, 15.0, 3.5]
    chosen = select_independent_bad_data_batch(
        H_chain,
        cand_rows,
        cand_norms,
        threshold=3.0,
        min_batch_norm=5.0,
        max_batch=16,
    )
    assert [cand_rows[i] for i in chosen] == [0, 3, 10]

    # 2. Block Woodbury 秩 k 杠杆批量更新与全矩阵重算精确一致
    if CHOLMOD_CHOLESKY is not None:
        rng = np.random.default_rng(123)
        n_s = 50
        m_s = 130
        H_dense = np.zeros((m_s, n_s), dtype=np.float64)
        for i in range(n_s):
            H_dense[i, i] = 1.0
        for i in range(n_s, m_s):
            a, b = rng.choice(n_s, size=2, replace=False)
            H_dense[i, a] = float(rng.uniform(0.5, 2.0))
            H_dense[i, b] = float(rng.uniform(-2.0, -0.5))
        H_sp = csr_matrix(H_dense)
        w = rng.uniform(10.0, 1000.0, size=m_s)
        G_old, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
        fac_old = CHOLMOD_CHOLESKY(G_old.tocsc())
        lev_old = measurement_leverage(H_sp, inverse_gain_for_bad_data(G_old))

        batch_remove = [55, 72, 91, 108, 121]
        lev_batch = update_leverage_after_batch_removal(lev_old, H_sp, w, batch_remove, fac_old, keep_shape=False)
        lev_batch_masked = update_leverage_after_batch_removal(lev_old, H_sp, w, batch_remove, fac_old, keep_shape=True)
        keep = np.ones(m_s, dtype=bool)
        keep[batch_remove] = False
        H_new = H_sp[keep]
        w_new = w[keep]
        G_new, _ = build_normal_equations(H_new, np.zeros(int(keep.sum())), w_new, dense_gain_limit=0)
        lev_exact = measurement_leverage(H_new, inverse_gain_for_bad_data(G_new))

        assert lev_batch is not None and lev_batch_masked is not None
        assert np.allclose(lev_batch, lev_exact, rtol=1e-10, atol=1e-11)
        assert np.all(lev_batch_masked[batch_remove] == 0.0)
        assert np.allclose(lev_batch_masked[keep], lev_exact, rtol=1e-10, atol=1e-11)


def test_se_initial_fast_bad_data_screening_exact_equivalence():
    """验证首轮精准筛查、w==1/u==2 超节点标量直算及宽行杠杆求解与全量精确解等价。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        _measurement_leverage_from_selected_inv,
        _supernodal_selected_inverse_from_cholmod,
        compute_normalized_residuals_for_bad_data,
        screen_initial_bad_data_fast,
    )

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    # 构造 12x12 二维网格雅可比（产生大量 w==1, u==2 超节点）并附加 2 条宽行（nnz=20）
    nx, ny = 12, 12
    n_s = nx * ny
    rows, cols, vals = [], [], []
    r_id = 0
    for i in range(n_s):
        rows.append(r_id)
        cols.append(i)
        vals.append(1.0)
        r_id += 1
    for ix in range(nx):
        for iy in range(ny):
            u = ix * ny + iy
            if ix + 1 < nx:
                v = (ix + 1) * ny + iy
                rows.extend([r_id, r_id])
                cols.extend([u, v])
                vals.extend([1.0, -1.0])
                r_id += 1
            if iy + 1 < ny:
                v = ix * ny + (iy + 1)
                rows.extend([r_id, r_id])
                cols.extend([u, v])
                vals.extend([1.0, -1.0])
                r_id += 1
    # 附加 2 条宽行以覆盖 _measurement_leverage_from_selected_inv 的 big_rows CHOLMOD 直算路径
    for offset in (0, 30):
        for k in range(20):
            rows.append(r_id)
            cols.append(offset + k)
            vals.append(0.3 if k % 2 == 0 else -0.25)
        r_id += 1

    m_s = r_id
    H_sp = csr_matrix((vals, (rows, cols)), shape=(m_s, n_s), dtype=np.float64)
    rng = np.random.default_rng(2026)
    w = rng.uniform(50.0, 500.0, size=m_s)
    G, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
    fac = CHOLMOD_CHOLESKY(G.tocsc())

    # 1. 验证超节点选元求逆（含 w==1, u==2 标量直算）与宽行杠杆计算和稠密求逆精确一致
    Z_low, P = _supernodal_selected_inverse_from_cholmod(fac)
    assert Z_low is not None and P is not None
    G_inv_dense = np.linalg.inv(G.toarray())
    G_inv_perm = G_inv_dense[np.ix_(P, P)]
    z_coo = Z_low.tocoo()
    assert np.allclose(z_coo.data, G_inv_perm[z_coo.row, z_coo.col], rtol=1e-11, atol=1e-11)

    lev_sel = _measurement_leverage_from_selected_inv(H_sp, Z_low, P, cholmod_factor=fac)
    lev_dense = measurement_leverage(H_sp, G_inv_dense)
    assert lev_sel is not None
    assert np.allclose(lev_sel, lev_dense, rtol=1e-11, atol=1e-11)

    # 2. 验证 screen_initial_bad_data_fast：大偏差坏数据精确一致
    x_true = rng.normal(0.0, 0.1, size=n_s)
    z = H_sp @ x_true + rng.normal(0.0, 1e-4, size=m_s)
    bad_targets = [10, 65, 130, 210, 350]
    for b in bad_targets:
        z[b] += 2.5
    rhs = H_sp.T @ (w * z)
    x_est = np.asarray(fac(rhs), dtype=np.float64)
    res = z - (H_sp @ x_est)

    exact_norm_full, _ = compute_normalized_residuals_for_bad_data(res, w, lev_dense)
    status, hit_rows, hit_norms = screen_initial_bad_data_fast(
        res, H_sp, w, np.zeros(m_s, dtype=bool), fac, threshold=3.0
    )
    assert status == "candidates"
    assert hit_rows is not None and hit_norms is not None
    for b in bad_targets:
        assert b in set(hit_rows.tolist())
    assert np.allclose(hit_norms, exact_norm_full[hit_rows], rtol=1e-10, atol=1e-10)
    assert np.all(np.diff(hit_norms) <= 1e-12)

    # 3. 验证全干净极小残差触发 fast_clean 短路
    res_clean = np.full(m_s, 1e-10, dtype=np.float64)
    st_clean, rows_clean, _ = screen_initial_bad_data_fast(
        res_clean, H_sp, w, np.zeros(m_s, dtype=bool), fac, threshold=3.0
    )
    assert st_clean == "fast_clean"
    assert rows_clean is not None and rows_clean.size == 0

    # 4. 验证含轻微噪声（未触发下界但需全量认证）安全回退 need_full
    res_mild = rng.normal(0.0, 1e-4, size=m_s)
    st_mild, rows_mild, norms_mild = screen_initial_bad_data_fast(
        res_mild, H_sp, w, np.zeros(m_s, dtype=bool), fac, threshold=3.0
    )
    assert st_mild == "need_full"
    assert rows_mild is None and norms_mild is None

    # 5. 验证零误差严格残差上界稀疏认证（少量行触发 ub>3.0，精确回代后全未超阈 -> verified_clean）
    from secore.se_math import _strict_upper_bound_bad_data_cert

    res_sparse = np.full(m_s, 1e-10, dtype=np.float64)
    res_sparse[:20] = 5e-5
    st_cert, rows_cert, ub_cert = _strict_upper_bound_bad_data_cert(
        res_sparse, H_sp, w, np.zeros(m_s, dtype=bool), fac, threshold=3.0
    )
    assert st_cert == "verified_clean"
    assert rows_cert is not None and rows_cert.size == 0
    assert ub_cert is not None and float(ub_cert.max()) <= 3.0


def test_se_large_blind_weak_limit_and_midround_lower_bound_screen():
    """验证大盲区 (>256) 一次性稀疏弱状态检出，以及中间轮大偏移下界精准筛查与终轮回退。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        compute_normalized_residuals_for_bad_data,
        observability_rank_details,
        screen_bad_data_after_removal,
    )

    # 1. 600 状态中 320 个完全盲态，应一次性返回全部 320 个弱状态（不再被 256 截断）
    n_obs_test = 600
    obs_cols = np.arange(0, 280, dtype=np.int32)
    H_blind = csr_matrix(
        (np.ones(obs_cols.size, dtype=np.float64), (np.arange(obs_cols.size), obs_cols)),
        shape=(obs_cols.size, n_obs_test),
    )
    G_blind = (H_blind.T @ H_blind).tocsc()
    rank, deficiency, _, weak_states = observability_rank_details(
        H_blind, n_obs_test, normal_matrix=G_blind, dense_svd_limit=256
    )
    assert rank == 280 and deficiency == 320
    assert len(weak_states) == 320
    assert {idx for idx, _ in weak_states} == set(range(280, 600))

    # 2. 中间轮在 prev_leverage=None 或大 state_delta_inf 下通过下界精确筛出大坏数据，清理后安全回退 need_full
    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")
    nx, ny = 12, 12
    n_s = nx * ny
    rows, cols, vals = [], [], []
    r_id = 0
    for i in range(n_s):
        rows.append(r_id)
        cols.append(i)
        vals.append(1.0)
        r_id += 1
    for ix in range(nx):
        for iy in range(ny):
            u = ix * ny + iy
            if ix + 1 < nx:
                rows.extend([r_id, r_id])
                cols.extend([u, (ix + 1) * ny + iy])
                vals.extend([1.0, -1.0])
                r_id += 1
            if iy + 1 < ny:
                rows.extend([r_id, r_id])
                cols.extend([u, ix * ny + (iy + 1)])
                vals.extend([1.0, -1.0])
                r_id += 1
    m_s = r_id
    H_sp = csr_matrix((vals, (rows, cols)), shape=(m_s, n_s), dtype=np.float64)
    rng = np.random.default_rng(2027)
    w = rng.uniform(100.0, 500.0, size=m_s)
    G, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
    fac = CHOLMOD_CHOLESKY(G.tocsc())
    lev_exact = measurement_leverage(H_sp, np.linalg.inv(G.toarray()))

    x_true = rng.normal(0.0, 0.1, size=n_s)
    z = H_sp @ x_true + rng.normal(0.0, 1e-4, size=m_s)
    bad_ids = [15, 90, 220, 340]
    for b in bad_ids:
        z[b] += 3.0
    x_est = np.asarray(fac(H_sp.T @ (w * z)), dtype=np.float64)
    res_bad = z - (H_sp @ x_est)
    norm_exact, _ = compute_normalized_residuals_for_bad_data(res_bad, w, lev_exact)

    # prev_leverage=None 且 state_delta_inf=1.0 > max_state_delta 时仍应通过下界筛出全部大坏数据
    st, prop_lev, hit_rows, hit_norms = screen_bad_data_after_removal(
        None,
        H_sp,
        w,
        [0, 1],
        fac,
        res_bad,
        H_sp,
        w,
        np.zeros(m_s, dtype=bool),
        fac,
        threshold=3.0,
        state_delta_inf=1.0,
        active_mask=np.ones(m_s, dtype=bool),
        keep_shape=True,
    )
    assert st == "candidates" and prop_lev is None
    assert hit_rows is not None and hit_norms is not None
    for b in bad_ids:
        assert b in set(hit_rows.tolist())
    assert np.allclose(hit_norms, norm_exact[hit_rows], rtol=1e-10, atol=1e-10)

    # 清除全部大坏数据后，仅余轻微随机噪声（触发 >256 行 ub>3.0 但无下界命中）时必须严格回退 need_full
    res_clean_noisy = rng.normal(0.0, 1e-4, size=m_s)
    st2, _, hit_rows2, _ = screen_bad_data_after_removal(
        None,
        H_sp,
        w,
        bad_ids,
        fac,
        res_clean_noisy,
        H_sp,
        w,
        np.zeros(m_s, dtype=bool),
        fac,
        threshold=3.0,
        state_delta_inf=1.0,
        active_mask=np.ones(m_s, dtype=bool),
        keep_shape=True,
    )
    assert st2 == "need_full" and hit_rows2 is None


def test_se_no_z22_selected_inverse_and_peak_priority_screening():
    """验证免 Z_22 超节点选元求逆（含宽超节点）与峰值优先坏数据采样的精确性与完整性。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        _measurement_leverage_from_selected_inv,
        _supernodal_selected_inverse_from_cholmod,
        compute_normalized_residuals_for_bad_data,
        screen_initial_bad_data_fast,
    )

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    # 1. 构造 20x20 网格（含多列超节点）验证免 Z_22 选元求逆与稠密求逆精确一致
    nx, ny = 20, 20
    n_s = nx * ny
    rows, cols, vals = [], [], []
    r_id = 0
    for i in range(n_s):
        rows.append(r_id)
        cols.append(i)
        vals.append(1.0)
        r_id += 1
    for ix in range(nx):
        for iy in range(ny):
            u = ix * ny + iy
            if ix + 1 < nx:
                v = (ix + 1) * ny + iy
                # 每条支路配 4 条冗余量测以模拟 AC 支路连带陪跑效应
                for sign in (1.0, -1.0, 0.8, -0.8):
                    rows.extend([r_id, r_id])
                    cols.extend([u, v])
                    vals.extend([sign, -sign])
                    r_id += 1
            if iy + 1 < ny:
                v = ix * ny + (iy + 1)
                for sign in (1.0, -1.0):
                    rows.extend([r_id, r_id])
                    cols.extend([u, v])
                    vals.extend([sign, -sign])
                    r_id += 1
    m_s = r_id
    H_sp = csr_matrix((vals, (rows, cols)), shape=(m_s, n_s), dtype=np.float64)
    rng = np.random.default_rng(2028)
    w = rng.uniform(50.0, 300.0, size=m_s)
    G, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
    fac = CHOLMOD_CHOLESKY(G.tocsc())

    Z_low, P = _supernodal_selected_inverse_from_cholmod(fac)
    assert Z_low is not None and P is not None
    G_inv_dense = np.linalg.inv(G.toarray())
    G_inv_perm = G_inv_dense[np.ix_(P, P)]
    z_coo = Z_low.tocoo()
    assert np.allclose(z_coo.data, G_inv_perm[z_coo.row, z_coo.col], rtol=1e-11, atol=1e-11)

    lev_sel = _measurement_leverage_from_selected_inv(H_sp, Z_low, P, cholmod_factor=fac)
    lev_dense = measurement_leverage(H_sp, G_inv_dense)
    assert lev_sel is not None
    assert np.allclose(lev_sel, lev_dense, rtol=1e-11, atol=1e-11)

    # 2. 在 40 个分散节点注入强弱不一的大偏差坏数据，连带邻域使触发下界行数远超 max_candidates=90
    #    峰值优先采样应 100% 命中全部 40 个真坏数据，而非被最严重坏数据的陪跑支路挤掉
    bad_nodes = [ix * ny + iy for ix in range(1, nx - 1, 2) for iy in range(1, ny - 1, 4)][:40]
    x_true = rng.normal(0.0, 0.05, size=n_s)
    z = H_sp @ x_true + rng.normal(0.0, 1e-4, size=m_s)
    for k, b in enumerate(bad_nodes):
        # 故意拉大强弱梯度（2.0 ~ 20.0），若不按列去重则强坏数据的连带支路会挤掉弱坏数据
        z[b] += 20.0 if k < 15 else 2.5
    x_est = np.asarray(fac(H_sp.T @ (w * z)), dtype=np.float64)
    res = z - (H_sp @ x_est)
    norm_exact, _ = compute_normalized_residuals_for_bad_data(res, w, lev_dense)

    st, hit_rows, hit_norms = screen_initial_bad_data_fast(
        res, H_sp, w, np.zeros(m_s, dtype=bool), fac, threshold=3.0, max_candidates=90
    )
    assert st == "candidates" and hit_rows is not None and hit_norms is not None
    hit_set = set(hit_rows.tolist())
    for b in bad_nodes:
        assert b in hit_set
    assert np.allclose(hit_norms, norm_exact[hit_rows], rtol=1e-10, atol=1e-10)


def test_se_noisy_grid_woodbury_self_cert_no_spurious_fallback():
    """验证 2%-3% 强噪声大网格下 cand_floor 与 Block Woodbury 实测杠杆误差自检认证不误回退且零漏检。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        compute_normalized_residuals_for_bad_data,
        screen_bad_data_after_removal,
    )

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    # 24x24 网格 (n_state=576 > 512)，覆盖大网自动自检认证路径
    nx, ny = 24, 24
    n_s = nx * ny
    rows, cols, vals = [], [], []
    r_id = 0
    for i in range(n_s):
        rows.append(r_id)
        cols.append(i)
        vals.append(1.0)
        r_id += 1
    for ix in range(nx):
        for iy in range(ny):
            u = ix * ny + iy
            if ix + 1 < nx:
                v = (ix + 1) * ny + iy
                for sign in (1.0, -1.0):
                    rows.extend([r_id, r_id])
                    cols.extend([u, v])
                    vals.extend([sign, -sign])
                    r_id += 1
            if iy + 1 < ny:
                v = ix * ny + (iy + 1)
                for sign in (1.0, -1.0):
                    rows.extend([r_id, r_id])
                    cols.extend([u, v])
                    vals.extend([sign, -sign])
                    r_id += 1
    m_s = r_id
    H_sp = csr_matrix((vals, (rows, cols)), shape=(m_s, n_s), dtype=np.float64)
    rng = np.random.default_rng(2029)
    sigma = 0.025
    w = np.full(m_s, 1.0 / (sigma * sigma), dtype=np.float64)
    x_true = rng.normal(0.0, 0.05, size=n_s)
    # 叠加全网 2%~3% 级截断噪声（确保超过 200 条正常量测落在 2.0~3.2 sigma，模拟带噪电网）
    noise = rng.normal(0.0, sigma * 0.9, size=m_s)
    noise = np.clip(noise, -2.5 * sigma, 2.5 * sigma)
    z = H_sp @ x_true + noise

    # 注入 1 条大坏数据 + 1 条远离它的中等坏数据
    bad_strong = 100
    bad_mild = 450
    z[bad_strong] += 12.0 * sigma
    z[bad_mild] += 6.0 * sigma

    G0, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
    fac0 = CHOLMOD_CHOLESKY(G0.tocsc())
    lev0 = measurement_leverage(H_sp, np.linalg.inv(G0.toarray()))

    # 1. 剔除 bad_strong 后，Block Woodbury 传播 + 0.85*thr 门限应直接精确命中 bad_mild，不因 2sigma 噪声误回退 need_full
    w1 = w.copy()
    w1[bad_strong] = 0.0
    mask1 = np.ones(m_s, dtype=bool)
    mask1[bad_strong] = False
    G1, _ = build_normal_equations(H_sp, np.zeros(m_s), w1, dense_gain_limit=0)
    fac1 = CHOLMOD_CHOLESKY(G1.tocsc())
    x1 = np.asarray(fac1(H_sp.T @ (w1 * z)), dtype=np.float64)
    res1 = z - (H_sp @ x1)
    lev1_exact = measurement_leverage(H_sp, np.linalg.inv(G1.toarray()))
    norm1_exact, _ = compute_normalized_residuals_for_bad_data(res1, w1, lev1_exact)

    st1, prop_lev1, hit_rows1, hit_norms1 = screen_bad_data_after_removal(
        lev0,
        H_sp,
        w,
        bad_strong,
        fac0,
        res1,
        H_sp,
        w1,
        np.zeros(m_s, dtype=bool),
        fac1,
        threshold=4.0,
        state_delta_inf=0.02,
        active_mask=mask1,
        keep_shape=True,
    )
    assert st1 == "candidates" and prop_lev1 is not None
    assert hit_rows1 is not None and bad_mild in set(hit_rows1.tolist())
    assert np.allclose(hit_norms1, norm1_exact[hit_rows1], rtol=1e-10, atol=1e-10)

    # 2. 再剔除 bad_mild 后，实测杠杆误差自检通过且无剩余坏数据，应直接返回 verified_clean（与全量精确判定一致）
    w2 = w1.copy()
    w2[bad_mild] = 0.0
    mask2 = mask1.copy()
    mask2[bad_mild] = False
    G2, _ = build_normal_equations(H_sp, np.zeros(m_s), w2, dense_gain_limit=0)
    fac2 = CHOLMOD_CHOLESKY(G2.tocsc())
    x2 = np.asarray(fac2(H_sp.T @ (w2 * z)), dtype=np.float64)
    res2 = z - (H_sp @ x2)
    lev2_exact = measurement_leverage(H_sp, np.linalg.inv(G2.toarray()))
    norm2_exact, _ = compute_normalized_residuals_for_bad_data(res2, w2, lev2_exact)
    norm2_exact[~mask2] = 0.0
    assert float(norm2_exact.max()) <= 4.0

    st2, prop_lev2, hit_rows2, norm2_out = screen_bad_data_after_removal(
        prop_lev1,
        H_sp,
        w1,
        bad_mild,
        fac1,
        res2,
        H_sp,
        w2,
        np.zeros(m_s, dtype=bool),
        fac2,
        threshold=4.0,
        state_delta_inf=0.01,
        active_mask=mask2,
        keep_shape=True,
    )
    assert st2 == "verified_clean" and prop_lev2 is not None
    assert hit_rows2 is not None and hit_rows2.size == 0
    assert norm2_out is not None and float(norm2_out[mask2].max()) <= 4.0

    # 3. 验证正定逆向批量 Woodbury（使用删除后 G_new 因子）与全矩阵重算在 1e-11 内精确一致
    from secore.se_math import update_leverage_after_batch_removal

    lev2_rev = update_leverage_after_batch_removal(
        lev0, H_sp, w, [bad_strong, bad_mild], fac2, keep_shape=True, factor_is_post_removal=True
    )
    assert lev2_rev is not None
    assert np.allclose(lev2_rev[mask2], lev2_exact[mask2], rtol=1e-10, atol=1e-11)


def test_se_woodbury_first_step_linear_solve_exact_and_singular_fallback():
    """验证 Exact Block Woodbury 删行后直接线性求解（单行/批量）与重构 G_new 等价，且临界量测安全回退 None。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import CHOLMOD_CHOLESKY, solve_woodbury_after_row_removal

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    rng = np.random.default_rng(2030)
    n_s = 60
    m_s = 160
    H_dense = np.zeros((m_s, n_s), dtype=np.float64)
    # 0..59: 节点量测，其中节点 0 仅由行 0 唯一观测（行 0 为临界量测）
    for i in range(n_s):
        H_dense[i, i] = 1.0
    for i in range(n_s, m_s):
        a, b = rng.choice(np.arange(1, n_s), size=2, replace=False)
        H_dense[i, a] = float(rng.uniform(0.5, 2.0))
        H_dense[i, b] = float(rng.uniform(-2.0, -0.5))
    H_sp = csr_matrix(H_dense)
    w = rng.uniform(20.0, 500.0, size=m_s)
    G_old, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
    fac_old = CHOLMOD_CHOLESKY(G_old.tocsc())
    rhs = rng.normal(0.0, 1.0, size=n_s)

    # 1. 单行剔除精确等价
    rem1 = [85]
    dx_wb1 = solve_woodbury_after_row_removal(H_sp, w, rem1, rhs, fac_old)
    w1 = w.copy()
    w1[rem1] = 0.0
    G1, _ = build_normal_equations(H_sp, np.zeros(m_s), w1, dense_gain_limit=0)
    dx_ex1 = np.linalg.solve(G1.toarray(), rhs)
    assert dx_wb1 is not None
    assert np.allclose(dx_wb1, dx_ex1, rtol=1e-12, atol=1e-12)

    # 2. 批量 10 行剔除精确等价
    rem10 = [62, 70, 79, 88, 95, 104, 118, 129, 141, 155]
    dx_wb10 = solve_woodbury_after_row_removal(H_sp, w, rem10, rhs, fac_old)
    w10 = w.copy()
    w10[rem10] = 0.0
    G10, _ = build_normal_equations(H_sp, np.zeros(m_s), w10, dense_gain_limit=0)
    dx_ex10 = np.linalg.solve(G10.toarray(), rhs)
    assert dx_wb10 is not None
    assert np.allclose(dx_wb10, dx_ex10, rtol=1e-11, atol=1e-11)

    # 3. 临界量测行 0（删除后系统失观）应安全返回 None 触发标准分解与失观保护
    assert solve_woodbury_after_row_removal(H_sp, w, [0], rhs, fac_old) is None
    assert solve_woodbury_after_row_removal(H_sp, w, [0, 85], rhs, fac_old) is None


def test_se_unanchored_voltage_island_and_cholmod_lm_damping(tmp_path: Path):
    """验证无电压锚定纯功率孤岛检测、CHOLMOD 原生 LM 阻尼不永久禁用及带噪 DC 孤岛平启动稳定收敛。"""
    from scipy.sparse import csc_matrix, csr_matrix
    from secore.se_math import (
        CHOLMOD_ANALYZE_AAT,
        CholmodAAtNormalEquationSolver,
        unanchored_voltage_island_state_indices,
    )

    # 1. 验证 unanchored_voltage_island_state_indices 精准识别无电压锚定孤岛
    # 分量 A: 节点 0-1-2（节点 0 有电压锚点）；分量 B: 节点 3-4-5（纯支路功率互连，无电压锚点）
    H_isl = csr_matrix(
        (
            [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            ([0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 0, 1, 1, 2, 3, 4, 4, 5]),
        ),
        shape=(5, 6),
        dtype=np.float64,
    )
    unanch = unanchored_voltage_island_state_indices(H_isl, range(6), anchored_voltage_cols=[0])
    assert unanch == [3]

    # 2. 验证 CholmodAAtNormalEquationSolver 奇异时不永久禁用且原生 LM 阻尼正确
    if CHOLMOD_ANALYZE_AAT is not None:
        solver = CholmodAAtNormalEquationSolver(assume_fixed_pattern=False)
        A_sing = csc_matrix(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64))
        rhs2 = np.array([1.0, 1.0], dtype=np.float64)
        # 无阻尼奇异回退但不永久拉黑 CHOLMOD
        _dx_fb, _ = solver.solve(A_sing, rhs2, damping=0.0)
        assert solver._cholmod_disabled is False
        # 原生 beta LM 阻尼求解精确匹配 (A A^T + beta I)^-1 rhs
        dx_lm, _ = solver.solve(A_sing, rhs2, damping=1e-2)
        expected_lm = np.linalg.solve(A_sing.toarray() @ A_sing.toarray().T + 1e-2 * np.eye(2), rhs2)
        assert np.allclose(dx_lm, expected_lm, rtol=1e-11, atol=1e-11)
        assert solver._cholmod_disabled is False
        # 紧接着健康矩阵仍正常走 CHOLMOD
        A_ok = csc_matrix(np.eye(2, dtype=np.float64))
        dx_ok, _ = solver.solve(A_ok, rhs2, damping=0.0)
        assert np.allclose(dx_ok, rhs2, rtol=1e-12, atol=1e-12)
        assert solver.last_cholmod_factor is not None

    # 3. 构造含 2x2 无电压锚定纯功率孤岛 + 反号噪声支路量测的 DC 算例，验证平启动稳定收敛于物理上支
    dc_e = tmp_path / "dc_unanchored_island.e"
    dc_m = tmp_path / "dc_unanchored_island.meas"
    dc_e.write_text(
        "\n".join(
            [
                "<Model>",
                "@ p_base u_unit p_unit i_unit",
                "# 100.0 kV MW kA",
                "</Model>",
                "<DCNode>",
                "@ idx name vbase voltage isl run_stat",
                "# 1 b1 400.0 400.0 1 1",
                "# 2 b2 400.0 400.0 1 1",
                "# 3 b3 400.0 400.0 1 1",
                "# 4 b4 400.0 400.0 1 1",
                "# 5 b5 400.0 400.0 1 1",
                "# 6 b6 400.0 400.0 1 1",
                "</DCNode>",
                "<DCBranch>",
                "@ idx name i_node j_node r run_stat",
                "# 1 br12 1 2 0.01 1",
                "# 2 br23 2 3 0.01 1",
                "# 3 br34 3 4 0.01 1",
                "# 4 br45 4 5 0.01 1",
                "# 5 br56 5 6 0.01 1",
                "# 6 br63 6 3 0.01 1",
                "</DCBranch>",
                "<DCGenerator>",
                "@ idx name node control_type p_set v_set droop_k run_stat",
                "# 1 g1 1 V 0.0 400.0 0.0 1",
                "</DCGenerator>",
                "<DCLoad>",
                "@ idx name node pbase pv0 pv1 pv2 run_stat",
                "# 1 l1 2 0.1 1.0 0.0 0.0 1",
                "</DCLoad>",
            ]
        )
        + "\n",
        encoding="utf8",
    )
    # 节点 1,2 有效；br23 缺测使 {3,4,5,6} 成为纯支路功率且带反号噪声的无电压锚定孤岛
    dc_m.write_text(
        "\n".join(
            [
                "<Measurement>",
                "@ idx name dev_type dev_name meas_type weight valid value",
                "# 1 m1 DCNode b1 V 1600.0 1 400.0",
                "# 2 m2 DCNode b2 V 1600.0 1 399.8",
                "# 3 m3 DCBranch br12 P_FROM 4000.0 1 0.1",
                "# 4 m4 DCBranch br34 P_FROM 4000.0 1 -0.6",
                "# 5 m5 DCBranch br34 P_TO 4000.0 1 0.75",
                "# 6 m6 DCBranch br45 P_FROM 4000.0 1 -0.5",
                "# 7 m7 DCBranch br56 P_FROM 4000.0 1 0.6",
                "# 8 m8 DCBranch br63 P_FROM 4000.0 1 -0.55",
                "# 9 m9 DCBranch br63 P_TO 4000.0 1 0.65",
                "</Measurement>",
            ]
        )
        + "\n",
        encoding="utf8",
    )
    est = DCStateEstimator(e_file=dc_e, meas_file=dc_m, parameters=SE_PARAMS, flat_start=True)
    res = est.estimate(verbose=False)
    assert res.converged is True
    assert float(res.x[: est.n_voltage].min()) > 0.8


def test_se_direct_strong_peaks_and_coarse_early_stop_equivalence(tmp_path: Path, monkeypatch):
    """验证强下界局部主导峰值免 CHOLMOD 回代、1 跳竞争对手强制精确回代及粗收敛早停终轮精确等价。"""
    from scipy.sparse import csr_matrix
    import secore.se_math as se_math_mod
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        compute_normalized_residuals_for_bad_data,
        screen_initial_bad_data_fast,
    )

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    # 1. 构造 10x10 网格雅可比
    nx, ny = 10, 10
    n_s = nx * ny
    rows, cols, vals = [], [], []
    r_id = 0
    for i in range(n_s):
        rows.append(r_id)
        cols.append(i)
        vals.append(1.0)
        r_id += 1
    for ix in range(nx):
        for iy in range(ny):
            u = ix * ny + iy
            if ix + 1 < nx:
                v = (ix + 1) * ny + iy
                rows.extend([r_id, r_id])
                cols.extend([u, v])
                vals.extend([1.0, -1.0])
                r_id += 1
            if iy + 1 < ny:
                v = ix * ny + (iy + 1)
                rows.extend([r_id, r_id])
                cols.extend([u, v])
                vals.extend([1.0, -1.0])
                r_id += 1
    m_s = r_id
    H_sp = csr_matrix((vals, (rows, cols)), shape=(m_s, n_s), dtype=np.float64)
    w = np.full(m_s, 100.0, dtype=np.float64)
    G, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
    fac = CHOLMOD_CHOLESKY(G.tocsc())
    lev_dense = measurement_leverage(H_sp, np.linalg.inv(G.toarray()))

    # 监控 exact_leverage_for_candidate_rows 调用次数与入参
    call_records = []
    orig_exact_lev = se_math_mod.exact_leverage_for_candidate_rows

    def _spy_exact_lev(H_arg, cand_arg, fac_arg, *args, **kwargs):
        call_records.append(np.asarray(cand_arg, dtype=np.intp).copy())
        return orig_exact_lev(H_arg, cand_arg, fac_arg, *args, **kwargs)

    monkeypatch.setattr(se_math_mod, "exact_leverage_for_candidate_rows", _spy_exact_lev)

    # 场景 A：5 个彼此远离的孤立强峰值 + 1 条大残差伪量测 -> 0 次 CHOLMOD 回代、伪量测被排除
    res_iso = np.full(m_s, 1e-4, dtype=np.float64)
    iso_rows = [0, 25, 55, 78, 99]
    for r in iso_rows:
        res_iso[r] = 2.0  # lb = 20.0 >= 8.0
    pseudo_row = 44
    res_iso[pseudo_row] = 5.0  # 伪量测即便下界 50 也应被排除
    is_pseudo = np.zeros(m_s, dtype=bool)
    is_pseudo[pseudo_row] = True

    st_a, rows_a, norms_a = screen_initial_bad_data_fast(
        res_iso,
        H_sp,
        w,
        is_pseudo,
        fac,
        threshold=4.0,
        skip_upper_cert=True,
        allow_direct_strong_peaks=True,
    )
    assert st_a == "candidates"
    assert len(call_records) == 0  # 完全免 CHOLMOD 回代
    assert set(rows_a.tolist()) == set(iso_rows)
    assert pseudo_row not in set(rows_a.tolist())
    exact_a, _ = compute_normalized_residuals_for_bad_data(res_iso, w, lev_dense, is_pseudo=is_pseudo)
    # 保守严格下界必须 <= 精确 r_N 且 > threshold
    assert np.all(norms_a <= exact_a[rows_a] + 1e-12)
    assert np.all(norms_a >= 8.0)

    # 场景 B：相邻 1 跳竞争对手 (节点 11 与 12 共享支路，lb 接近 15.0 vs 14.0 -> 14 > 0.6*15)
    # 必须触发精确回代以精确比较二者 r_N，仅保留胜者，且远处孤立强峰值 88 仍免回代
    res_comp = np.full(m_s, 1e-4, dtype=np.float64)
    res_comp[11] = 1.5   # lb = 15.0
    res_comp[12] = 1.4   # lb = 14.0 > 0.6 * 15.0（1 跳竞争对手）
    res_comp[88] = 2.5   # lb = 25.0（远距孤立强峰值）
    call_records.clear()
    st_b, rows_b, norms_b = screen_initial_bad_data_fast(
        res_comp,
        H_sp,
        w,
        np.zeros(m_s, dtype=bool),
        fac,
        threshold=4.0,
        skip_upper_cert=True,
        allow_direct_strong_peaks=True,
    )
    assert st_b == "candidates"
    assert len(call_records) == 1
    queried_set = set(call_records[0].tolist())
    assert 11 in queried_set and 12 in queried_set
    assert 88 not in queried_set  # 远距孤立强峰值不进入回代
    assert {11, 12, 88}.issubset(set(rows_b.tolist()))
    exact_b, _ = compute_normalized_residuals_for_bad_data(res_comp, w, lev_dense)
    norm_map_b = dict(zip(rows_b.tolist(), norms_b.tolist()))
    assert np.isclose(norm_map_b[11], exact_b[11], rtol=1e-10, atol=1e-10)
    assert np.isclose(norm_map_b[12], exact_b[12], rtol=1e-10, atol=1e-10)

    # 2. 验证 AC estimate() 粗收敛早停机制：干净场景绝不误触发，多坏数据场景触发后精算与无粗停完全等价
    ac_e, ac_m = _generate_ac_case(18, tmp_path, seed=2031, inject_bad=False)
    ac_est = ACStateEstimator(e_file=ac_e, meas_file=ac_m, parameters=SE_PARAMS, flat_start=True)
    ac_est_ref = ACStateEstimator(e_file=ac_e, meas_file=ac_m, parameters=SE_PARAMS, flat_start=True)
    res_exact_ac = ac_est_ref.estimate(verbose=False)
    assert res_exact_ac.converged is True
    assert not getattr(res_exact_ac, "_coarse_stopped", False)
    res_clean_coarse = ac_est.estimate(verbose=False, _coarse_bad_data_threshold=4.0)
    assert res_clean_coarse.converged is True
    assert not getattr(res_clean_coarse, "_coarse_stopped", False)
    assert np.allclose(res_clean_coarse.x, res_exact_ac.x, rtol=1e-12, atol=1e-12)

    # 验证带 12 处大坏数据时，粗收敛早停在 iter>=3 安全停止，且热启动补精算与直接精算完全等价
    base_plan_tables = ac_est._active_measurement_plan_tables_ref()
    base_table = ac_est._common_measurement_plan_table(base_plan_tables).table
    orig_vals = base_table.value.copy()
    try:
        # 注入 12 条显性大坏数据
        step_stride = max(1, int(base_table.value.size) // 14)
        for k in range(12):
            idx_k = (k + 1) * step_stride
            if idx_k < base_table.value.size:
                base_table.value[idx_k] += 0.35
        res_bad_exact = ac_est.estimate(base_plan_tables, verbose=False)
        res_bad_coarse = ac_est.estimate(base_plan_tables, verbose=False, _coarse_bad_data_threshold=4.0)
        assert res_bad_exact.converged is True
        if getattr(res_bad_coarse, "_coarse_stopped", False):
            assert res_bad_coarse.iterations <= res_bad_exact.iterations
            res_bad_refined = ac_est.estimate(base_plan_tables, x0=res_bad_coarse.x, verbose=False)
            assert res_bad_refined.converged is True
            assert np.allclose(res_bad_refined.x, res_bad_exact.x, rtol=1e-6, atol=1e-6)
        else:
            assert res_bad_coarse.converged is True
            assert np.allclose(res_bad_coarse.x, res_bad_exact.x, rtol=1e-6, atol=1e-6)
    finally:
        base_table.value[:] = orig_vals


def test_se_cholesky_L_selected_inv_spmm_backfill_and_medium_skip(monkeypatch):
    """验证 factor.L() 超节点选元求逆、半矩阵 Z_sub SpMM 杠杆（含缺失补齐）及 >=32 强峰值跳过中等下界回代。"""
    from scipy.sparse import csc_matrix, csr_matrix
    import secore.se_math as se_math_mod
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        _measurement_leverage_from_selected_inv,
        _supernodal_selected_inverse_from_cholmod,
        screen_initial_bad_data_fast,
    )

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    # 1. 构造 16x16 二维网格（256 状态），包含 4 元/8 元宽行与显式零元，验证 L() 选元求逆与 SpMM 杠杆精确性
    nx, ny = 16, 16
    n_s = nx * ny
    rows, cols, vals = [], [], []
    r_id = 0
    for i in range(n_s):
        rows.append(r_id)
        cols.append(i)
        vals.append(1.0)
        r_id += 1
    for ix in range(nx):
        for iy in range(ny):
            u = ix * ny + iy
            if ix + 1 < nx:
                v = (ix + 1) * ny + iy
                rows.extend([r_id, r_id])
                cols.extend([u, v])
                vals.extend([1.0, -1.0])
                r_id += 1
            if iy + 1 < ny:
                v = ix * ny + (iy + 1)
                rows.extend([r_id, r_id])
                cols.extend([u, v])
                vals.extend([1.0, -1.0])
                r_id += 1
    # 附加 8 元星形功率平衡行（含显式 0.0 项测试）
    for ix in range(1, nx - 1, 2):
        for iy in range(1, ny - 1, 2):
            c = ix * ny + iy
            nbrs = [c, c - 1, c + 1, c - ny, c + ny, c - ny - 1, c + ny + 1]
            for k, col_id in enumerate(nbrs):
                rows.append(r_id)
                cols.append(col_id)
                vals.append(0.0 if k == len(nbrs) - 1 else (1.5 if k == 0 else -0.3))
            r_id += 1
    m_s = r_id
    H_sp = csr_matrix((vals, (rows, cols)), shape=(m_s, n_s), dtype=np.float64)
    rng = np.random.default_rng(2032)
    w = rng.uniform(50.0, 400.0, size=m_s)
    G, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
    fac = CHOLMOD_CHOLESKY(G.tocsc())

    Z_low, P = _supernodal_selected_inverse_from_cholmod(fac)
    assert Z_low is not None and P is not None
    G_inv_dense = np.linalg.inv(G.toarray())
    G_inv_perm = G_inv_dense[np.ix_(P, P)]
    z_coo = Z_low.tocoo()
    assert np.allclose(z_coo.data, G_inv_perm[z_coo.row, z_coo.col], rtol=1e-11, atol=1e-11)

    lev_dense = measurement_leverage(H_sp, G_inv_dense)
    lev_spmm = _measurement_leverage_from_selected_inv(H_sp, Z_low, P, cholmod_factor=fac, _force_spmm=True)
    assert lev_spmm is not None
    assert np.allclose(lev_spmm, lev_dense, rtol=1e-11, atol=1e-11)

    # 人工抹去 Z_low 中 3 个严格下三角非零元以模拟数值抵消缺失，验证 SpMM 路径 CHOLMOD 补齐精确一致
    z_coo2 = Z_low.tocoo()
    strict_lower = np.flatnonzero(z_coo2.row > z_coo2.col)
    drop_idx = set(strict_lower[:3].tolist())
    keep_mask = np.array([i not in drop_idx for i in range(z_coo2.nnz)], dtype=bool)
    Z_dropped = csc_matrix(
        (z_coo2.data[keep_mask], (z_coo2.row[keep_mask], z_coo2.col[keep_mask])),
        shape=Z_low.shape,
    )
    Z_dropped.sort_indices()
    lev_backfill = _measurement_leverage_from_selected_inv(H_sp, Z_dropped, P, cholmod_factor=fac, _force_spmm=True)
    assert lev_backfill is not None
    assert np.allclose(lev_backfill, lev_dense, rtol=1e-11, atol=1e-11)

    # 2. 验证 40 个强主导峰值 + 15 个中等下界行并存时：40 个强峰值免回代，仅 15 个中等下界行进入精确回代且无遗漏
    call_records = []
    orig_exact_lev = se_math_mod.exact_leverage_for_candidate_rows

    def _spy_exact_lev(H_arg, cand_arg, fac_arg, *args, **kwargs):
        call_records.append(np.asarray(cand_arg, dtype=np.intp).copy())
        return orig_exact_lev(H_arg, cand_arg, fac_arg, *args, **kwargs)

    monkeypatch.setattr(se_math_mod, "exact_leverage_for_candidate_rows", _spy_exact_lev)

    H_iso = csr_matrix(
        (np.ones(n_s, dtype=np.float64), (np.arange(n_s), np.arange(n_s))),
        shape=(n_s, n_s),
    )
    w_iso = np.full(n_s, 100.0, dtype=np.float64)
    G_iso, _ = build_normal_equations(H_iso, np.zeros(n_s), w_iso, dense_gain_limit=0)
    fac_iso = CHOLMOD_CHOLESKY(G_iso.tocsc())
    res_many = np.full(n_s, 1e-4, dtype=np.float64)
    strong_ids = list(range(0, 40))
    medium_ids = list(range(100, 115))
    res_many[strong_ids] = 1.5  # lb = 15.0 >= 8.0
    res_many[medium_ids] = 0.5  # lb = 5.0 (4.0 < lb < 8.0)

    st_m, rows_m, norms_m = screen_initial_bad_data_fast(
        res_many,
        H_iso,
        w_iso,
        np.zeros(n_s, dtype=bool),
        fac_iso,
        threshold=4.0,
        skip_upper_cert=True,
        allow_direct_strong_peaks=True,
    )
    assert st_m == "candidates"
    assert len(call_records) == 1
    assert set(call_records[0].tolist()) == set(medium_ids)  # 40 个强峰值免回代，仅 15 个中等下界行回代
    assert set(strong_ids).issubset(set(rows_m.tolist()))


def test_se_4hop_batch_woodbury_solver_and_pseudo_moved_exemption():
    """验证 >=4 跳隔离批删、WoodburyRemovalSolver 多步复用精确性及纯伪量测盲节点位移豁免 moved_cols。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        screen_bad_data_after_removal,
        select_independent_bad_data_batch,
        solve_woodbury_after_row_removal,
    )

    # 1. 链式图 15 状态：测试 min_hops=4 严格封锁距离 1/2/3 跳的节点，仅放行距离 >=4 跳的节点
    n = 15
    rows, cols, vals = [], [], []
    r_id = 0
    for c in range(n):
        rows.append(r_id)
        cols.append(c)
        vals.append(1.0)
        r_id += 1
    for c in range(n - 1):
        rows.extend([r_id, r_id])
        cols.extend([c, c + 1])
        vals.extend([1.0, -1.0])
        r_id += 1
    H_chain = csr_matrix((vals, (rows, cols)), shape=(r_id, n), dtype=np.float64)
    cands = [0, 3, 4, 8, 11]
    norms = [4.9, 4.8, 4.7, 4.6, 4.5]
    chosen_3 = select_independent_bad_data_batch(H_chain, cands, norms, threshold=4.0, min_batch_norm=4.0, min_hops=3)
    chosen_4 = select_independent_bad_data_batch(H_chain, cands, norms, threshold=4.0, min_batch_norm=4.0, min_hops=4)
    assert [cands[i] for i in chosen_3] == [0, 3, 8, 11]
    assert [cands[i] for i in chosen_4] == [0, 4, 8]  # 3 跳的 3 和 11(距8仅3跳)均被 4 跳策略封锁

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    # 2. 验证 WoodburyRemovalSolver 单行与批量多步复用精确等价于重构 G_new
    rng = np.random.default_rng(2033)
    n_s = 60
    m_s = 160
    H_dense = np.zeros((m_s, n_s), dtype=np.float64)
    for i in range(n_s):
        H_dense[i, i] = 1.0
    for i in range(n_s, m_s):
        a, b = rng.choice(n_s, size=2, replace=False)
        H_dense[i, a] = float(rng.uniform(0.5, 2.0))
        H_dense[i, b] = float(rng.uniform(-2.0, -0.5))
    H_sp = csr_matrix(H_dense)
    w = rng.uniform(50.0, 400.0, size=m_s)
    G_old, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
    fac_old = CHOLMOD_CHOLESKY(G_old.tocsc())

    for rem_rows in ([80], [70, 82, 95, 110, 140]):
        w_new = w.copy()
        w_new[rem_rows] = 0.0
        G_new, _ = build_normal_equations(H_sp, np.zeros(m_s), w_new, dense_gain_limit=0)
        G_new_dense = G_new.toarray()
        rhs1 = rng.normal(0.0, 1.0, size=n_s)
        rhs2 = rng.normal(0.0, 1.0, size=n_s)
        dx1, wb_solver = solve_woodbury_after_row_removal(H_sp, w, rem_rows, rhs1, fac_old, return_solver=True)
        assert dx1 is not None and wb_solver is not None
        dx2 = wb_solver.solve(rhs2)
        assert dx2 is not None
        assert np.allclose(dx1, np.linalg.solve(G_new_dense, rhs1), rtol=1e-11, atol=1e-11)
        assert np.allclose(dx2, np.linalg.solve(G_new_dense, rhs2), rtol=1e-11, atol=1e-11)

    # 3. 验证纯伪量测盲节点状态列发生 >0.3 大位移（数量=100 > 64）时，不误触发 local_only_shift=False 回退
    n_real = 450
    n_blind = 100
    n_tot = n_real + n_blind  # 550 > 512 触发大网自检
    r_rows, r_cols, r_vals = [], [], []
    rid = 0
    for i in range(n_real):
        r_rows.append(rid); r_cols.append(i); r_vals.append(1.0); rid += 1
        if i + 1 < n_real:
            r_rows.extend([rid, rid]); r_cols.extend([i, i + 1]); r_vals.extend([1.0, -1.0]); rid += 1
    n_real_meas = rid
    # 100 个纯伪量测盲节点（不连接任何真实量测）
    for i in range(n_real, n_tot):
        r_rows.append(rid); r_cols.append(i); r_vals.append(1.0); rid += 1
    m_tot = rid
    H_mix = csr_matrix((r_vals, (r_rows, r_cols)), shape=(m_tot, n_tot), dtype=np.float64)
    w_mix = np.full(m_tot, 100.0, dtype=np.float64)
    w_mix[n_real_meas:] = 1e-3
    is_pseudo_mix = np.zeros(m_tot, dtype=bool)
    is_pseudo_mix[n_real_meas:] = True

    G_m0, _ = build_normal_equations(H_mix, np.zeros(m_tot), w_mix, dense_gain_limit=0)
    fac_m0 = CHOLMOD_CHOLESKY(G_m0.tocsc())
    lev_m0 = measurement_leverage(H_mix, np.linalg.inv(G_m0.toarray()))

    rem_r = 10
    w_m1 = w_mix.copy()
    w_m1[rem_r] = 0.0
    mask_m1 = np.ones(m_tot, dtype=bool)
    mask_m1[rem_r] = False
    G_m1, _ = build_normal_equations(H_mix, np.zeros(m_tot), w_m1, dense_gain_limit=0)
    fac_m1 = CHOLMOD_CHOLESKY(G_m1.tocsc())
    res_clean = rng.normal(0.0, 0.05, size=m_tot)  # 约 0.5 sigma，无坏数据
    res_clean[rem_r] = 0.0

    # 100 个纯伪量测盲节点全部位移 0.35（>0.25 且数量 100 > 64），真实节点仅微动 1e-4
    dx_vec = np.full(n_tot, 1e-4, dtype=np.float64)
    dx_vec[n_real:] = 0.35
    st, prop_l, hit_r, _ = screen_bad_data_after_removal(
        lev_m0,
        H_mix,
        w_mix,
        rem_r,
        fac_m0,
        res_clean,
        H_mix,
        w_m1,
        is_pseudo_mix,
        fac_m1,
        threshold=4.0,
        state_delta_inf=0.35,
        state_delta_vec=dx_vec,
        active_mask=mask_m1,
        keep_shape=True,
    )
    assert st == "verified_clean" and prop_l is not None and hit_r is not None and hit_r.size == 0


def test_se_p2_woodbury_callable_deferred_anchor_and_high_blind_moved_rows(tmp_path):
    """验证 WoodburyRemovalSolver 多右端 __call__、延迟因子防近似终止与高盲区 moved_rows 伪量测过滤。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        build_normal_equations,
        exact_leverage_for_candidate_rows,
        measurement_leverage,
        screen_bad_data_after_removal,
        solve_woodbury_after_row_removal,
        update_leverage_after_row_removal,
    )

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    rng = np.random.default_rng(2034)
    n_s = 80
    m_s = 220
    H_dense = np.zeros((m_s, n_s), dtype=np.float64)
    for i in range(n_s):
        H_dense[i, i] = 1.0
    for i in range(n_s, m_s):
        a, b = rng.choice(n_s, size=2, replace=False)
        H_dense[i, a] = float(rng.uniform(0.6, 1.8))
        H_dense[i, b] = float(rng.uniform(-1.8, -0.6))
    H_sp = csr_matrix(H_dense)
    w = rng.uniform(50.0, 300.0, size=m_s)
    G_old, _ = build_normal_equations(H_sp, np.zeros(m_s), w, dense_gain_limit=0)
    fac_old = CHOLMOD_CHOLESKY(G_old.tocsc())
    lev_old = measurement_leverage(H_sp, np.linalg.inv(G_old.toarray()))

    for rem_rows in ([95], [95, 112, 130, 155]):
        w_new = w.copy()
        w_new[rem_rows] = 0.0
        G_new, _ = build_normal_equations(H_sp, np.zeros(m_s), w_new, dense_gain_limit=0)
        G_new_dense = G_new.toarray()
        G_new_inv = np.linalg.inv(G_new_dense)
        rhs_1d = rng.normal(0.0, 1.0, size=n_s)
        dx1, wb_solver = solve_woodbury_after_row_removal(H_sp, w, rem_rows, rhs_1d, fac_old, return_solver=True)
        assert dx1 is not None and wb_solver is not None and getattr(wb_solver, "is_woodbury_solver", False)
        # 1. 多右端 2D __call__ 与精确线性求解一致
        rhs_2d = rng.normal(0.0, 1.0, size=(n_s, 7))
        sol_2d = wb_solver(rhs_2d)
        assert sol_2d.shape == (n_s, 7)
        assert np.allclose(sol_2d, np.linalg.solve(G_new_dense, rhs_2d), rtol=1e-11, atol=1e-11)
        # 2. exact_leverage_for_candidate_rows 无缝兼容 wb_solver
        cand_idx = np.array([0, 10, 85, 140, 205], dtype=np.intp)
        lev_wb = exact_leverage_for_candidate_rows(H_sp, cand_idx, wb_solver)
        lev_exact_all = measurement_leverage(H_sp, G_new_inv)
        assert lev_wb is not None
        assert np.allclose(lev_wb, lev_exact_all[cand_idx], rtol=1e-11, atol=1e-11)
        if len(rem_rows) == 1:
            prop_lev = update_leverage_after_row_removal(
                lev_old, H_sp, w, rem_rows[0], wb_solver, keep_shape=True, factor_is_post_removal=True
            )
            assert prop_lev is not None
            keep_m = np.ones(m_s, dtype=bool)
            keep_m[rem_rows[0]] = False
            assert np.allclose(prop_lev[keep_m], lev_exact_all[keep_m], rtol=1e-10, atol=1e-10)

    # 3. 延迟 Woodbury 因子在 screen_bad_data_after_removal 中：有坏数据时返回 candidates，无坏数据时绝不近似 verified_clean
    rem_r = 95
    w_1 = w.copy()
    w_1[rem_r] = 0.0
    mask_1 = np.ones(m_s, dtype=bool)
    mask_1[rem_r] = False
    _, wb_1 = solve_woodbury_after_row_removal(H_sp, w, [rem_r], np.ones(n_s), fac_old, return_solver=True)
    res_bad = np.full(m_s, 1e-4, dtype=np.float64)
    res_bad[rem_r] = 0.0
    res_bad[120] = 0.8  # 强坏数据
    st_cand, _, hit_cand, _ = screen_bad_data_after_removal(
        lev_old, H_sp, w, rem_r, fac_old, res_bad, H_sp, w_1, np.zeros(m_s, dtype=bool), wb_1, threshold=4.0,
        active_mask=mask_1, keep_shape=True, allow_fast_final_verify=True,
    )
    assert st_cand == "candidates" and hit_cand is not None and 120 in hit_cand.tolist()

    w_1_safe = np.where(w_1 > 0.0, w_1, 1.0)
    G_1, _ = build_normal_equations(H_sp, np.zeros(m_s), w_1, dense_gain_limit=0)
    lev_1 = measurement_leverage(H_sp, np.linalg.inv(G_1.toarray()))
    omega_1 = np.maximum(1.0 / w_1_safe - lev_1, 1e-12)
    res_quiet = 1.5 * np.sqrt(omega_1)
    res_quiet[rem_r] = 0.0
    st_clean, _, _, _ = screen_bad_data_after_removal(
        lev_old, H_sp, w, rem_r, fac_old, res_quiet, H_sp, w_1, np.zeros(m_s, dtype=bool), wb_1, threshold=4.0,
        active_mask=mask_1, keep_shape=True, allow_fast_final_verify=True,
    )
    assert st_clean == "need_full"

    # 4. 用 80 节点 AC 端到端多坏数据（n_state > 128，实际触发 CHOLMOD AAt 与跨轮持久锚点延迟 Woodbury 刷新）
    ac_e, ac_m = _generate_ac_case(80, tmp_path, seed=2035, inject_bad=False)
    ac_est = ACStateEstimator(e_file=ac_e, meas_file=ac_m, parameters=SE_PARAMS, flat_start=True)
    base_plan_tables = ac_est._active_measurement_plan_tables_ref()
    base_table = ac_est._common_measurement_plan_table(base_plan_tables).table
    orig_vals = base_table.value.copy()
    orig_active_z = ac_est.active_z.copy()
    try:
        # 在同一局部区域连续注入 3 条耦合坏数据，迫使走多轮连续单删并触发跨轮持久锚点 Woodbury 延迟刷新
        for pos in (20, 21, 22):
            if pos < base_table.value.size:
                base_table.value[pos] += 0.35
                ac_est.active_z[pos] += 0.35
        res_rm, removed = ac_est.estimate_with_bad_data_removal(threshold=4.0, max_remove=50)
        assert res_rm.converged is True
        assert len(removed) >= 3
        rem_after, _ = ac_est.identify_bad_data(res_rm, threshold=4.0)
        assert len(rem_after) == 0
    finally:
        base_table.value[:] = orig_vals
        ac_est.active_z[:] = orig_active_z


def test_se_small_scale_structural_rank_no_svd_and_cholmod_eye_leverage(monkeypatch):
    """验证小规模稀疏结构缺秩零 SVD 一次定位、数值缺秩 SVD 兜底与小矩阵 cholmod_factor(I) 精确杠杆。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        build_normal_equations,
        measurement_leverage,
        measurement_leverage_for_bad_data,
        observability_rank_details,
    )

    # 1. 100 状态稀疏矩阵缺失 35 个盲节点：应一次性定位全部 35 个弱状态，绝不调用稠密 SVD
    n_small = 100
    blind_cols = set(range(10, 45))
    rows, cols, vals = [], [], []
    rid = 0
    for c in range(n_small):
        if c not in blind_cols:
            rows.append(rid); cols.append(c); vals.append(1.0); rid += 1
    H_blind = csr_matrix((vals, (rows, cols)), shape=(rid, n_small), dtype=np.float64)

    orig_svd = np.linalg.svd
    svd_calls = []
    def tracked_svd(*args, **kwargs):
        svd_calls.append(1)
        return orig_svd(*args, **kwargs)
    monkeypatch.setattr(np.linalg, "svd", tracked_svd)

    rank_b, def_b, _, weak_b = observability_rank_details(H_blind, n_small)
    assert len(svd_calls) == 0
    assert def_b == len(blind_cols)
    assert rank_b == n_small - len(blind_cols)
    assert {idx for idx, _ in weak_b} == blind_cols

    # 2. 结构满匹配但数值共线（列 0 与列 1 完全成比例）：应安全回退稠密 SVD 准确检出 deficiency=1
    H_num = np.eye(12, dtype=np.float64)
    H_num[:, 1] = 2.0 * H_num[:, 0]
    H_num[1, 1] = 0.0
    H_num_sp = csr_matrix(np.vstack([H_num, H_num[0:1, :]]))
    # 第 1 行改回让结构满匹配但数值线性相关：row0=[1,2,0..], row1=[2,4,0..]
    H_collinear = np.eye(12, dtype=np.float64)
    H_collinear[0, 0] = 1.0; H_collinear[0, 1] = 2.0
    H_collinear[1, 0] = 2.0; H_collinear[1, 1] = 4.0
    svd_calls.clear()
    rank_c, def_c, _, weak_c = observability_rank_details(csr_matrix(H_collinear), 12)
    assert len(svd_calls) == 1
    assert rank_c == 11 and def_c == 1 and len(weak_c) > 0

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    # 3. 小规模（n=120 <= 320）带缓存 cholmod_factor 且 gain=None：直接走 cholmod_factor(I) 与稠密逆一致
    rng = np.random.default_rng(2036)
    n_m = 120
    m_m = 320
    H_d = np.zeros((m_m, n_m), dtype=np.float64)
    for i in range(n_m):
        H_d[i, i] = 1.0
    for i in range(n_m, m_m):
        a, b = rng.choice(n_m, size=2, replace=False)
        H_d[i, a] = float(rng.uniform(0.5, 2.0))
        H_d[i, b] = float(rng.uniform(-2.0, -0.5))
    H_sp = csr_matrix(H_d)
    w = rng.uniform(20.0, 200.0, size=m_m)
    G_sp, _ = build_normal_equations(H_sp, np.zeros(m_m), w, dense_gain_limit=0)
    fac = CHOLMOD_CHOLESKY(G_sp.tocsc())
    lev_fast = measurement_leverage_for_bad_data(H_sp, None, cholmod_factor=fac)
    lev_ref = measurement_leverage(H_sp, np.linalg.inv(G_sp.toarray()))
    assert lev_fast is not None
    assert np.allclose(lev_fast, lev_ref, rtol=1e-11, atol=1e-11)


def test_se_4scale_leverage_threshold_verify_cap_and_dc_deferred_woodbury(tmp_path: Path, monkeypatch):
    """验证 ns=400 走 cholmod_factor(I) 杠杆精确一致、>=128 强峰值成对截断回代与 DC 延迟锚点 Woodbury 一致性。"""
    from scipy.sparse import csr_matrix
    import secore.se_math as se_math_mod
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        build_normal_equations,
        measurement_leverage,
        measurement_leverage_for_bad_data,
        screen_initial_bad_data_fast,
    )

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    # 1. ns=400 (320 < ns <= 512) 应走 cholmod_factor(I) 精确杠杆而不触发选元求逆，且与稠密逆完全一致
    rng = np.random.default_rng(2037)
    n_m = 400
    m_m = 900
    rows_l, cols_l, vals_l = [], [], []
    for i in range(n_m):
        rows_l.append(i)
        cols_l.append(i)
        vals_l.append(1.0)
    for i in range(n_m, m_m):
        a, b = rng.choice(n_m, size=2, replace=False)
        rows_l.extend([i, i])
        cols_l.extend([int(a), int(b)])
        vals_l.extend([float(rng.uniform(0.5, 2.0)), float(rng.uniform(-2.0, -0.5))])
    H_400 = csr_matrix((vals_l, (rows_l, cols_l)), shape=(m_m, n_m), dtype=np.float64)
    w_400 = rng.uniform(20.0, 200.0, size=m_m)
    G_400, _ = build_normal_equations(H_400, np.zeros(m_m), w_400, dense_gain_limit=0)
    fac_400 = CHOLMOD_CHOLESKY(G_400.tocsc())

    selinv_calls = []
    orig_selinv = se_math_mod._supernodal_selected_inverse_from_cholmod

    def _spy_selinv(*args, **kwargs):
        selinv_calls.append(1)
        return orig_selinv(*args, **kwargs)

    monkeypatch.setattr(se_math_mod, "_supernodal_selected_inverse_from_cholmod", _spy_selinv)
    lev_400 = measurement_leverage_for_bad_data(H_400, None, cholmod_factor=fac_400)
    assert len(selinv_calls) == 0
    assert lev_400 is not None
    lev_400_ref = measurement_leverage(H_400, np.linalg.inv(G_400.toarray()))
    assert np.allclose(lev_400, lev_400_ref, rtol=1e-11, atol=1e-11)

    # 2. >=128 个强主导峰值 + 大批 1 跳竞争对手：精确回代预算收紧至 <=512 且每个入选竞争对成对保留 (peak, rival)
    n_p = 760
    r_rows, r_cols, r_vals = [], [], []
    rid = 0
    # 前 330 对形成独立 2-节点组件（共 660 个状态，每对 2 条孤立节点量测 + 1 条连接支路，互不干扰）
    pair_a_rows, pair_b_rows = [], []
    for k in range(330):
        u, v = 2 * k, 2 * k + 1
        r_rows.append(rid); r_cols.append(u); r_vals.append(1.0); pair_a_rows.append(rid); rid += 1
        r_rows.append(rid); r_cols.append(v); r_vals.append(1.0); pair_b_rows.append(rid); rid += 1
        r_rows.extend([rid, rid]); r_cols.extend([u, v]); r_vals.extend([1.0, -1.0]); rid += 1
    # 其余 100 个独立状态各挂 1 条量测
    iso_rows = []
    for u in range(660, n_p):
        r_rows.append(rid); r_cols.append(u); r_vals.append(1.0); iso_rows.append(rid); rid += 1
    H_p = csr_matrix((r_vals, (r_rows, r_cols)), shape=(rid, n_p), dtype=np.float64)
    w_p = np.full(rid, 100.0, dtype=np.float64)
    G_p, _ = build_normal_equations(H_p, np.zeros(rid), w_p, dense_gain_limit=0)
    fac_p = CHOLMOD_CHOLESKY(G_p.tocsc())

    res_p = np.full(rid, 1e-4, dtype=np.float64)
    # 140 个高优先级孤立强峰值（前 60 对仅激活 pair_a_rows，加上 80 个 iso_rows -> >=128 强主导免回代）
    for k in range(60):
        res_p[pair_a_rows[k]] = 9.0 + 0.01 * k
    for idx_i, r in enumerate(iso_rows[:80]):
        res_p[r] = 9.5 + 0.01 * idx_i
    # 另外 270 对设置强竞争对手（peak 与 rival 同处 1 跳且 lb 比 > 0.6），总竞争行 540 > 512
    rival_map = {}
    for k in range(60, 330):
        pa, pb = pair_a_rows[k], pair_b_rows[k]
        res_p[pa] = 3.0 + 0.02 * (k - 60)
        res_p[pb] = 2.8 + 0.02 * (k - 60)
        rival_map[pa] = pb
        rival_map[pb] = pa

    call_records = []
    orig_exact_lev = se_math_mod.exact_leverage_for_candidate_rows

    def _spy_exact_lev(H_arg, cand_arg, fac_arg, *args, **kwargs):
        call_records.append(np.asarray(cand_arg, dtype=np.intp).copy())
        return orig_exact_lev(H_arg, cand_arg, fac_arg, *args, **kwargs)

    monkeypatch.setattr(se_math_mod, "exact_leverage_for_candidate_rows", _spy_exact_lev)
    st_p, rows_p, _ = screen_initial_bad_data_fast(
        res_p,
        H_p,
        w_p,
        np.zeros(rid, dtype=bool),
        fac_p,
        threshold=4.0,
        skip_upper_cert=True,
        allow_direct_strong_peaks=True,
    )
    assert st_p == "candidates"
    assert len(call_records) == 1
    queried = set(call_records[0].tolist())
    assert len(queried) <= 512
    # 每个进入精确回代的竞争行都必须成对携带其竞争对手，绝不单边截断
    for q in queried:
        if q in rival_map:
            assert rival_map[q] in queried
    assert len(rows_p) >= 140

    # 3. DC 端到端多坏数据剔除：验证跨轮持久锚点延迟 Woodbury 正常工作且最终不泄漏 WoodburyRemovalSolver
    dc_e, dc_m = _generate_dc_case(80, tmp_path, seed=2038, inject_bad=False)
    dc_est = DCStateEstimator(e_file=dc_e, meas_file=dc_m, parameters=SE_PARAMS, flat_start=True)
    base_plan_tables = dc_est._active_measurement_plan_tables_ref()
    base_table = dc_est._common_measurement_plan_table(base_plan_tables).table
    orig_vals = base_table.value.copy()
    orig_active_z = dc_est.active_z.copy()
    try:
        # 在同一局部区域连续注入 3 条耦合坏数据，触发连续单删与跨轮持久锚点 Woodbury 延迟刷新
        for pos in (15, 16, 17):
            if pos < base_table.value.size:
                base_table.value[pos] += 0.35
                dc_est.active_z[pos] += 0.35
        res_dc, removed_dc = dc_est.estimate_with_bad_data_removal(threshold=4.0, max_remove=50)
        assert res_dc.converged is True
        assert len(removed_dc) >= 3
        fac_cache = getattr(dc_est, "_last_estimate_cholmod_factor", None)
        if fac_cache is not None:
            assert not getattr(fac_cache[1], "is_woodbury_solver", False)
        rem_after, _ = dc_est.identify_bad_data(res_dc, threshold=4.0)
        assert len(rem_after) == 0
    finally:
        base_table.value[:] = orig_vals
        dc_est.active_z[:] = orig_active_z


def test_se_adaptive_scale_profile_solveL_nesdis_and_simplicial_selinv():
    """验证 SEScaleProfile 多尺度自适应参数、solve_L 稀疏/稠密杠杆、nesdis 排序与双指针 simplicial 选元求逆精确性。"""
    from scipy.sparse import csc_matrix, csr_matrix
    from secore.se_math import (
        CHOLMOD_ANALYZE_AAT,
        CHOLMOD_CHOLESKY,
        CholmodAAtNormalEquationSolver,
        SEScaleProfile,
        _simplicial_selected_inverse_fast,
        _supernodal_selected_inverse_from_cholmod,
        build_normal_equations,
        exact_leverage_for_candidate_rows,
        measurement_leverage,
        measurement_leverage_for_bad_data,
    )

    # 1. 验证 SEScaleProfile 在 20 / 200 / 2000 / 20000(DC vs AC) / 100000 尺度下的平滑自适应行为
    p20 = SEScaleProfile.for_problem(20, 60, 120, multi_solve=True)
    p200_sp = SEScaleProfile.for_problem(200, 600, 1200, multi_solve=True)
    p2k = SEScaleProfile.for_problem(2000, 8000, 24000, multi_solve=True)
    p20k_dc = SEScaleProfile.for_problem(20000, 80000, 130000, multi_solve=True)
    p20k_ac = SEScaleProfile.for_problem(48060, 220000, 720000, multi_solve=True)
    p100k = SEScaleProfile.for_problem(100000, 400000, 1600000, multi_solve=True)

    assert p20.use_aat_normal_solver is True and p20.aat_ordering_method == "default"
    assert p20.woodbury_max_rank == 4 and not p20.fast_init_screen_enabled and not p20.coarse_early_stop_enabled
    assert p200_sp.leverage_eye_max_state == 768 and p200_sp.exact_leverage_use_sparse_rhs is True
    assert p2k.fast_init_screen_enabled is True and p2k.coarse_early_stop_enabled is False
    assert p2k.woodbury_max_rank > p200_sp.woodbury_max_rank
    assert p20k_dc.aat_ordering_method == "default" and p20k_dc.exact_leverage_use_sparse_rhs is True
    assert p20k_ac.aat_ordering_method == "nesdis" and p20k_ac.exact_leverage_use_sparse_rhs is False
    assert p20k_ac.woodbury_max_rank == 48 and p20k_ac.coarse_early_stop_enabled is True
    assert p100k.sparse_leverage_max_state >= 250000
    assert p100k.exact_leverage_use_sparse_rhs is False
    assert p100k.screen_verify_cap_rich == 16 and p100k.screen_verify_cap == 256
    assert p100k.exact_leverage_chunk_size <= p20k_ac.exact_leverage_chunk_size <= p2k.exact_leverage_chunk_size

    if CHOLMOD_CHOLESKY is None or CHOLMOD_ANALYZE_AAT is None:
        pytest.skip("CHOLMOD unavailable")

    rng = np.random.default_rng(2040)

    # 2. 极稀疏图（avg_nnz < 10 -> 稀疏 CSC solve_L）与较稠密图（avg_nnz >= 10 -> 列主序稠密分块 solve_L）精确杠杆一致
    for avg_tag, extra_rows, min_deg in (("sparse", 120, 2), ("dense", 420, 4)):
        n_s = 120
        r_l, c_l, v_l = [], [], []
        rid = 0
        for i in range(n_s):
            r_l.append(rid); c_l.append(i); v_l.append(1.0); rid += 1
        for _ in range(extra_rows):
            cols = rng.choice(n_s, size=min_deg, replace=False)
            for k_c, c_id in enumerate(cols):
                r_l.append(rid); c_l.append(int(c_id))
                v_l.append(float(rng.uniform(0.4, 1.8) * (1.0 if k_c % 2 == 0 else -1.0)))
            rid += 1
        H_sp = csr_matrix((v_l, (r_l, c_l)), shape=(rid, n_s), dtype=np.float64)
        w = rng.uniform(30.0, 200.0, size=rid)
        G, _ = build_normal_equations(H_sp, np.zeros(rid), w, dense_gain_limit=0)
        fac = CHOLMOD_CHOLESKY(G.tocsc())
        lev_ref = measurement_leverage(H_sp, np.linalg.inv(G.toarray()))
        cand = np.arange(0, rid, max(1, rid // 40), dtype=np.intp)
        lev_fast = exact_leverage_for_candidate_rows(H_sp, cand, fac)
        assert lev_fast is not None
        assert np.allclose(lev_fast, lev_ref[cand], rtol=1e-11, atol=1e-11), f"mismatch on {avg_tag}"

        # 验证 CholmodAAtNormalEquationSolver(ordering_method='nesdis') 与 default 求解等价且记录 last_normal_operand
        A_csc = (H_sp.T.multiply(np.sqrt(w))).tocsc()
        rhs = rng.normal(0.0, 1.0, size=n_s)
        s_def = CholmodAAtNormalEquationSolver(assume_fixed_pattern=True, ordering_method="default")
        s_nes = CholmodAAtNormalEquationSolver(assume_fixed_pattern=True, ordering_method="nesdis")
        dx_def, _ = s_def.solve(A_csc, rhs)
        dx_nes, _ = s_nes.solve(A_csc, rhs)
        assert np.allclose(dx_nes, dx_def, rtol=1e-11, atol=1e-11)
        assert s_nes.last_normal_operand is A_csc

        # 验证双指针 _simplicial_selected_inverse_fast 与超节点选元求逆精确一致
        L_csc = fac.L().tocsc()
        L_csc.sort_indices()
        Z_simp = _simplicial_selected_inverse_fast(L_csc)
        Z_sup, P_sup = _supernodal_selected_inverse_from_cholmod(fac, aat_operand=A_csc)
        assert Z_sup is not None and P_sup is not None
        G_inv = np.linalg.inv(G.toarray())
        P_fac = np.asarray(fac.P(), dtype=np.intp)
        G_inv_p = G_inv[np.ix_(P_fac, P_fac)]
        zc = Z_simp.tocoo()
        assert np.allclose(zc.data, G_inv_p[zc.row, zc.col], rtol=1e-11, atol=1e-11)

        # 验证 measurement_leverage_for_bad_data 透传 aat_operand 精确一致
        lev_aat = measurement_leverage_for_bad_data(
            H_sp, None, cholmod_factor=fac, aat_operand=A_csc, sparse_threshold=16
        )
        assert lev_aat is not None
        assert np.allclose(lev_aat, lev_ref, rtol=1e-11, atol=1e-11)


def test_ultra_scale_observability_no_4096_truncation_and_fast_screen_rich_peaks():
    """验证 13 尺度优化核心不变量：>4096 缺秩单轮全量返回 + 强主导峰值富余免冗余回代。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        SEScaleProfile,
        build_normal_equations,
        observability_rank_details,
        screen_initial_bad_data_fast,
    )

    # 1. 构造含 5000 个结构零覆盖列的 6000 维稀疏雅可比，验证 observability_rank_details 单轮全部返回（不被 4096 截断）
    n_state = 6000
    obs_cols = np.arange(0, 1000, dtype=np.int64)
    H_def = csr_matrix(
        (np.ones(obs_cols.size, dtype=np.float64), (np.arange(obs_cols.size, dtype=np.int64), obs_cols)),
        shape=(obs_cols.size, n_state),
    )
    rank, deficiency, _sv, weak_states = observability_rank_details(H_def, n_state)
    assert deficiency == 5000 and rank == 1000
    assert len(weak_states) == 5000
    assert weak_states[0][0] == 1000 and weak_states[-1][0] == 5999

    # 2. 验证 10 ~ 250k 状态 SEScaleProfile 批删与回代上限平滑单调且覆盖 10 万节点 AC
    p250k = SEScaleProfile.for_problem(250000, 1150000, 3800000, multi_solve=True)
    assert p250k.sparse_leverage_max_state >= 250000
    assert p250k.batch_max_base == 8192
    assert p250k.screen_primary_cap == 16384
    assert p250k.screen_verify_cap == 256 and p250k.screen_verify_cap_rich == 16

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    # 3. 构造含 40 个相互隔离的强坏数据（>32 触发 screen_verify_cap_rich=0），验证快速初筛精准锁定全部强峰且零漏检
    n_s = 600
    r_idx = np.concatenate([np.arange(n_s), np.arange(n_s, 2 * n_s - 1), np.arange(n_s, 2 * n_s - 1)])
    c_idx = np.concatenate([np.arange(n_s), np.arange(n_s - 1), np.arange(1, n_s)])
    v_dat = np.concatenate([np.ones(n_s), np.ones(n_s - 1), -np.ones(n_s - 1)])
    H_chain = csr_matrix((v_dat, (r_idx, c_idx)), shape=(2 * n_s - 1, n_s), dtype=np.float64)
    w = np.full(2 * n_s - 1, 100.0, dtype=np.float64)
    G, _ = build_normal_equations(H_chain, np.zeros(2 * n_s - 1), w, dense_gain_limit=0)
    fac = CHOLMOD_CHOLESKY(G.tocsc())
    res = np.zeros(2 * n_s - 1, dtype=np.float64)
    bad_rows = np.arange(5, n_s - 5, 14, dtype=np.intp)[:40]
    res[bad_rows] = 3.0  # lb = 3.0 * 10 = 30.0 >> direct_lb_gate=8.0
    status, hit_rows, hit_norms = screen_initial_bad_data_fast(
        res, H_chain, w, None, fac, threshold=4.0, allow_direct_strong_peaks=True
    )
    assert status == "candidates"
    assert set(bad_rows.tolist()).issubset(set(hit_rows.tolist()))
    assert float(np.min(hit_norms)) >= 29.9


def test_woodbury_incremental_u_col_cache_and_zero_backsolve_leverage():
    """验证同锚点 Woodbury 增量列缓存仅求解新增行、切换因子自动清空，且与精确解逐位等价。"""
    from scipy.sparse import csr_matrix
    from secore.se_math import (
        CHOLMOD_CHOLESKY,
        SEScaleProfile,
        build_normal_equations,
        measurement_leverage,
        solve_woodbury_after_row_removal,
        update_leverage_after_batch_removal,
    )

    p50k = SEScaleProfile.for_problem(120000, 360000, 1900000, multi_solve=True)
    assert p50k.frozen_max_steps >= 5 and p50k.local_warm_frozen_max_steps >= 4
    assert p50k.local_warm_frozen_max_dx >= 0.60 and p50k.woodbury_max_steps >= 6

    if CHOLMOD_CHOLESKY is None:
        pytest.skip("CHOLMOD unavailable")

    rng = np.random.default_rng(2050)
    n_s = 120
    rows_l, cols_l, vals_l = [], [], []
    rid = 0
    for i in range(n_s):
        rows_l.append(rid); cols_l.append(i); vals_l.append(1.0); rid += 1
    for _ in range(360):
        a, b = rng.choice(n_s, size=2, replace=False)
        rows_l.extend([rid, rid]); cols_l.extend([int(a), int(b)])
        vals_l.extend([float(rng.uniform(0.5, 1.5)), float(rng.uniform(-1.5, -0.5))])
        rid += 1
    H = csr_matrix((vals_l, (rows_l, cols_l)), shape=(rid, n_s), dtype=np.float64)
    w = rng.uniform(50.0, 200.0, size=rid)
    G, _ = build_normal_equations(H, np.zeros(rid), w, dense_gain_limit=0)
    fac = CHOLMOD_CHOLESKY(G.tocsc())
    lev0 = measurement_leverage(H, np.linalg.inv(G.toarray()))

    col_counts = []

    class _CountingFactor:
        def __call__(self, rhs):
            arr = np.asarray(rhs)
            col_counts.append(1 if arr.ndim == 1 else int(arr.shape[1]))
            return fac(rhs)

    cfac = _CountingFactor()
    cache = {}
    rhs1 = rng.normal(size=n_s)
    rem_seq = [130, 145, 170, 210]

    # 第 1 次移除 [130, 145]：需解 rhs + 2 列 = 3 列
    dx1, wb1 = solve_woodbury_after_row_removal(H, w, rem_seq[:2], rhs1, cfac, return_solver=True, u_col_cache=cache)
    assert dx1 is not None and wb1 is not None
    assert col_counts[-1] == 3

    # 第 2 次累计移除 [130, 145, 170]：仅新增 1 列，故只解 rhs + 1 列 = 2 列（复用 130, 145）
    dx2, wb2 = solve_woodbury_after_row_removal(H, w, rem_seq[:3], rhs1, cfac, return_solver=True, u_col_cache=cache)
    assert dx2 is not None and wb2 is not None
    assert col_counts[-1] == 2

    # 同锚点利用缓存传播杠杆：3 列全命中，底层因子回代次数增加 0！
    n_calls_before_lev = len(col_counts)
    lev_cached = update_leverage_after_batch_removal(
        lev0, H, w, rem_seq[:3], cfac, keep_shape=True, factor_is_post_removal=False, u_col_cache=cache
    )
    assert len(col_counts) == n_calls_before_lev
    assert lev_cached is not None

    # 与直接重算 G_new^-1 精确对比
    w_new = w.copy()
    w_new[rem_seq[:3]] = 0.0
    G_new, _ = build_normal_equations(H, np.zeros(rid), w_new, dense_gain_limit=0)
    dx_exact = np.linalg.solve(G_new.toarray(), rhs1)
    lev_exact = measurement_leverage(H, np.linalg.inv(G_new.toarray()))
    lev_exact[rem_seq[:3]] = 0.0
    assert np.allclose(dx2, dx_exact, rtol=1e-11, atol=1e-11)
    assert np.allclose(lev_cached, lev_exact, rtol=1e-11, atol=1e-11)

    # 切换新因子实例时缓存应自动失效重建（不串缓存）
    cfac2 = _CountingFactor()
    dx3 = solve_woodbury_after_row_removal(H, w, rem_seq[:3], rhs1, cfac2, u_col_cache=cache)
    assert col_counts[-1] == 4
    assert np.allclose(dx3, dx_exact, rtol=1e-11, atol=1e-11)




