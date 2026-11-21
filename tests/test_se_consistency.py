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
        "jacobian_sparse",
        "measurement_device_counts",
        "measurement_device_names",
        "state_layout",
        "print_state",
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
        est.prepare()
        se_res = est.run(result_mode=mode, skip_bad_data=False, verbose=False)
        assert est.estimate_result is not None and est.estimate_result.converged
        assert len(est.bad_items) >= 1
        assert est.normalized_residual.size == est.estimate_result.residual.size
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
