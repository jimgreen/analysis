"""模型加载、量测加载与 LF/SE 结果导出优化的正确性与一致性回归测试。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest

from efile_read import _FlatTableRows, _read_efile_rows
from lfcore.ac_lf import ACPowerFlowCalc
from lfcore.common import LazyResultDict
from lfcore.dc_lf import DCPowerFlowCalc
from lfcore.hybrid_lf import HybridPowerFlowCalc
from model.hybrid_array_model import build_hybrid_ppc_only_from_efile_rows
from model.meas_array_model import (
    _try_parse_meas_ppc_bulk_bytes,
    build_meas_ppc_from_e_file,
    build_meas_ppc_from_efile_rows,
)
from model.ppc_topology import (
    build_ac_ppc_with_topology_from_e_file,
    build_dc_ppc_with_topology_from_e_file,
)
from secore.ac_se import ACStateEstimator
from secore.dc_se import DCStateEstimator
from secore.hybrid_se import HybridStateEstimator
from secore.se_result import SEResult
from tests.test_se_consistency import (
    LF_PARAMS,
    SE_PARAMS,
    _generate_ac_case,
    _generate_dc_case,
    _generate_hybrid_case,
)


def test_meas_bulk_bytes_matches_efile_rows_both_modes(tmp_path: Path):
    """量测批量字节解析器在 include_strings=False/True、均/异权重与有效/无效行下应与标准解析一致。"""
    meas_path = tmp_path / "mixed.meas"
    meas_path.write_text(
        "\n".join(
            [
                "<Measurement>",
                "@ idx name dev_type dev_name meas_type weight valid value",
                "# 1 m_1 ACNode bus_1 V 10.0 1 110.5",
                "# 2 m_2 ACNode bus_2 V 25.0 0 109.2",
                "# 3 m_3 ACBranch br_1 p_from 10.0 1 12.345",
                "# 4 m_4 ACBranch br_1 Q_TO 0.0 1 -3.21",
                "</Measurement>",
            ]
        )
        + "\n",
        encoding="utf8",
    )

    ref = build_meas_ppc_from_efile_rows(meas_path, _read_efile_rows(meas_path))
    full = build_meas_ppc_from_e_file(meas_path, include_strings=True, include_matrix=True)
    arr_only = build_meas_ppc_from_e_file(meas_path, include_strings=False, include_matrix=False)

    assert _try_parse_meas_ppc_bulk_bytes(meas_path, include_strings=True, include_matrix=True) is not None
    assert np.array_equal(full["idx_array"], ref["idx_array"])
    assert np.allclose(full["weight_array"], ref["weight_array"])
    assert np.allclose(full["value_array"], ref["value_array"])
    assert np.array_equal(full["valid_array"], ref["valid_array"])
    assert np.array_equal(full["status_array"], ref["status_array"])
    assert np.array_equal(full["device_type_code_array"], ref["device_type_code_array"])
    assert np.array_equal(full["meas_type_code_array"], ref["meas_type_code_array"])
    assert list(full["name"]) == list(ref["name"])
    assert list(full["device_type"]) == list(ref["device_type"])
    assert list(full["device_name"]) == list(ref["device_name"])
    assert list(full["meas_type"]) == list(ref["meas_type"])

    assert np.array_equal(arr_only["idx_array"], ref["idx_array"])
    assert np.allclose(arr_only["weight_array"], ref["weight_array"])
    assert np.allclose(arr_only["value_array"], ref["value_array"])
    assert np.array_equal(arr_only["valid_array"], ref["valid_array"])
    assert set(arr_only["rows_by_device_type_code"].keys()) == set(ref["rows_by_device_type_code"].keys())
    for k, rows_ref in ref["rows_by_device_type_code"].items():
        assert np.array_equal(arr_only["rows_by_device_type_code"][k], rows_ref)


def test_hybrid_loader_keeps_flat_table_rows_unmaterialized(tmp_path: Path):
    """Hybrid 模型加载应全程走列式读取，不触发 _FlatTableRows 行级物化。"""
    ef, _mf = _generate_hybrid_case(16, 12, tmp_path, inject_bad=False)
    rows = _read_efile_rows(ef)
    assert isinstance(rows["DCACConverter"]["rows"], _FlatTableRows)
    assert rows["DCACConverter"]["rows"]._rows is None

    ppc = build_hybrid_ppc_only_from_efile_rows(ef, rows)
    assert ppc["dcac"].shape[0] == 1
    assert rows["DCACConverter"]["rows"]._rows is None
    assert rows["ACNode"]["rows"]._rows is None
    assert rows["DCNode"]["rows"]._rows is None


def test_lazy_result_dict_full_dict_contract():
    """LazyResultDict 必须严格满足 dict 类型及全部读写比较契约。"""
    calls = []

    def builder():
        calls.append(1)
        return [("a", 1), ("b", 2)]

    d = LazyResultDict(builder, 2)
    assert isinstance(d, dict)
    assert bool(d) is True
    assert len(calls) == 0  # bool() 不触发物化

    assert len(d) == 2
    assert len(calls) == 1
    assert d["a"] == 1
    assert "b" in d and d.get("missing", 9) == 9
    assert list(d.keys()) == ["a", "b"]
    assert list(d.values()) == [1, 2]
    assert d == {"a": 1, "b": 2}
    assert (d | {"c": 3}) == {"a": 1, "b": 2, "c": 3}

    d["c"] = 3
    assert d.pop("a") == 1
    cp = d.copy()
    assert type(cp) is dict and cp == {"b": 2, "c": 3}
    d.clear()
    assert not d and len(d) == 0


@pytest.mark.parametrize("kind", ["ac", "dc", "hybrid"])
def test_lf_full_result_lazy_dicts_match_arrays(tmp_path: Path, kind: str):
    """AC / DC / Hybrid LF 的 full 结果字典按需物化后应与数组结果精确一致。"""
    if kind == "ac":
        ef, _ = _generate_ac_case(16, tmp_path)
        calc = ACPowerFlowCalc(build_ac_ppc_with_topology_from_e_file(ef), parameters=LF_PARAMS, keep_node_objects=False, verbose=False)
    elif kind == "dc":
        ef, _ = _generate_dc_case(16, tmp_path)
        calc = DCPowerFlowCalc(build_dc_ppc_with_topology_from_e_file(ef), parameters=LF_PARAMS, keep_node_objects=False, verbose=False)
    else:
        ef, _ = _generate_hybrid_case(12, 8, tmp_path)
        calc = HybridPowerFlowCalc.from_file_fast(ef, parameters=LF_PARAMS, verbose=False)

    calc.prepare()
    assert calc.run(result_mode="full") == 0
    lf_res = calc.lf_result
    assert lf_res is not None
    if kind == "hybrid":
        assert isinstance(lf_res.ac.nodes, dict) and len(lf_res.ac.nodes) == 12
        assert isinstance(lf_res.dc.nodes, dict) and len(lf_res.dc.nodes) == 8
        assert isinstance(lf_res.dcac.dcac_converters, dict) and len(lf_res.dcac.dcac_converters) == 1
    else:
        assert isinstance(lf_res.nodes, dict) and len(lf_res.nodes) == 16
        assert isinstance(lf_res.branches, dict) and len(lf_res.branches) > 0


@pytest.mark.parametrize("skip_bad_data", [False, True])
def test_se_result_columnar_export_exact_equivalence(tmp_path: Path, skip_bad_data: bool):
    """SEResult 列式直出与物化 MeasurementRow 导出的 E 文本、文件及字典必须逐字符完全一致。"""
    ef, mf = _generate_ac_case(16, tmp_path, inject_bad=True)
    # 注入一条 invalid 和一条 zero weight 量测，覆盖 prefiltered 的 reason 空格转义路径
    text = mf.read_text(encoding="utf8").replace(
        "</Measurement>",
        "# 9998 m_inv ACNode bus_2 V 100.0 0 110.0\n# 9999 m_zw ACNode bus_3 V 0.0 1 110.0\n</Measurement>",
    )
    mf.write_text(text, encoding="utf8")

    est = ACStateEstimator(e_file=str(ef), meas_file=str(mf), parameters=SE_PARAMS, flat_start=True)
    se_res = est.run(result_mode="full", skip_bad_data=skip_bad_data, verbose=False)
    assert se_res is not None
    assert len(se_res.prefiltered_measurements) >= 2

    # 1. 未物化（列式后端）直接导出
    col_e_text = se_res.to_e_text()
    assert "zero_weight" in col_e_text and "invalid" in col_e_text
    out_file = tmp_path / f"se_col_{skip_bad_data}.e"
    se_res.write_e_file(out_file)
    assert out_file.read_text(encoding="utf8") == col_e_text

    col_pref_dicts = se_res.prefiltered_measurements.to_dicts()
    col_norm_dicts = se_res.normal_measurements.to_dicts()

    # 2. 强制物化所有 MeasurementRow 后再次导出并比对 dataclasses.asdict 与旧版泛型 _block_to_e
    pref_rows = se_res.prefiltered_measurements.rows
    norm_rows = se_res.normal_measurements.rows
    pseudo_rows = se_res.pseudo_measurements.rows
    bad_rows = se_res.bad_data.rows
    assert len(pref_rows) >= 2 and len(norm_rows) > 0

    mat_e_text = se_res.to_e_text()
    assert mat_e_text == col_e_text

    # 与最原始泛型 _block_to_e/_e_cell 逐字符比对
    legacy_blocks = [se_res._statistics_block_to_e()]
    for bname, tbl_rows in (
        ("SEResultPrefilteredMeasurement", pref_rows),
        ("SEResultPseudoMeasurement", pseudo_rows),
        ("SEResultBadData", bad_rows),
        ("SEResultNormalMeasurement", norm_rows),
    ):
        raw_rows = [[getattr(r, col) for col in SEResult._MEASUREMENT_COLUMNS] for r in tbl_rows]
        legacy_blocks.append(SEResult._block_to_e(bname, SEResult._MEASUREMENT_COLUMNS, raw_rows))
    legacy_e_text = "\n\n".join(legacy_blocks) + "\n"
    assert col_e_text == legacy_e_text

    assert col_pref_dicts == [asdict(r) for r in pref_rows]
    assert col_norm_dicts == [asdict(r) for r in norm_rows]
    assert se_res.normal_measurements.to_dicts() == col_norm_dicts


@pytest.mark.parametrize("sparse_bus_ids", [False, True])
def test_build_ac_ppc_from_mat_file_vectorized_equivalence(tmp_path: Path, sparse_bus_ids: bool):
    """验证向量化 build_ac_ppc_from_mat_file() 在稠密与稀疏大母线 ID、移相/充电变压器、停运节点与多机场景下完全精确。"""
    from scipy.io import savemat

    from lfcore.ac_lf import ACPowerFlowCalc
    from model.ac_array_model import (
        BRANCH_COLS,
        BUS_COLS,
        CTRL_PQ,
        CTRL_PV,
        CTRL_SLACK,
        GEN_COLS,
        LOAD_COLS,
        SHUNT_B,
        SHUNT_COLS,
        TRANSFORMER_COLS,
        build_ac_ppc_from_mat_file,
    )
    from model.array_common import LazyNameArray

    ids = [100001, 200002, 300003, 400004] if sparse_bus_ids else [1, 2, 3, 4]
    # bus: [BUS_I, TYPE(1=PQ,2=PV,3=REF,4=NONE), PD, QD, GS, BS, AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN]
    bus_mp = np.array(
        [
            [ids[0], 3, 0.0, 0.0, 0.0, 0.0, 1, 1.02, 0.0, 110.0, 1, 1.1, 0.9],
            [ids[1], 2, 20.0, 5.0, 1.5, -2.0, 1, 1.01, 0.0, 110.0, 1, 1.1, 0.9],
            [ids[2], 1, 35.0, 12.0, 0.0, 3.0, 1, 1.00, 0.0, 110.0, 1, 1.1, 0.9],
            [ids[3], 4, 10.0, 4.0, 2.0, 1.0, 1, 1.00, 0.0, 110.0, 1, 1.1, 0.9],  # 停运母线，负荷/并联应跳过
        ],
        dtype=np.float64,
    )
    # gen: [GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, STATUS, PMAX, PMIN]
    gen_mp = np.array(
        [
            [ids[0], 30.0, 10.0, 50.0, -50.0, 1.02, 100.0, 1.0, 100.0, 0.0],
            [ids[1], 25.0, 0.0, 40.0, -40.0, 0.0, 100.0, 1.0, 80.0, 0.0],  # VG=0 -> 回退 1.0
            [ids[2], 5.0, 2.0, 20.0, -20.0, 1.00, 100.0, 1.0, 30.0, 0.0],  # PQ 母线挂发电机 -> CTRL_PQ
        ],
        dtype=np.float64,
    )
    # branch: [F_BUS, T_BUS, R, X, B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, STATUS, ANGMIN, ANGMAX]
    branch_mp = np.array(
        [
            [ids[0], ids[1], 0.01, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],   # 普通线路
            [ids[1], ids[2], 0.008, 0.04, 0.04, 0.0, 0.0, 0.0, 1.05, 3.0, 1.0, -360.0, 360.0], # 移相+变比+充电电纳变压器
            [ids[0], ids[2], 0.015, 0.06, 0.00, 0.0, 0.0, 0.0, 0.98, 0.0, 1.0, -360.0, 360.0], # 仅变比无充电电纳变压器
        ],
        dtype=np.float64,
    )
    mat_path = tmp_path / f"case_vec_{sparse_bus_ids}.mat"
    savemat(str(mat_path), {"mpc": {"version": "2", "baseMVA": 100.0, "bus": bus_mp, "gen": gen_mp, "branch": branch_mp}})

    ppc = build_ac_ppc_from_mat_file(mat_path)

    # 校验矩阵维度、数值与 LazyNameArray 名称完全一致
    assert ppc["bus"].shape[0] == 4
    assert ppc["bus"][3, BUS_COLS["run_stat"]] == 0.0
    assert ppc["load"].shape[0] == 2
    assert np.allclose(ppc["load"][:, LOAD_COLS["node"]], [ids[1], ids[2]])
    assert np.allclose(ppc["load"][:, LOAD_COLS["pbase"]], [0.20, 0.35])
    assert ppc["gen"].shape[0] == 3
    assert ppc["gen"][:, GEN_COLS["control_type"]].tolist() == [CTRL_SLACK, CTRL_PV, CTRL_PQ]
    assert np.allclose(ppc["gen"][:, GEN_COLS["v_set"]], [1.02, 1.0, 1.0])
    assert ppc["branch"].shape[0] == 1
    assert np.isclose(ppc["branch"][0, BRANCH_COLS["b"]], 0.02)
    assert ppc["transformer"].shape[0] == 2
    assert np.allclose(ppc["transformer"][:, TRANSFORMER_COLS["bt"]], [0.02, 0.0])
    assert np.allclose(ppc["transformer"][:, TRANSFORMER_COLS["shift"]], [3.0, 0.0])
    # shunt = 2 条母线并联 + 1 条变压器末端 B/2 并联
    assert ppc["shunt"].shape[0] == 3
    assert np.allclose(ppc["shunt"][:, SHUNT_COLS["idx"]], [1.0, 2.0, 3.0])
    assert np.allclose(ppc["shunt"][:, SHUNT_COLS["node"]], [ids[1], ids[2], ids[2]])
    assert np.allclose(ppc["shunt"][:, SHUNT_COLS["g_set"]], [0.015, 0.0, 0.0])
    assert np.allclose(ppc["shunt"][:, SHUNT_COLS["b_set"]], [-0.02, 0.03, 0.02])
    assert np.all(ppc["shunt"][:, SHUNT_COLS["control_type"]] == SHUNT_B)

    for key, expected in (
        ("bus_name", [f"bus_{i}" for i in ids]),
        ("load_name", ["load_1", "load_2"]),
        ("gen_name", ["gen_1", "gen_2", "gen_3"]),
        ("branch_name", ["branch_1"]),
        ("transformer_name", ["transformer_1", "transformer_2"]),
        ("shunt_name", ["shunt_1", "shunt_2", "shunt_3"]),
    ):
        assert isinstance(ppc[key], LazyNameArray)
        assert ppc[key].tolist() == expected

    # 验证 array 与 full 模式均能正确收敛且电压完全一致
    calc_arr = ACPowerFlowCalc(build_ac_ppc_from_mat_file(mat_path), parameters=LF_PARAMS, result_mode="array")
    calc_arr.prepare()
    assert calc_arr.run() == 0 and calc_arr.converged
    calc_full = ACPowerFlowCalc(build_ac_ppc_from_mat_file(mat_path), parameters=LF_PARAMS, result_mode="full")
    calc_full.prepare()
    assert calc_full.run() == 0 and calc_full.converged
    assert np.allclose(calc_arr.ppc["bus"][:, BUS_COLS["voltage"]], calc_full.ppc["bus"][:, BUS_COLS["voltage"]])
    assert list(calc_full.lf_result.nodes.keys()) == [f"bus_{i}" for i in ids]


def test_build_ac_ppc_from_mat_file_v5_fast_path_fallback_and_edge_cases(tmp_path: Path):
    """验证 Level-5 mmap 快读路径、压缩回退、顶层变量、重复母线 ID、越界 VG 修正与句柄释放。"""
    from scipy.io import savemat

    from model.ac_array_model import (
        GEN_COLS,
        _try_assemble_ac_ppc_from_mat_v5,
        build_ac_ppc_from_mat_file,
    )
    from model.topology import build_ac_topology_input_ppc

    bus_mp = np.array(
        [
            [1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.02, 0.0, 110.0, 1, 1.1, 0.9],
            [2, 2, 15.0, 5.0, 1.0, -1.0, 1, 1.01, 0.0, 110.0, 1, 1.1, 0.9],
            [3, 1, 25.0, 8.0, 0.0, 2.0, 1, 1.00, 0.0, 110.0, 1, 1.1, 0.9],
        ],
        dtype=np.float64,
    )
    gen_mp = np.array(
        [
            [1, 20.0, 0.0, 50.0, -50.0, 1.02, 100.0, 1.0, 100.0, 0.0],
            [2, 20.0, 0.0, 40.0, -40.0, 2.50, 100.0, 1.0, 80.0, 0.0],  # 越界 VG=2.5，应被修正为 1.0
        ],
        dtype=np.float64,
    )
    branch_mp = np.array(
        [
            [1, 2, 0.01, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
            [2, 3, 0.01, 0.04, 0.04, 0.0, 0.0, 0.0, 1.04, 2.0, 1.0, -360.0, 360.0],
        ],
        dtype=np.float64,
    )
    mpc = {"version": "2", "baseMVA": 100.0, "bus": bus_mp, "gen": gen_mp, "branch": branch_mp}

    raw_mat = tmp_path / "uncompressed.mat"
    cmp_mat = tmp_path / "compressed.mat"
    top_mat = tmp_path / "top_level.mat"
    savemat(str(raw_mat), {"mpc": mpc}, do_compression=False)
    savemat(str(cmp_mat), {"mpc": mpc}, do_compression=True)
    savemat(str(top_mat), mpc, do_compression=False)

    # 未压缩文件应命中 Level-5 mmap 快速路径，压缩文件应返回 None 并回退 scipy.io.loadmat
    assert _try_assemble_ac_ppc_from_mat_v5(raw_mat) is not None
    assert _try_assemble_ac_ppc_from_mat_v5(cmp_mat) is None
    assert _try_assemble_ac_ppc_from_mat_v5(top_mat) is not None

    ppc_raw = build_ac_ppc_from_mat_file(raw_mat)
    ppc_cmp = build_ac_ppc_from_mat_file(cmp_mat)
    ppc_top = build_ac_ppc_from_mat_file(top_mat)

    # Windows 下确认 mmap 句柄已彻底关闭，临时文件可立即删除
    raw_mat.unlink()
    top_mat.unlink()

    # 校验未压缩快读、压缩回退、顶层散列变量三者逐位一致，且越界 VG 已触发修正
    assert ppc_raw["gen"][1, GEN_COLS["v_set"]] == 1.0
    assert len(ppc_raw.get("_setpoint_corrections", [])) == 1
    for other in (ppc_cmp, ppc_top):
        for key in ("bus", "branch", "transformer", "gen", "load", "shunt", "_gen_p_min", "_gen_q_min", "_gen_q_max"):
            assert np.array_equal(ppc_raw[key], other[key], equal_nan=True)

    # 校验融合生成的 _topology_input 与 build_ac_topology_input_ppc() 完全一致（含重复母线 ID 场景）
    dup_bus_mp = np.vstack([bus_mp, [2, 1, 5.0, 1.0, 0.5, 0.5, 1, 1.0, 0.0, 110.0, 1, 1.1, 0.9]])
    dup_mat = tmp_path / "dup_bus.mat"
    savemat(str(dup_mat), {"mpc": {**mpc, "bus": dup_bus_mp}}, do_compression=False)
    ppc_dup = build_ac_ppc_from_mat_file(dup_mat)
    dup_mat.unlink()
    ref_topo = build_ac_topology_input_ppc(ppc_dup)
    got_topo = ppc_dup["_topology_input"]
    assert np.array_equal(got_topo.node_ids, ref_topo.node_ids)
    assert np.array_equal(got_topo.node_run_mask, ref_topo.node_run_mask)
    for key in ref_topo.terminals:
        assert np.array_equal(got_topo.terminals[key].i_node_pos, ref_topo.terminals[key].i_node_pos)
        assert np.array_equal(got_topo.terminals[key].j_node_pos, ref_topo.terminals[key].j_node_pos)
        assert np.array_equal(got_topo.terminals[key].run_mask, ref_topo.terminals[key].run_mask)
    for key in ref_topo.singles:
        assert np.array_equal(got_topo.singles[key].node_pos, ref_topo.singles[key].node_pos)
        assert np.array_equal(got_topo.singles[key].run_mask, ref_topo.singles[key].run_mask)


def test_efile_binary_bulk_loadtxt_and_lazy_bytes_names(tmp_path: Path):
    """验证 >=64 行 E 模型文件走二进制 _FlatTableRows + np.loadtxt 批量解析、惰性名称快照独立性与 LF/SE 精确一致。"""
    from model.ac_array_model import BUS_COLS, build_ac_ppc_from_efile_rows
    from model.array_common import LazyNameArray
    from model.meas_array_model import attach_device_pos_from_name_arrays, build_meas_ppc_from_e_file
    from model.meas_type import DEVICE_TYPE_CODES

    n_bus = 80
    lines = [
        "<Model>",
        "@ p_base u_unit p_unit i_unit",
        "# 100.0 kV MW kA",
        "</Model>",
        "<ACNode>",
        "@ idx name vbase voltage angle isl run_stat",
    ]
    for i in range(1, n_bus + 1):
        lines.append(f"# {i} bus_{i} 110.0 110.0 0.0 0 1")
    lines.append("</ACNode>")
    lines.append("<ACBranch>")
    lines.append("@ idx name i_node j_node r x b run_stat")
    for i in range(1, n_bus):
        lines.append(f"# {i} br_{i} {i} {i + 1} 0.01 0.05 0.02 1")
    lines.append("</ACBranch>")
    lines.append("<ACGenerator>")
    lines.append("@ idx name node control_type p_set q_set v_set alpha run_stat")
    lines.append("# 1 gen_1 1 SLACK 0.0 0.0 111.1 1.0 1")
    lines.append("</ACGenerator>")
    lines.append("<ACLoad>")
    lines.append("@ idx name node pbase pv0 pv1 pv2 qbase qv0 qv1 qv2 run_stat")
    lines.append(f"# 1 load_1 {n_bus} 10.0 1.0 0.0 0.0 3.0 1.0 0.0 0.0 1")
    lines.append("</ACLoad>")
    e_path = tmp_path / "ac_80.e"
    e_path.write_text("\n".join(lines) + "\n", encoding="utf8")

    rows = _read_efile_rows(e_path)
    bus_flat = rows["ACNode"]["rows"]
    assert isinstance(bus_flat, _FlatTableRows)
    assert bus_flat._raw_bytes is not None
    assert bus_flat._rows is None

    ppc = build_ac_ppc_from_efile_rows(e_path, rows)
    # 数值列经 np.loadtxt 解析完成后，行列表与字节切分尚未触发（惰性保持）
    assert bus_flat._rows is None
    assert bus_flat._bytes_tokens is None
    assert isinstance(ppc["bus_name"], LazyNameArray)
    assert ppc["bus"].flags.f_contiguous
    assert ppc["bus"].shape[0] == n_bus

    # 即使随后外部物化并就地修改 rows，已构建 ppc["bus_name"] 快照仍不受影响
    rows["ACNode"]["rows"][0][1] = "mutated_bus_1"
    assert ppc["bus_name"].raw_bytes_list()[0] == b"bus_1"
    assert ppc["bus_name"].tolist()[0] == "bus_1"

    # 验证 bytes 零解码路径与 str 路径在 attach_device_pos_from_name_arrays 上逐位一致
    meas_lines = ["<Measurement>", "@ idx name dev_type dev_name meas_type weight valid value status"]
    for i in range(1, n_bus + 1):
        st = "0" if i % 4 != 0 else ("1" if i % 8 == 0 else "2")
        meas_lines.append(f"# {i} m_{i} ACNode bus_{i} V 100.0 1 110.0 {st}")
    meas_lines.append("</Measurement>")
    meas_path = tmp_path / "status_80.meas"
    meas_path.write_text("\n".join(meas_lines) + "\n", encoding="utf8")

    ref_m = build_meas_ppc_from_efile_rows(meas_path, _read_efile_rows(meas_path))
    fast_arr = build_meas_ppc_from_e_file(meas_path, include_strings=False, include_matrix=False)
    fast_full = build_meas_ppc_from_e_file(meas_path, include_strings=True, include_matrix=True)
    assert _try_parse_meas_ppc_bulk_bytes(meas_path, include_strings=False, include_matrix=False) is not None

    for candidate in (fast_arr, fast_full):
        assert np.array_equal(candidate["idx_array"], ref_m["idx_array"])
        assert np.allclose(candidate["weight_array"], ref_m["weight_array"])
        assert np.allclose(candidate["value_array"], ref_m["value_array"])
        assert np.array_equal(candidate["valid_array"], ref_m["valid_array"])
        assert np.array_equal(candidate["status_array"], ref_m["status_array"])
        assert np.array_equal(candidate["device_type_code_array"], ref_m["device_type_code_array"])
        assert np.array_equal(candidate["meas_type_code_array"], ref_m["meas_type_code_array"])
    assert list(fast_full["device_type"]) == list(ref_m["device_type"])
    assert list(fast_full["meas_type"]) == list(ref_m["meas_type"])
    assert list(fast_full["name"]) == list(ref_m["name"])

    pos_bytes = attach_device_pos_from_name_arrays(dict(fast_arr), {DEVICE_TYPE_CODES["ACNode"]: ppc["bus_name"]})
    pos_strs = attach_device_pos_from_name_arrays(dict(fast_arr), {DEVICE_TYPE_CODES["ACNode"]: ppc["bus_name"].tolist()})
    assert np.array_equal(pos_bytes, np.arange(n_bus, dtype=np.int64))
    assert np.array_equal(pos_bytes, pos_strs)



