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
