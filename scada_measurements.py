"""Explicit point-catalog to SE measurement mapping (never uses seed/readback values)."""

import json
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from sqlite_io import connect_readonly


def measurement_type(table, metric):
    if table in {'ACNode', 'DCNode', 'ACRealBs', 'DCRealBs'}:
        return {'uab': 'V', 'u': 'V'}.get(metric)
    if table in {'ACGenerator', 'DCGenerator', 'ACLoad', 'DCLoad'}:
        suffix = 'GEN' if table.endswith('Generator') else 'LOAD'
        field = {'p': 'P', 'q': 'Q', 'uab': 'V', 'u': 'V', 'ia': 'I', 'i': 'I'}.get(metric)
        return field + '_' + suffix if field else None
    if table == 'DCACConverter':
        return {'ac_p': 'P_AC', 'ac_q': 'Q_AC', 'ac_uab': 'V_AC', 'ac_ia': 'I_AC',
                'dc_p': 'P_DC', 'dc_u': 'V_DC', 'dc_i': 'I_DC'}.get(metric)
    if table in {'ACBranch', 'DCBranch', 'ACZeroBranch', 'DCZeroBranch',
                 'ACTransformer', 'ACACConverter', 'DCDCConverter'}:
        side, _, metric = metric.partition('_')
        field = {'p': 'P', 'q': 'Q', 'uab': 'V', 'u': 'V', 'ia': 'I', 'i': 'I'}.get(metric)
        return field + ('_FROM' if side == 'i' else '_TO') if field and side in {'i', 'j'} else None
    return None


def unit_scale(unit, kind, model):
    dimension = kind[0]
    scales = {
        'P': {'W': 1., 'kW': 1e3, 'MW': 1e6},
        'Q': {'var': 1., 'kvar': 1e3, 'Mvar': 1e6},
        'V': {'V': 1., 'kV': 1e3},
        'I': {'A': 1., 'kA': 1e3},
    }
    target = model[{'P': 'p_unit', 'Q': 'p_unit', 'V': 'u_unit', 'I': 'i_unit'}[dimension]]
    if dimension == 'Q':
        target = target.replace('W', 'var')
    try:
        return scales[dimension][unit] / scales[dimension][target]
    except KeyError as exc:
        raise ValueError(f'Unsupported {kind} unit conversion: {unit} -> {target}') from exc


def build_scada_measurements(model_blocks, database, catalog, *, max_age=120., now=None):
    """One SCADA read transaction. Return kernel rows plus an audit of every YC mapping."""
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError('max_age must be positive seconds')
    now = time.time() if now is None else now
    points = json.loads(Path(catalog).read_text(encoding='utf-8-sig'))['points']
    model_rows = {name: [dict(zip(b['header_list'], r)) for r in b['rows']]
                  for name, b in model_blocks.items()}
    units = model_rows['Model'][0]
    devices = {(t, int(r['idx'])): r for t, rows in model_rows.items() for r in rows if 'idx' in r}
    with closing(connect_readonly(database)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN')
        values = {(int(r['rtu_id']), int(r['pnt_no'])): dict(r)
                  for r in conn.execute('SELECT * FROM fes_yc_define')}
    measurements, audit, seen = [], [], set()
    for point in points:
        if point.get('kind') != 'yc':
            continue
        key = (int(point['rtu_id']), int(point['pnt_no']))
        if key in seen:
            raise ValueError(f'Duplicate YC mapping: {key}')
        seen.add(key)
        table = point['model_table']
        device = devices.get((table, int(point['model_idx'])))
        kind = point.get('meas_type') or measurement_type(table, point['metric'])
        source = values.get(key)
        row = {**{k: point.get(k) for k in ('rtu_id', 'pnt_no', 'dev_id', 'model_table', 'model_idx',
                                           'device_name', 'metric', 'unit')},
               'meas_type': kind, 'accepted': False, 'reason': '',
               'value': None if source is None else source.get('value'),
               'quality': None if source is None else source.get('status'),
               'refresh_time': None if source is None else source.get('refresh_time')}
        audit.append(row)
        if not kind:
            row['reason'] = 'unsupported_metric'
            continue
        if device is None or str(device.get('name')) != str(point['device_name']):
            row['reason'] = 'model_mapping_mismatch'
            continue
        if source is None:
            row['reason'] = 'missing_point'
            continue
        if source.get('dev_id') is not None and int(source['dev_id']) != int(point['dev_id']):
            row['reason'] = 'scada_device_mismatch'
            continue
        try:
            value = float(source['value'])
            quality = int(source['status'])
            stamp = float(source['refresh_time']) / 1000.  # SCADA contract: Unix milliseconds
            if quality != 0 or not math.isfinite(value):
                row['reason'] = 'invalid_quality_or_value'
                continue
            if not math.isfinite(stamp) or stamp <= 0 or now - stamp > max_age or stamp > now + 5:
                row['reason'] = 'stale_or_missing_timestamp'
                continue
            scale = unit_scale(point['unit'], kind, units)
            weight = float(point.get('weight', 1.0)) / scale ** 2
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError('weight must be positive')
        except (ValueError, TypeError, KeyError) as exc:
            row['reason'] = str(exc)
            continue
        if table in {'ACRealBs', 'DCRealBs'}:
            table = table[:2] + 'Node'
            device = devices.get((table, int(device['node'])))
            if device is None:
                row['reason'] = 'missing_bus_node'
                continue
        from model.meas_type import DEVICE_TYPE_CODES, MEAS_TYPE_CODES
        if table not in DEVICE_TYPE_CODES or kind not in MEAS_TYPE_CODES:
            row['reason'] = 'unsupported_measurement_type'
            continue
        row.update(accepted=True, converted_value=value * scale, weight=weight)
        measurements.append([len(measurements)+1, f'yc_{key[0]}_{key[1]}', table,
                             device['name'], kind, weight, 1, value * scale])
    return {'Measurement': {'table_name': 'Measurement', 'lv': 0,
                            'header_list': ['idx', 'name', 'dev_type', 'dev_name', 'meas_type',
                                            'weight', 'valid', 'value'], 'rows': measurements}}, audit
