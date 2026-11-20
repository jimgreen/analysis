"""Run SE/LF/CA in a parameterized workspace and persist each completed attempt."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile

from algorithm_parameters import PowerFlowParameters, StateEstimationParameters
from result_sqlite import json_text, store_run
from sqlite_io import read_sqlite_rows, snapshot_database, write_row_database
from workspace import Workspace


# Same numerical defaults as the original project's lf.para and se.para.
LF_PARAMETERS = PowerFlowParameters(1e-8, 50, 0.01, 1e6)
SE_PARAMETERS = StateEstimationParameters(1e-8, 20, 1e-6, False, 3., 10, 1e-4,
                                         2000, 0., 10, 0.05, 0.01, 1e-8, 200, 0.01)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def run_lf(model, records):
    from lfcore.hybrid_lf import HybridPowerFlowCalc, _read_lf_network_from_file
    calc = HybridPowerFlowCalc(_read_lf_network_from_file(model), parameters=LF_PARAMETERS,
                               verbose=False, result_mode='full')
    rc = calc.run()
    result = calc.lf_result
    summary = {'rc': rc, 'converged': bool(result and result.converged),
               'iterations': calc.iterations, 'norm_f': calc.normF,
               'failure_reason': getattr(calc, 'failure_reason', ''),
               'units': 'kernel result units (electrical per-unit; fluid native units)'}
    if result:
        for domain in ('ac', 'dc', 'dcac'):
            part = getattr(result, domain, None)
            for category in ('nodes', 'branches', 'transformers', 'three_winding_transformers',
                             'generators', 'loads', 'breakers', 'zero_branches',
                             'acac_converters', 'dcdc_converters', 'dcac_converters'):
                for key, row in (getattr(part, category, {}) or {}).items():
                    records.append((domain + '.' + category, key, row))
        summary['errors'] = {'ac': result.ac_errors, 'dc': result.dc_errors, 'fluid': result.fluid_errors}
        for name, fluid in calc.fluid_calcs.items():
            # Result arrays have named fields and contain no network/calculator cycles.
            records.append(('fluid', name, fluid.lf_result))
        records.append(('arrays', 'hybrid', calc.result))
    return ('success' if summary['converged'] else 'not_converged'), summary, ''


def run_se(model, workspace, temporary, records, *, scada=None, catalog=None, max_age=120.):
    from scada_measurements import build_scada_measurements
    from secore.hybrid_se import HybridStateEstimator
    blocks = read_sqlite_rows(model)
    scada = Path(scada) if scada else workspace.scada
    catalog = Path(catalog) if catalog else workspace.point_catalog
    measurements, audit = build_scada_measurements(blocks, scada, catalog, max_age=max_age)
    records.extend(('scada_mapping', f"{r['rtu_id']}:{r['pnt_no']}", r) for r in audit)
    summary = {'scada_file': str(scada), 'point_catalog': str(catalog), 'max_age_seconds': max_age,
               'point_count': len(audit), 'accepted_measurements': len(measurements['Measurement']['rows']),
               'rejected_reasons': dict(Counter(r['reason'] for r in audit if not r['accepted']))}
    if not summary['accepted_measurements']:
        return 'input_unavailable', summary, 'SCADA has no valid, fresh, mapped measurements'
    meas_file = temporary / 'measurements.sqlite3'
    write_row_database(meas_file, measurements)
    estimator = HybridStateEstimator(model, meas_file, parameters=SE_PARAMETERS,
                                     power_flow_linear_solver='superlu', auto_prepare=False)
    estimator.prepare()
    se = estimator.run(result_mode='full', verbose=False)
    multi = estimator.multi_energy_result
    summary.update(converged=multi.converged, observable=multi.observable,
                   errors=multi.errors, warnings=multi.warnings)
    for domain, part in [('electric', se), *multi.fluids.items()]:
        if part is None:
            continue
        records.append(('statistics', domain, part.statistics))
        for category in ('normal_measurements', 'pseudo_measurements', 'prefiltered_measurements', 'bad_data'):
            for i, row in enumerate(getattr(part, category)):
                records.append((domain + '.' + category, i, row))
    estimate = estimator.multi_energy_estimate_result or estimator.estimate_result
    if estimate is not None:
        records.append(('state', 'vector', {'values': estimate.x, 'units': 'kernel state units'}))
    status = 'success' if multi.converged and multi.observable else 'not_converged'
    return status, summary, ''


def run_ca(model, temporary, records, *, limit=None, scan_n1=True):
    from cacore.hybrid_ca import run_hybrid_ca
    from efile_read import EBook
    if limit is None:
        # Never invent security thresholds. N-1 and base convergence remain available.
        limit = temporary / 'no_limits.e'
        limit.write_text('', encoding='utf8')
        limited = True
    else:
        limited = False
    info = temporary / 'ca_info.e'
    n0, n1 = run_hybrid_ca(model, limit, temporary / 'n0.e', temporary / 'n1.e', info,
                           scan_n1=scan_n1, parameters=LF_PARAMETERS)
    records.extend(('n0_risk', i, row) for i, row in enumerate(n0))
    records.extend(('n1_risk', i, row) for i, row in enumerate(n1))
    summary = dict(EBook(info).data['ca_calc_info'].data[-1])
    summary.update(scan_n1=scan_n1, limits_supplied=not limited, limit_file=None if limited else str(limit))
    base_failed = any(row.risk_type == '潮流发散' for row in n0)
    status = 'not_converged' if base_failed else 'partial' if limited else 'success'
    message = 'No security limit file supplied; only convergence and N-1 load-loss checks performed' if limited else ''
    return status, summary, message


def run_workspace(workspace=None, tasks=('se', 'lf', 'ca'), *, scada=None, catalog=None,
                  max_age=120., limit=None, scan_n1=True):
    """Read a new SQLite snapshot per invocation; no global or persisted network cache."""
    workspace = workspace or Workspace()
    tasks = tuple(tasks)
    if not tasks or any(t not in {'se', 'lf', 'ca'} for t in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('tasks must be unique SE/LF/CA names')
    outcomes = []
    # One consistent model snapshot shared by the requested algorithms within this invocation only.
    with tempfile.TemporaryDirectory(prefix='analysis_run_') as directory:
        temporary = Path(directory)
        model = temporary / 'model.sqlite3'
        snapshot_error = None
        digest = None
        try:
            snapshot_database(workspace.model, model)
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
        except Exception as exc:
            snapshot_error = exc
        for task in tasks:
            metadata = {'scenario': workspace.scenario, 'runtime': workspace.runtime,
                        'started_at': utc_now(), 'model_file': str(workspace.model), 'input_sha256': digest}
            records, summary = [], {}
            try:
                if snapshot_error:
                    raise snapshot_error
                blocks = read_sqlite_rows(model)
                records.append(('input_model', 'metadata', {
                    'tables': {name: len(block['rows']) for name, block in blocks.items()},
                    'Model': blocks.get('Model', blocks.get('PowerBase')),
                }))
                if task == 'lf':
                    status, summary, message = run_lf(model, records)
                elif task == 'se':
                    status, summary, message = run_se(model, workspace, temporary, records,
                                                      scada=scada, catalog=catalog, max_age=max_age)
                else:
                    status, summary, message = run_ca(model, temporary, records, limit=limit, scan_n1=scan_n1)
            except Exception as exc:
                status, message = 'error', f'{type(exc).__name__}: {exc}'
            metadata.update(finished_at=utc_now(), status=status, message=message)
            output = workspace.result(task)
            run_id = store_run(output, task, metadata, summary, records)
            outcomes.append({'task': task, 'run_id': run_id, 'status': status, 'database': str(output),
                             'message': message, 'summary': summary})
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(r'D:\codex\workspace'))
    parser.add_argument('--runtime', '--state', default='realtime')
    parser.add_argument('--scenario', default='qinling')
    parser.add_argument('--task', choices=('all', 'se', 'lf', 'ca'), default='all')
    parser.add_argument('--scada', type=Path, help='Override sqldb/scada/scdfe_c.sqlite3')
    parser.add_argument('--point-map', type=Path, help='Override sqldb/scada/point_catalog.json')
    parser.add_argument('--max-age', type=float, default=120., help='Maximum measurement age in seconds')
    parser.add_argument('--limit', type=Path, help='CA limits in original .calmt format and physical units')
    parser.add_argument('--no-n1', action='store_true')
    args = parser.parse_args(argv)
    ws = Workspace(args.root, args.runtime, args.scenario)
    tasks = ('se', 'lf', 'ca') if args.task == 'all' else (args.task,)
    results = run_workspace(ws, tasks, scada=args.scada, catalog=args.point_map,
                            max_age=args.max_age, limit=args.limit, scan_n1=not args.no_n1)
    print(json_text({'work_directory': str(ws.directory), 'results': results}))
    return 0 if all(row['status'] == 'success' for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
