"""Atomic, append-only calculation runs and queryable result records."""

import dataclasses
import json
import math
import sqlite3
from contextlib import closing


def json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    if dataclasses.is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, '__dict__'):
        return json_value(vars(value))
    raise TypeError(f'Unsupported result value: {type(value)}')


def json_text(value):
    return json.dumps(json_value(value), ensure_ascii=False, allow_nan=False)


def store_run(path, task, metadata, summary, records):
    """Commit status and ALL details together. Never replace a previous result."""
    if task not in {'se', 'lf', 'ca'}:
        raise ValueError(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=30)) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA foreign_keys=ON')
        with conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('''CREATE TABLE IF NOT EXISTS analysis_run (
                run_id INTEGER PRIMARY KEY, task TEXT NOT NULL, scenario TEXT NOT NULL,
                runtime TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT NOT NULL,
                status TEXT NOT NULL, model_file TEXT NOT NULL, input_sha256 TEXT,
                message TEXT NOT NULL, summary_json TEXT NOT NULL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS analysis_row (
                run_id INTEGER NOT NULL REFERENCES analysis_run(run_id),
                category TEXT NOT NULL, row_key TEXT NOT NULL, data_json TEXT NOT NULL,
                PRIMARY KEY (run_id, category, row_key))''')
            conn.execute('CREATE VIEW IF NOT EXISTS latest_run AS SELECT * FROM analysis_run '
                         'WHERE run_id=(SELECT MAX(run_id) FROM analysis_run)')
            conn.execute('CREATE VIEW IF NOT EXISTS ' + task + '_result AS SELECT r.* FROM analysis_row r '
                         'JOIN latest_run s ON r.run_id=s.run_id')
            run_id = conn.execute('''INSERT INTO analysis_run
                (task,scenario,runtime,started_at,finished_at,status,model_file,input_sha256,message,summary_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)''', (task, *[metadata[k] for k in (
                    'scenario', 'runtime', 'started_at', 'finished_at', 'status', 'model_file',
                    'input_sha256', 'message')], json_text(summary))).lastrowid
            conn.executemany('INSERT INTO analysis_row VALUES (?,?,?,?)',
                             ((run_id, category, str(key), json_text(data)) for category, key, data in records))
    return run_id
