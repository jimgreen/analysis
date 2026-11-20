"""Read ordinary and RTDB embedded-schema SQLite tables without modifying inputs."""

import sqlite3
from contextlib import closing
from pathlib import Path


def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def connect_readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)


def read_sqlite_rows(path):
    """Return the same typed row blocks as _read_efile_rows, in one transaction."""
    blocks = {}
    with closing(connect_readonly(path)) as conn, conn:
        conn.execute('BEGIN')
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        embedded = 'rtdb_schema_tables' in names
        tables = (list(conn.execute('SELECT table_name FROM rtdb_schema_tables ORDER BY ordinal'))
                  if embedded else [(name,) for name in sorted(names) if not name.startswith(('_', 'sqlite_'))])
        for (name,) in tables:
            physical = 'rtdb_data_' + name if embedded else name
            if embedded:
                columns = list(conn.execute(
                    'SELECT ordinal, eng_name FROM rtdb_schema_columns WHERE table_name=? ORDER BY ordinal',
                    (name,),
                ))
                header = [r[1] for r in columns]
                selection = ', '.join(quote_identifier('c' + str(r[0])) for r in columns)
                order = ' ORDER BY _pos'
            else:
                header = [r[1] for r in conn.execute('PRAGMA table_info(' + quote_identifier(name) + ')')]
                selection = ', '.join(map(quote_identifier, header))
                order = ''
            if not header:
                raise ValueError(f'No column metadata for {name}')
            rows = [list(row) for row in conn.execute(
                'SELECT ' + selection + ' FROM ' + quote_identifier(physical) + order
            )]
            blocks[name] = {'table_name': name, 'header_list': header, 'rows': rows, 'lv': 0}
    return blocks


def snapshot_database(source, destination):
    """SQLite online backup includes committed WAL contents; never a reused network cache."""
    with closing(connect_readonly(source)) as src, closing(sqlite3.connect(destination)) as dst:
        src.backup(dst)


def write_row_database(path, blocks):
    """Write a new private calculation snapshot with named columns."""
    with closing(sqlite3.connect(path)) as conn, conn:
        for name, block in blocks.items():
            header = block['header_list']
            if not header:
                continue
            table = quote_identifier(name)
            conn.execute('CREATE TABLE ' + table + ' (' + ','.join(map(quote_identifier, header)) + ')')
            conn.executemany('INSERT INTO ' + table + ' VALUES (' + ','.join('?' for _ in header) + ')',
                             block['rows'])
