"""Paths for root / runtime / scenario, with no process-wide chdir or mutable globals."""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Workspace:
    root: Path = Path(r'D:\codex\workspace')
    runtime: str = 'realtime'
    scenario: str = 'qinling'

    def __post_init__(self):
        for value in (self.runtime, self.scenario):
            if not re.fullmatch(r'[\w-]+', value):
                raise ValueError('runtime and scenario must be single directory names')
        object.__setattr__(self, 'root', Path(self.root).resolve())
        self._contained(self.root / self.runtime / self.scenario)

    def _contained(self, path):
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f'Workspace path escapes root: {path}')
        return resolved

    @property
    def directory(self):
        return self._contained(self.root / self.runtime / self.scenario)

    @property
    def model(self):
        return self._contained(self.directory / 'sqldb/model/devices.sqlite3')

    @property
    def scada(self):
        return self._contained(self.directory / 'sqldb/scada/scdfe_c.sqlite3')

    @property
    def point_catalog(self):
        return self._contained(self.directory / 'sqldb/scada/point_catalog.json')

    def result(self, task):
        if task not in {'se', 'lf', 'ca'}:
            raise ValueError(f'Unknown task: {task}')
        return self._contained(self.directory / 'sqldb/pas' / (task + '.sqlite3'))
