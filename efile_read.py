import re
import threading
import weakref
from pathlib import Path

from paths import resolve_project_file


_QUOTED_SPLIT_RE = re.compile(r"\s+(?=(?:[^']*'[^']*')*[^']*$)")


def _looks_numeric_cell(value):
    if not isinstance(value, str) or not value:
        return False
    return value[0] in "+-.0123456789"


def _numeric_cell_kind(value):
    if not _looks_numeric_cell(value):
        return None
    return float if "." in value or "e" in value or "E" in value else int


def _safe_int_cell(value):
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def _safe_float_cell(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _identity_cell(value):
    return value


def _split_data_row(text):
    """Split one E-file data row, using the regex path only when quotes exist."""
    if "'" not in text:
        return text.split()
    return _QUOTED_SPLIT_RE.split(text.lstrip())


def _convert_cell(value):
    """Convert E-file scalar text to int/float only when it is numeric-like."""
    if not isinstance(value, str):
        return value
    if not value:
        return value
    first = value[0]
    if first not in "+-.0123456789":
        return value
    if "." in value or "e" in value or "E" in value:
        try:
            return float(value)
        except ValueError:
            return value
    try:
        return int(value)
    except ValueError:
        return value


class SingletonType(type):
    _instance_lock = threading.Lock()
    _instance_ref = weakref.WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance_ref:
            with SingletonType._instance_lock:
                if cls not in cls._instance_ref:
                    instance = super(SingletonType, cls).__call__(*args, **kwargs)
                    cls._instance_ref[cls] = instance
        return cls._instance_ref[cls]

class EBlock(object):
    """Represents a block."""
    __slots__ = ("name", "header_list", "data")

    def __init__(self, name):
        """Constructor
        :type name: str
        """
        self.name = name
        self.header_list = []
        self.data = []

    def AddRow(self, attr_list):
        """
        :type attr_list: ['attr1','attr2',...]
        """
        header = self.header_list
        if len(attr_list) != len(header):
            raise TypeError(f'attr_list length {len(attr_list)} {attr_list} not equal to header_list {header} length {len(header)}')
        self.data.append(dict(zip(header, attr_list)))
    
    def to_dict(self):
        return {'table_name':self.name,'header_list':self.header_list,'data':self.data}

cond_operators = {
    'eq': lambda a,b: a == b,
    'gt': lambda a,b: a > b,
    'lt': lambda a,b: a < b,
    'gte': lambda a,b: a >= b,
    'lte': lambda a,b: a <= b,
    'neq': lambda a,b: a != b
}

class EBook():
    """Represents a book."""

    # _write_lock = threading.Lock()
    def __init__(self, input):
        """Constructor
        :type name: str
        """
        self.data = {}
        if isinstance(input,str) or isinstance(input,Path):
            self.file_path = input
            self._read_file_(self.file_path)
        else:
            self._read_dict_(input)

        
    def _read_file_(self, file_path):
        file_path = resolve_project_file(file_path)
        data = self.data
        block_read = None
        split_data_row = _split_data_row
        # efile 解析：绝大多数文件没有带空格的引号字段，优先走 str.split 快路径。
        with open(file_path, mode='rt', encoding='utf8') as fp:
            for idx, line in enumerate(fp):
                line = line.strip()
                if not line:  # empty line
                    continue
                first = line[0]
                if first == '<':
                    if line.startswith('</'):  # block end
                        data[block_read.name] = block_read
                        continue
                    block_read = EBlock(line[1:-1])
                elif first == '@':  # header
                    block_read.header_list = line[1:].split()
                elif first == '#':
                    block_read.AddRow(split_data_row(line[1:]))
                else:
                    raise SyntaxError(f"Invalid row at line {idx + 1} {line} in {file_path}")
    
    def _tab_name_check(self, tab_name):
        if not self.data.get(tab_name, None):
            raise ValueError(f"Not found table {tab_name}")
        
    def to_dict(self):
        data_dict = {}
        for key, block in self.data.items():
            if ' lv ' in key:
                key_list = key.split(' lv ')
                data_dict[key_list[0]] = block.to_dict() | {'lv': int(key_list[1].strip().split('=')[1])}
            else:
                data_dict[key] = block.to_dict() | {'lv': 0}
        return data_dict

    def _read_dict_(self,data_dict):

        for key,v in data_dict.items():
            self.data[key] = EBlock(key)
            self.data[key].header_list = list(v[0].keys()) if v else []
            for d in v:
                self.data[key].AddRow(d.values())


    def apply_to_file(self,file_path=None):
        
        """将当前内容写入文件
        """
        if file_path:
            write_file_path = file_path
        else:
            write_file_path = self.file_path
        with open(write_file_path, mode='w', encoding='gb18030') as fp:
            parts = []
            for e in self.data.values():
                parts.append('<' + e.name + '>\n')
                parts.append('@ ' + '\t'.join(e.header_list) + '\n')
                for row in e.data:
                    parts.append('# ' + '\t'.join(str(item) for item in row.values()) + '\n')
                parts.append('</' + e.name + '>\n')
            fp.write(''.join(parts))


class _FlatTableRows:
    """Lazy row view backed by a flat token list from bulk block splitting.

    Column readers (EFileTableRows) slice ``tokens[col + 1 :: stride]`` directly
    in C without constructing per-row Python lists. Any row-level iteration or
    indexing materializes standard mutable Python row lists once and releases
    the flat buffer so in-place callers (e.g. contingency mutation) stay exact.
    """

    __slots__ = ("_tokens", "_stride", "_nrows", "_rows")

    def __init__(self, tokens, stride: int, nrows: int):
        self._tokens = tokens
        self._stride = int(stride)
        self._nrows = int(nrows)
        self._rows = None

    def _materialize(self):
        rows = self._rows
        if rows is None:
            tokens = self._tokens
            stride = self._stride
            rows = [tokens[i + 1 : i + stride] for i in range(0, self._nrows * stride, stride)]
            self._rows = rows
            self._tokens = None
        return rows

    def column_tokens(self, col: int):
        if self._rows is not None:
            return None
        col_idx = int(col)
        if col_idx < 0 or col_idx + 1 >= self._stride:
            return None
        return self._tokens[col_idx + 1 :: self._stride]

    def __len__(self) -> int:
        return len(self._rows) if self._rows is not None else self._nrows

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self):
        return iter(self._materialize())

    def __getitem__(self, item):
        return self._materialize()[item]

    def __eq__(self, other):
        if isinstance(other, _FlatTableRows):
            return self._materialize() == other._materialize()
        return self._materialize() == other


def _try_read_efile_rows_bulk(text: str):
    """Fast bulk-token parser for standard unquoted, comment-free E-files."""
    if "'" in text or "//" in text:
        return None
    data = {}
    pos = 0
    n = len(text)
    while pos < n:
        lt = text.find("<", pos)
        if lt < 0:
            if text[pos:].strip():
                return None
            break
        if text[pos:lt].strip():
            return None
        gt = text.find(">", lt + 1)
        if gt < 0:
            return None
        block_name = text[lt + 1 : gt]
        if not block_name or block_name.startswith("/"):
            return None
        table_name = block_name
        lv = 0
        if " lv " in block_name:
            key_list = block_name.split(" lv ")
            table_name = key_list[0]
            try:
                lv = int(key_list[1].strip().split("=")[1])
            except (IndexError, ValueError):
                return None
        end_tag = f"</{table_name}>"
        end_pos = text.find(end_tag, gt + 1)
        if end_pos < 0:
            return None
        body = text[gt + 1 : end_pos]
        at = body.find("@")
        if at < 0:
            header = []
            data_part = body.strip()
        else:
            if body[:at].strip():
                return None
            nl = body.find("\n", at)
            if nl < 0:
                header = body[at + 1 :].split()
                data_part = ""
            else:
                header = body[at + 1 : nl].split()
                data_part = body[nl + 1 :].strip()
        if not data_part:
            rows = []
        else:
            if "\n\n" in data_part or "\r\n\r\n" in data_part:
                return None
            nrows = data_part.count("\n") + 1
            tokens = data_part.split()
            stride = len(header) + 1
            if stride <= 1 or len(tokens) != nrows * stride:
                return None
            # 直接统计首列 '#' 个数，避免构造临时集合与字符串哈希
            if tokens[0::stride].count("#") != nrows:
                return None
            rows = _FlatTableRows(tokens, stride, nrows)
        data[table_name] = {
            "table_name": table_name,
            "header_list": header,
            "rows": rows,
            "lv": lv,
        }
        pos = end_pos + len(end_tag)
    return data


def _read_efile_rows(file_path):
    file_path = resolve_project_file(file_path)
    with open(file_path, mode="rt", encoding="utf8") as fp:
        text = fp.read()

    bulk = _try_read_efile_rows_bulk(text)
    if bulk is not None:
        return bulk

    data = {}
    block_name = None
    header = None
    rows = None
    rows_append = None
    split_data_row = _split_data_row

    def finish_block():
        if block_name is None:
            return
        table_name = block_name
        lv = 0
        if " lv " in block_name:
            key_list = block_name.split(" lv ")
            table_name = key_list[0]
            lv = int(key_list[1].strip().split("=")[1])
        data[table_name] = {
            "table_name": table_name,
            "header_list": header or [],
            "rows": rows or [],
            "lv": lv,
        }

    has_quotes = "'" in text
    for idx, raw_line in enumerate(text.splitlines()):
        if not raw_line:
            continue
        first = raw_line[0]
        if first == "#":
            if block_name is None:
                raise SyntaxError(f"Data row outside block at line {idx + 1} in {file_path}")
            if not has_quotes:
                if len(raw_line) > 1 and raw_line[1] <= " ":
                    rows_append(raw_line.split()[1:])
                else:
                    rows_append(raw_line[1:].split())
            else:
                body = raw_line[1:]
                rows_append(body.split() if "'" not in body else split_data_row(body))
            continue
        line = raw_line.strip()
        if not line:
            continue
        first = line[0]
        if first == "<":
            if line.startswith("</"):
                finish_block()
                block_name = header = rows = rows_append = None
                continue
            block_name = line[1:-1]
            header = []
            rows = []
            rows_append = rows.append
        elif first == "@":
            if block_name is None:
                raise SyntaxError(f"Header outside block at line {idx + 1} in {file_path}")
            header = line[1:].split()
        elif first == "#":
            if block_name is None:
                raise SyntaxError(f"Data row outside block at line {idx + 1} in {file_path}")
            body = line[1:]
            rows_append(body.split() if (not has_quotes or "'" not in body) else split_data_row(body))
        else:
            raise SyntaxError(f"Invalid row at line {idx + 1} {line} in {file_path}")
    return data


class _Base:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, _convert_cell(value))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


def _class_factory(class_name, attrs, converters=None):
    if not class_name:
        raise ValueError("JSON data must contain a 'data' key")
    if converters is None:
        converters = tuple(_convert_cell for _ in attrs)
    else:
        converters = tuple(converters)
    attr_names = tuple(attrs)

    init_globals = {}
    init_lines = [
        "def __init__(self, row=None, **kwargs):",
        "    source = kwargs if row is None else row",
        "    values = self.__dict__",
    ]
    for idx, (attr, convert) in enumerate(zip(attr_names, converters)):
        source_expr = f"source[{attr!r}]"
        if convert is _identity_cell:
            init_lines.append(f"    values[{attr!r}] = {source_expr}")
        elif convert is _safe_int_cell:
            value_name = f"_value_{idx}"
            init_lines.extend(
                [
                    f"    {value_name} = {source_expr}",
                    f"    if isinstance({value_name}, str):",
                    "        try:",
                    f"            values[{attr!r}] = int({value_name})",
                    "        except ValueError:",
                    "            try:",
                    f"                values[{attr!r}] = float({value_name})",
                    "            except ValueError:",
                    f"                values[{attr!r}] = {value_name}",
                    "    else:",
                    f"        values[{attr!r}] = {value_name}",
                ]
            )
        elif convert is _safe_float_cell:
            value_name = f"_value_{idx}"
            init_lines.extend(
                [
                    f"    {value_name} = {source_expr}",
                    f"    if isinstance({value_name}, str):",
                    "        try:",
                    f"            values[{attr!r}] = float({value_name})",
                    "        except ValueError:",
                    f"            values[{attr!r}] = {value_name}",
                    "    else:",
                    f"        values[{attr!r}] = {value_name}",
                ]
            )
        else:
            converter_name = f"_converter_{idx}"
            init_globals[converter_name] = convert
            init_lines.append(f"    values[{attr!r}] = {converter_name}({source_expr})")
    exec("\n".join(init_lines), init_globals)
    __init__ = init_globals["__init__"]

    attributes = {attr: None for attr in attrs}
    attributes["__init__"] = __init__
    return type(class_name, (_Base,), attributes)


def _row_class_factory(class_name, attrs, converters=None):
    if not class_name:
        raise ValueError("Row data must contain a table name")
    if converters is None:
        converters = tuple(_convert_cell for _ in attrs)
    else:
        converters = tuple(converters)

    init_globals = {}
    init_lines = [
        "def __init__(self, row):",
        "    values = self.__dict__",
    ]
    for idx, (attr, convert) in enumerate(zip(attrs, converters)):
        source_expr = f"row[{idx}]"
        if convert is _identity_cell:
            init_lines.append(f"    values[{attr!r}] = {source_expr}")
        elif convert is _safe_int_cell:
            value_name = f"_value_{idx}"
            init_lines.extend(
                [
                    f"    {value_name} = {source_expr}",
                    f"    if isinstance({value_name}, str):",
                    "        try:",
                    f"            values[{attr!r}] = int({value_name})",
                    "        except ValueError:",
                    "            try:",
                    f"                values[{attr!r}] = float({value_name})",
                    "            except ValueError:",
                    f"                values[{attr!r}] = {value_name}",
                    "    else:",
                    f"        values[{attr!r}] = {value_name}",
                ]
            )
        elif convert is _safe_float_cell:
            value_name = f"_value_{idx}"
            init_lines.extend(
                [
                    f"    {value_name} = {source_expr}",
                    f"    if isinstance({value_name}, str):",
                    "        try:",
                    f"            values[{attr!r}] = float({value_name})",
                    "        except ValueError:",
                    f"            values[{attr!r}] = {value_name}",
                    "    else:",
                    f"        values[{attr!r}] = {value_name}",
                ]
            )
        else:
            converter_name = f"_converter_{idx}"
            init_globals[converter_name] = convert
            init_lines.append(f"    values[{attr!r}] = {converter_name}({source_expr})")
    exec("\n".join(init_lines), init_globals)
    attributes = {attr: None for attr in attrs}
    attributes["__init__"] = init_globals["__init__"]
    return type(class_name, (_Base,), attributes)


def _infer_row_converters(header, rows):
    converters = [None] * len(header)
    unresolved = len(header)
    if unresolved:
        for row in rows:
            for idx, sample in enumerate(row[: len(header)]):
                if converters[idx] is not None or sample == "":
                    continue
                kind = _numeric_cell_kind(sample)
                if kind is float:
                    converters[idx] = _safe_float_cell
                elif kind is int:
                    converters[idx] = _safe_int_cell
                else:
                    converters[idx] = _identity_cell
                unresolved -= 1
            if unresolved == 0:
                break
    return [_identity_cell if converter is None else converter for converter in converters]


def efile_factory_from_rows(data):
    """Build model objects from raw row tokens, bypassing EBook row dictionaries."""
    new_cls_dict = _Base()
    for key, value in data.items():
        header = value["header_list"]
        rows = value["rows"]
        converters = _infer_row_converters(header, rows)
        Class_Def = _row_class_factory(key, header, converters)
        setattr(new_cls_dict, key, [Class_Def(item) for item in rows])
        if "lv" in value:
            setattr(new_cls_dict, f"{key}_lv", value["lv"])
    return new_cls_dict


def efile_factory_from_file(file_path):
    return efile_factory_from_rows(_read_efile_rows(file_path))


def efile_factory(data):
    new_cls_dict = _Base()
    for key, value in data.items():
        header = value['header_list']
        rows = value["data"]
        converters = [None] * len(header)
        unresolved = len(header)
        if unresolved:
            for item in rows:
                for idx, attr in enumerate(header):
                    if converters[idx] is not None:
                        continue
                    sample = item.get(attr, "")
                    if sample == "":
                        continue
                    kind = _numeric_cell_kind(sample)
                    if kind is float:
                        converters[idx] = _safe_float_cell
                    elif kind is int:
                        converters[idx] = _safe_int_cell
                    else:
                        converters[idx] = _identity_cell
                    unresolved -= 1
                if unresolved == 0:
                    break
        converters = [_identity_cell if converter is None else converter for converter in converters]
        Class_Def = _class_factory(key, header, converters)
        setattr(new_cls_dict, key, [Class_Def(item) for item in rows])

    return new_cls_dict
