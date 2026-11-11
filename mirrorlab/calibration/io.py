"""Read original survey workbooks and tab-delimited Russian coordinate files."""
import csv
import posixpath
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .models import Measurement

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
FILES = {"west_to_east": "Геодезические измерения из-к+.xlsx",
         "east_to_west": "Геодезические измерения из+к-.xlsx"}


def number(value):
    if value is None or str(value).strip() in ("", "-", "—", "–"):
        return None
    return float(str(value).strip().replace(",", "."))


def _workbook_rows(path):
    """Read numeric XML directly: date-formatted coordinates must stay numbers."""
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(node.itertext()) for node in
                       ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        styles = ET.fromstring(archive.read("xl/styles.xml"))
        formats = {int(x.attrib["numFmtId"]): x.attrib["formatCode"]
                   for x in styles.findall("m:numFmts/m:numFmt", NS)}
        date_styles = set()
        for i, style in enumerate(styles.find("m:cellXfs", NS)):
            fmt = int(style.attrib.get("numFmtId", 0))
            if fmt in set(range(14, 23)) | {45, 46, 47} or re.search(
                    r"[dmyhs]", formats.get(fmt, ""), re.I):
                date_styles.add(i)
        rels = {x.attrib["Id"]: x.attrib["Target"] for x in
                ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in book.findall("m:sheets/m:sheet", NS):
            target = rels[sheet.attrib[f"{{{REL}}}id"]]
            target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            xml = ET.fromstring(archive.read(target))
            for row in xml.findall("m:sheetData/m:row", NS):
                cells, dates = {}, []
                for cell in row:
                    ref = cell.attrib["r"]
                    col = re.match(r"[A-Z]+", ref).group()
                    node = cell.find("m:v", NS)
                    kind = cell.attrib.get("t")
                    if kind == "s" and node is not None:
                        value = strings[int(node.text)]
                    elif kind == "inlineStr":
                        value = "".join(cell.find("m:is", NS).itertext())
                    else:
                        value = node.text if node is not None else None
                    cells[col] = value
                    if int(cell.attrib.get("s", 0)) in date_styles and value is not None:
                        dates.append(ref)
                yield sheet.attrib["name"], int(row.attrib["r"]), cells, dates


def read_survey(path):
    records, warnings, rod = [], [], None
    seen = set()
    for sheet, row, cells, dates in _workbook_rows(path):
        label = cells.get("A") or ""
        match = re.fullmatch(r"(\d+)_(\d+)", label.strip())
        if not match:
            heading = re.search(r"(?:стержень|сектор|пучок|сборка)\s*(\d+)", label, re.I)
            # Supplied sheets use 'Положение стержня N' and 'Стержень N'.
            if heading is None and re.search(r"стерж", label, re.I):
                heading = re.search(r"(\d+)\s*$", label)
            if heading:
                rod = int(heading.group(1))
            continue
        if rod is None:
            raise ValueError(f"{path}: {sheet}!A{row}: отсутствует номер стержня")
        window, ray = map(int, match.groups())
        key = rod, window, ray
        if key in seen:
            raise ValueError(f"Повтор измерения {key} в {path}")
        seen.add(key)
        screens = []
        for cols in (("B", "C", "D"), ("F", "G", "H")):
            values = [number(cells.get(col)) for col in cols]
            screens.append([v / 1000 if v is not None else None for v in values])
        source = f"{Path(path).name} / {sheet}!A{row}:H{row}"
        records.append(Measurement(rod, window, ray, screens, source))
        for cell in dates:
            if re.match(r"[A-Z]+", cell).group() in ("B", "C", "D", "F", "G", "H"):
                warnings.append(f"{sheet}!{cell}: формат даты; использовано исходное числовое значение XML ({cells[re.match(r'[A-Z]+', cell).group()]} мм).")
    if not records:
        raise ValueError(f"В {path} не найдены измерения")
    return records, warnings


def read_equations(path):
    records, rod = {}, None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) < 8 or not re.fullmatch(r"\d+_\d+", row[1].strip()):
                continue
            if row[0].strip():
                rod = int(row[0])
            if rod is None:
                raise ValueError("В уравнениях отсутствует номер стержня")
            key = (rod, *map(int, row[1].split("_")))
            if key in records:
                raise ValueError(f"Повтор уравнения {key}")
            records[key] = [number(x) for x in row[2:8]]
    return records


def read_lavsan(path):
    points, rod, window = {}, None, None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"Стержень\s+(\d+)", line.strip(), re.I)
        if match:
            rod, window = int(match.group(1)), None
            continue
        match = re.match(r"Окно\s+(\d+)", line.strip(), re.I)
        if match:
            window = int(match.group(1))
            continue
        cols = line.split("\t")
        if len(cols) < 4 or not cols[0].strip().isdigit():
            continue
        if rod is None or window is None:
            raise ValueError("Не определены стержень/окно для точки лавсана")
        values = [number(x) for x in cols[1:4]]
        key = rod, window, int(cols[0])
        if key in points:
            raise ValueError(f"Повтор точки лавсана {key}")
        if all(x is not None for x in values):
            points[key] = [x / 1000 for x in values]
    return points
