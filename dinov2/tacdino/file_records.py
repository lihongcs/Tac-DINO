"""Shared fair-comparison contract. No TEST-driven model selection."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()
def read(path):
    return json.loads(Path(path).read_text())
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name("."+path.name+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.replace(tmp,path)
def file_record(path):
    p=Path(path).resolve()
    return dict(path=str(p), sha256=sha(p))
def checked_file(record):
    p=Path(record["path"])
    if sha(p)!=record["sha256"]: raise ValueError(f"changed input: {p}")
    return p
