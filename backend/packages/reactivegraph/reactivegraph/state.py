"""Transactional reactive state, Python side (Task 5 / docs/spec/state-and-transactions.md).

`TrackedStateProxy` wraps plain dicts/lists with recursive proxies that:
- record read paths during access,
- record mutation patches for direct mutation and returned update dicts,
- provide `snapshot()` returning a plain (deep, non-proxied) dictionary for
  opaque libraries that require a plain dict,
- round-trip canonical patches identically to the TypeScript store (shared corpus).

Mirrors packages/driver/src/state/*.ts. Paths use the canonical encoding
`a.b[0].c` from the TS path codec.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

from reactivegraph.protocol import encode_value
from reactivegraph.wire import to_wire_value


def path_to_string(path: tuple) -> str:
    """Canonical path string, e.g. ('a', 0, 'b') -> 'a[0].b'."""
    out = ""
    for i, seg in enumerate(path):
        if isinstance(seg, int):
            out += f"[{seg}]"
        else:
            if i > 0:
                out += "."
            out += str(seg).replace(".", "\\.").replace("[", "\\[")
    return out


def parse_path(canonical: str) -> tuple:
    """Split a dotted/bracketed path into its segments."""
    if canonical == "":
        return ()
    segments: list = []
    i = 0
    current = ""
    while i < len(canonical):
        ch = canonical[i]
        if ch == "\\":
            if i + 1 < len(canonical):
                current += canonical[i + 1]
            i += 2
            continue
        if ch == "[":
            j = i + 1
            num_str = ""
            while j < len(canonical) and canonical[j] != "]":
                num_str += canonical[j]
                j += 1
            if num_str != "":
                if current != "":
                    segments.append(current)
                    current = ""
                segments.append(int(num_str))
            i = j + 1
            continue
        if ch == ".":
            if current != "":
                segments.append(current)
            current = ""
            i += 1
            continue
        current += ch
        i += 1
    if current != "":
        segments.append(current)
    return tuple(segments)


_MISSING = object()
_MISSING_MARKER = "rgp:missing"


def _get_at_path(root: Any, path: tuple) -> Any:
    cur = root
    for seg in path:
        if isinstance(cur, dict):
            cur = cur.get(seg, _MISSING)
        elif isinstance(cur, (list, tuple)) and isinstance(seg, int):
            if -len(cur) <= seg < len(cur):
                cur = cur[seg]
            else:
                return _MISSING
        else:
            return _MISSING
        if cur is _MISSING:
            return _MISSING
    return cur


def canonical_hash(value: Any) -> str:
    """SHA-256 hex over the canonical msgpack encoding (mirror of TS hash.ts).

    Missing values (`_MISSING`) hash the fixed marker "rgp:missing" so pre-image
    checks for missing paths match the TypeScript side exactly; `None` stays a
    canonical msgpack nil.
    """
    if value is _MISSING:
        payload = encode_value(_MISSING_MARKER)
    else:
        payload = encode_value(to_wire_value(value))
    return hashlib.sha256(payload).hexdigest()


def _deep_copy(value: Any) -> Any:
    if isinstance(value, TrackedStateProxy):
        return value.snapshot()
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_copy(v) for v in value]
    return value


class TrackedStateProxy:
    """Recursive mapping/list proxy that records reads and mutation patches."""

    def __init__(
        self,
        initial: dict[Any, Any] | list[Any] | None = None,
        *,
        base_path: tuple[Any, ...] = (),
        parent: TrackedStateProxy | None = None,
    ) -> None:
        self._base_path = base_path
        self._parent = parent
        self._reads: set[str] = set()
        self._patches: list[dict] = []
        self._root_data = initial if initial is not None else {}
        self._children: dict[Any, TrackedStateProxy] = {}

    # -- raw data -----------------------------------------------------------

    def _root(self) -> TrackedStateProxy:
        root = self
        while root._parent is not None:
            root = root._parent
        return root

    def _raw(self) -> Any:
        """The underlying raw object at this proxy's base path (from the root)."""
        if self._parent is None:
            return self._root_data
        return _get_at_path(self._root()._root_data, self._base_path)

    def _record_read(self, path: tuple) -> None:
        full = self._base_path + path
        canonical = path_to_string(full)
        self._reads.add(canonical)
        if self._parent is not None:
            self._root()._reads.add(canonical)

    def _record_patch(self, op: str, path: tuple, value: Any, before: Any) -> None:
        full = self._base_path + path
        patch = {
            "path": full,
            "operation": op,
            "beforeHash": canonical_hash(before),
            "value": _deep_copy(value),
            "taskId": "",
            "transactionId": "",
        }
        self._patches.append(patch)
        if self._parent is not None:
            self._root()._patches.append(patch)

    # -- reads --------------------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        raw = self._raw()
        if isinstance(raw, dict):
            self._record_read((key,))
            if key not in raw:
                raise KeyError(key)
            value = raw[key]
            return self._wrap_child(key, value)
        if isinstance(raw, list):
            if isinstance(key, int):
                self._record_read((key,))
                return self._wrap_child(key, raw[key])
            if isinstance(key, slice):
                # 切片读取：返回深拷贝切片（防 mutate 切片元素绕过 patch 追踪；
                # 浅拷贝会共享原始元素引用——security-review LOW）
                return _deep_copy(raw[key])
        raise KeyError(key)

    def _wrap_child(self, key: Any, value: Any) -> Any:
        if isinstance(value, dict) or isinstance(value, list):
            child = self._children.get(key)
            if child is None:
                child = TrackedStateProxy(
                    initial=value,
                    base_path=self._base_path + (key,),
                    parent=self,
                )
                self._children[key] = child
            return child
        return value

    def get(self, key: Any, default: Any = None) -> Any:
        """Tracked read of *key*, recording it in the current scope."""
        raw = self._raw()
        self._record_read((key,))
        if isinstance(raw, dict) and key in raw:
            return self._wrap_child(key, raw[key])
        return default

    def keys(self) -> list:
        """Tracked state keys in insertion order."""
        raw = self._raw()
        return list(raw.keys()) if isinstance(raw, dict) else []

    def __iter__(self) -> Iterator:
        raw = self._raw()
        if isinstance(raw, dict):
            return iter(raw.keys())
        if isinstance(raw, list):
            # 嵌套 list 迭代元素（realworld Task 5 会话历史累积暴露：原实现
            # 对 list 迭代空，`list(state["history"])` 恒空 → 状态无法读取）。
            return iter(raw)
        return iter(())

    def __contains__(self, key: Any) -> bool:
        raw = self._raw()
        self._record_read((key,))
        return key in raw if isinstance(raw, dict) else False

    def __len__(self) -> int:
        raw = self._raw()
        return len(raw) if isinstance(raw, (dict, list)) else 0

    # -- writes -------------------------------------------------------------

    def __setitem__(self, key: Any, value: Any) -> None:
        raw = self._raw()
        before = raw.get(key, _MISSING) if isinstance(raw, dict) else raw[key]
        if isinstance(raw, dict):
            raw[key] = value
            self._record_patch("set", (key,), value, None if before is _MISSING else before)
        elif isinstance(raw, list) and isinstance(key, int):
            raw[key] = value
            self._record_patch("set", (key,), value, before)

    def __delitem__(self, key: Any) -> None:
        raw = self._raw()
        if isinstance(raw, dict):
            before = raw.pop(key, _MISSING)
            if before is not _MISSING:
                self._record_patch("delete", (key,), None, before)
        elif isinstance(raw, list) and isinstance(key, int):
            before = raw.pop(key)
            self._record_patch("remove", (key,), key, before)

    def append(self, value: Any) -> None:
        """Append one tracked path mutation."""
        raw = self._raw()
        if isinstance(raw, list):
            before = list(raw)
            raw.append(value)
            self._record_patch("append", (), value, before)

    # -- snapshots ----------------------------------------------------------

    def __repr__(self) -> str:
        """Repr 展示内容而非对象地址——模板拼接/日志产出可读文本
        （deerflow 移植暴露：f-string 拼接 state 容器值曾输出
        `<TrackedStateProxy object at ...>`）。"""
        return repr(self.snapshot())

    def snapshot(self) -> dict:
        """Return a plain deep-copied dict (no proxies) for opaque libraries."""
        return _deep_copy(self._root_data)

    def read_paths(self) -> list[str]:
        """Recorded read paths (canonical form)."""
        return sorted(self._reads)

    def patches(self) -> list[dict]:
        """Recorded mutation patches."""
        return list(self._patches)

    def clear_tracking(self) -> None:
        """Reset the tracked read/write sets."""
        self._reads.clear()
        self._patches.clear()
        self._children.clear()


class TrackedState:
    """Top-level tracked state container with transaction semantics (light)."""

    def __init__(self, initial: dict | None = None) -> None:
        self._proxy = TrackedStateProxy(initial)
        self._committed: dict = _deep_copy(self._proxy.snapshot())
        self._version = 0

    @property
    def proxy(self) -> TrackedStateProxy:
        """Create a tracked proxy over this state."""
        return self._proxy

    def begin(self) -> TrackedState:
        """Start a new tracking scope."""
        self._proxy.clear_tracking()
        return self

    def commit(self) -> list[dict]:
        """Apply a plain dict of returned updates (like a reducer return)."""
        self._version += 1
        return self._proxy.patches()

    def rollback(self) -> None:
        """Restore the committed snapshot (raw replacement)."""
        self._proxy = TrackedStateProxy(_deep_copy(self._committed))

    def version(self) -> int:
        """State version incremented on every commit."""
        return self._version

    def state(self) -> dict:
        """The underlying plain dict."""
        return self._proxy.snapshot()


__all__ = (
    "TrackedState",
    "TrackedStateProxy",
    "canonical_hash",
    "parse_path",
    "path_to_string",
)