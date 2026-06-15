from __future__ import annotations

from collections.abc import Iterable, Iterator


class _Missing:
    def __bool__(self) -> bool:
        return False

    def __contains__(self, item: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        # Overriding __eq__ implicitly sets __hash__ = None in Python 3. Without
        # this, `MISSING in {...}` and using MISSING as a dict key would raise
        # TypeError — defeating the sentinel's purpose. All instances of
        # _Missing collide, which is fine because __eq__ returns False so
        # sets/dicts will not actually treat them as equal.
        return 0

    def __ge__(self, other: object) -> bool:
        return False

    def __gt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __lt__(self, other: object) -> bool:
        return False

    def __add__(self, other: object) -> _Missing:
        return self

    def __radd__(self, other: object) -> _Missing:
        return self

    def __sub__(self, other: object) -> _Missing:
        return self

    def __rsub__(self, other: object) -> _Missing:
        return self

    def __getitem__(self, key: object) -> _Missing:
        return self

    def __iter__(self) -> Iterator[object]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "<missing>"

    def __str__(self) -> str:
        return "<missing>"

    def get(self, key: object, default: object = None) -> object:
        return default

    def items(self) -> Iterable[tuple[object, object]]:
        return ()

    def keys(self) -> Iterable[object]:
        return ()

    def values(self) -> Iterable[object]:
        return ()


MISSING = _Missing()


class SafeDict(dict):
    def __getitem__(self, key: object) -> object:
        return super().get(key, MISSING)


class SafeList(list):
    def __getitem__(self, index: object) -> object:
        try:
            return super().__getitem__(index)  # type: ignore[arg-type]
        except (IndexError, TypeError):
            return MISSING


def _wrap_nested(value: object) -> object:
    if isinstance(value, dict):
        return SafeDict({key: _wrap_nested(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return SafeList(_wrap_nested(item) for item in value)
    return value


def safe_result(value: object) -> object:
    if isinstance(value, dict | list | tuple):
        return _wrap_nested(value)
    return MISSING
