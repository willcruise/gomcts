from typing import Optional, Tuple


def pass_index(size: int) -> int:
    return int(size) * int(size)


def action_count(size: int) -> int:
    s = int(size)
    return s * s + 1


def rc_to_action(r: int, c: int, size: int) -> int:
    return int(r) * int(size) + int(c)


def action_to_rc(a: int, size: int) -> Optional[Tuple[int, int]]:
    if int(a) == pass_index(size):
        return None
    return divmod(int(a), int(size))


