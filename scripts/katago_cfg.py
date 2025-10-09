#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple


def _default_cfg_paths() -> List[str]:
    return [
        os.environ.get("KATAGO_CFG") or "",
        "/katago-bin/default_gtp.cfg",
        "/root/katago-bin/default_gtp.cfg",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "katago", "default_gtp.cfg"),
    ]


def _resolve_cfg_path(explicit: str | None) -> str:
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(_default_cfg_paths())
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    raise SystemExit("Config file not found. Provide --cfg or set KATAGO_CFG.")


def _parse_kv(line: str) -> Tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith("#") or s.startswith(";"):
        return None
    if "=" not in s:
        return None
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def cmd_list(cfg: str) -> None:
    with open(cfg, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            kv = _parse_kv(ln)
            if kv is None:
                continue
            k, v = kv
            print(f"{k} = {v}")


def cmd_get(cfg: str, key: str) -> None:
    key_l = key.strip().lower()
    with open(cfg, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            kv = _parse_kv(ln)
            if kv is None:
                continue
            k, v = kv
            if k.strip().lower() == key_l:
                print(v)
                return
    sys.exit(1)


def cmd_set(cfg: str, key: str, value: str) -> None:
    key_l = key.strip().lower()
    lines = []
    found = False
    with open(cfg, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            kv = _parse_kv(ln)
            if kv is None:
                lines.append(ln)
                continue
            k, _ = kv
            if k.strip().lower() == key_l:
                lines.append(f"{k} = {value}\n")
                found = True
            else:
                lines.append(ln)
    if not found:
        lines.append(f"{key} = {value}\n")
    with open(cfg, "w", encoding="utf-8") as f:
        f.writelines(lines)


def cmd_unset(cfg: str, key: str) -> None:
    key_l = key.strip().lower()
    lines = []
    with open(cfg, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            kv = _parse_kv(ln)
            if kv is None:
                lines.append(ln)
                continue
            k, _ = kv
            if k.strip().lower() == key_l:
                continue
            lines.append(ln)
    with open(cfg, "w", encoding="utf-8") as f:
        f.writelines(lines)


def cmd_preset(cfg: str, name: str) -> None:
    name_l = name.strip().lower()
    if name_l == "nano-safe":
        # Maxwell Jetson Nano defaults
        cmd_set(cfg, "cudaUseFP16", "false")
        cmd_set(cfg, "cudaUseNHWC", "false")
        cmd_set(cfg, "numSearchThreads", "2")
        cmd_unset(cfg, "maxPlayouts")
        cmd_set(cfg, "cudaGraphMaxBatchSize", "0")
        cmd_set(cfg, "cudaGraphWorkspaceMB", "0")
    elif name_l == "orin-safe":
        # Orin (Ampere) defaults
        cmd_set(cfg, "cudaUseFP16", "true")
        cmd_set(cfg, "numSearchThreads", "8")
        cmd_unset(cfg, "maxPlayouts")
    else:
        raise SystemExit("Unknown preset. Use 'nano-safe' or 'orin-safe'.")


def main() -> None:
    p = argparse.ArgumentParser("KataGo cfg helper (list/get/set/unset/preset)")
    p.add_argument("action", choices=["list", "get", "set", "unset", "preset"]) 
    p.add_argument("key", nargs="?", help="Key for get/set/unset or preset name")
    p.add_argument("value", nargs="?", help="Value for set")
    p.add_argument("--cfg", dest="cfg", default=None, help="Path to default_gtp.cfg (optional)")
    args = p.parse_args()

    cfg = _resolve_cfg_path(args.cfg)
    act = args.action
    if act == "list":
        cmd_list(cfg)
    elif act == "get":
        if not args.key:
            raise SystemExit("get requires KEY")
        cmd_get(cfg, args.key)
    elif act == "set":
        if not args.key or args.value is None:
            raise SystemExit("set requires KEY VALUE")
        cmd_set(cfg, args.key, args.value)
    elif act == "unset":
        if not args.key:
            raise SystemExit("unset requires KEY")
        cmd_unset(cfg, args.key)
    elif act == "preset":
        if not args.key:
            raise SystemExit("preset requires NAME (nano-safe|orin-safe)")
        cmd_preset(cfg, args.key)


if __name__ == "__main__":
    main()


