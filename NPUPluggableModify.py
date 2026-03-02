#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据指定内容定位行，并删除该行中所有的 const。
不读取除目标文件外的任何文件。
"""

TARGETS = [
    {
        "path": "/usr/local/libtorch_npu/include/torch_npu/csrc/npu/NPUPluggableAllocator.h",
        "contains": "c10::DataPtr allocate(size_t size)",
    },
    {
        "path": "/usr/local/lib64/python3.11/site-packages/torch/include/c10/core/Allocator.h",
        "contains": "virtual DataPtr allocate(size_t n)",
    },
    {
        "path": "/usr/local/lib64/python3.11/site-packages/torch/include/c10/core/TensorImpl.h",
        "contains": "storage_.allocator();",
    },
]


def process_file(filepath: str, line_contains: str) -> bool:
    """找到包含 line_contains 的行，删除该行中所有的 const 并写回文件。"""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line_contains in line:
            new_line = line.replace("const", "")
            if new_line != line:
                lines[i] = new_line
                with open(filepath, "w", encoding="utf-8", newline="") as f:
                    f.writelines(lines)
                print(f"[{filepath}:{i + 1}] 已删除该行所有 const")
                print(f"  原: {line.rstrip()}")
                print(f"  新: {new_line.rstrip()}")
                return True
            else:
                print(f"[{filepath}:{i + 1}] 该行无 const，未修改")
                return True
    else:
        print(f"警告: 在 {filepath} 中未找到包含 '{line_contains}' 的行")
        return False


if __name__ == "__main__":
    for t in TARGETS:
        process_file(t["path"], t["contains"])
