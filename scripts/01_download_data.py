#!/usr/bin/env python3
"""
Step 01: Resumable & Non-Destructive Downloader for OpenNeuro ds004902 (71 Subjects)
支持模式：
1. 默认增量补齐 (--resume-missing): 自动扫描本地 data/raw，仅下载缺失或未完整下载的被试 (如 sub-01 ~ sub-39)，绝不覆盖已有的完整文件。
2. 指定被试范围 (--start-sub 1 --end-sub 39): 精确控制下载范围。
3. 试跑模式 (--pilot): 仅下载前 N 名被试用于快速测试。
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S3_BUCKET = "s3://openneuro.org/ds004902"


def ensure_aws_cli():
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing awscli...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "awscli"], check=True)


def is_subject_complete(sub_dir: Path) -> bool:
    """检查该被试是否已包含 ses-1 和 ses-2 的 eyesopen .set 与 .fdt 文件"""
    if not sub_dir.exists():
        return False
    for ses in ["ses-1", "ses-2"]:
        set_files = list((sub_dir / ses).rglob("*eyesopen_eeg.set"))
        fdt_files = list((sub_dir / ses).rglob("*eyesopen_eeg.fdt"))
        if not set_files or not fdt_files:
            return False
    return True


def download_metadata(output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    for fname in ["dataset_description.json", "participants.tsv", "participants.json", "README", "CHANGES"]:
        target = output_path / fname
        if not target.exists():
            subprocess.run(
                ["aws", "s3", "cp", f"{S3_BUCKET}/{fname}", str(target), "--no-sign-request"],
                check=False
            )


def download_subjects(output_path: Path, sub_ids: list, overwrite: bool = False):
    download_metadata(output_path)
    for idx in sub_ids:
        sub_str = f"sub-{idx:02d}"
        sub_dir = output_path / sub_str
        if not overwrite and is_subject_complete(sub_dir):
            print(f"  ✓ {sub_str} 已完整存在，跳过 (不覆盖本地数据)")
            continue

        print(f"  ↓ 正在同步 {sub_str} ...")
        cmd = [
            "aws", "s3", "sync",
            f"{S3_BUCKET}/{sub_str}",
            str(sub_dir),
            "--no-sign-request",
        ]
        subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description="Resumable OpenNeuro ds004902 Downloader")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "raw"),
                        help="数据保存目录 (默认: data/raw)")
    parser.add_argument("--start-sub", type=int, default=1, help="起始被试编号 (1-71)")
    parser.add_argument("--end-sub", type=int, default=71, help="结束被试编号 (1-71)")
    parser.add_argument("--pilot", action="store_true", help="仅下载前 6 名被试")
    parser.add_argument("--overwrite", action="store_true", help="是否强制重新同步已存在被试")
    args = parser.parse_args()

    ensure_aws_cli()
    out_dir = Path(args.output)
    if args.pilot:
        sub_list = list(range(1, 7))
    else:
        sub_list = list(range(args.start_sub, args.end_sub + 1))

    print(f"目标目录: {out_dir} | 计划检查/下载被试范围: sub-{sub_list[0]:02d} ~ sub-{sub_list[-1]:02d}")
    download_subjects(out_dir, sub_list, overwrite=args.overwrite)
    print("✓ 数据检查与同步完成！")


if __name__ == "__main__":
    main()
