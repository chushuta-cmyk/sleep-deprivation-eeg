#!/usr/bin/env python3
"""
Step 01: Pure-Python Multi-Threaded Resumable S3 Downloader for OpenNeuro ds004902 (71 Subjects)
特点：
1. 零外部依赖：无需安装 awscli，直接通过 AWS S3 REST XML API + 多线程 HTTPS 流式下载（自动兼容 macOS Python 3.13 SSL 证书环境）。
2. 智能过滤：默认仅下载主分析所需的 eyesopen EEG (.set/.fdt/.json/channels/electrodes) 与 PVT 行为文件 (*_beh.tsv)，体积减半、速度翻倍。
3. 安全断点续传：自动校验本地已存在的完整文件（比对字节大小），绝不覆盖本地已有的 sub-40 ~ sub-71 数据。
"""

import argparse
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S3_BASE_URL = "https://s3.amazonaws.com/openneuro.org"
DATASET_PREFIX = "ds004902"
SSL_CTX = ssl._create_unverified_context()


def list_s3_keys(prefix: str) -> list:
    """列出指定 S3 prefix 下的所有文件 (key, size_bytes)"""
    results = []
    continuation_token = None
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}

    while True:
        url = f"{S3_BASE_URL}?list-type=2&prefix={prefix}"
        if continuation_token:
            url += f"&continuation-token={urllib.parse.quote(continuation_token)}"
        with urllib.request.urlopen(url, timeout=20, context=SSL_CTX) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        for content in root.findall('s3:Contents', ns):
            key = content.find('s3:Key', ns).text
            size = int(content.find('s3:Size', ns).text)
            results.append((key, size))
        is_truncated = root.find('s3:IsTruncated', ns)
        if is_truncated is not None and is_truncated.text == 'true':
            continuation_token = root.find('s3:NextContinuationToken', ns).text
        else:
            break
    return results


def download_single_file(key: str, expected_size: int, output_root: Path, overwrite: bool = False) -> tuple:
    rel_path = key[len(DATASET_PREFIX) + 1:]
    local_path = output_root / rel_path
    if not overwrite and local_path.exists() and local_path.stat().st_size == expected_size:
        return (rel_path, "skipped", expected_size)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3_BASE_URL}/{key}"
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")

    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60, context=SSL_CTX) as resp, open(tmp_path, 'wb') as out_f:
                while True:
                    chunk = resp.read(262144)
                    if not chunk:
                        break
                    out_f.write(chunk)
            if tmp_path.stat().st_size == expected_size:
                tmp_path.replace(local_path)
                return (rel_path, "downloaded", expected_size)
        except Exception as e:
            if attempt == 2:
                return (rel_path, f"error: {e}", 0)
            time.sleep(1.5)
    return (rel_path, "size_mismatch", 0)


def download_cohort(output_dir: Path, start_sub: int = 1, end_sub: int = 71, include_eyesclosed: bool = False,
                    max_workers: int = 6, overwrite: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    target_subs = {f"sub-{i:02d}" for i in range(start_sub, end_sub + 1)}

    print(f"正在从 OpenNeuro S3 获取 {DATASET_PREFIX} 文件列表...")
    all_items = list_s3_keys(f"{DATASET_PREFIX}/")

    filtered_items = []
    for key, size in all_items:
        rel = key[len(DATASET_PREFIX) + 1:]
        parts = rel.split('/')
        # 顶层元数据文件
        if len(parts) == 1:
            filtered_items.append((key, size))
            continue
        sub_id = parts[0]
        if sub_id not in target_subs:
            continue
        if not include_eyesclosed and "task-eyesclosed" in rel:
            continue
        filtered_items.append((key, size))

    total_mb = sum(s for _, s in filtered_items) / 1e6
    print(f"待校验/同步文件总数: {len(filtered_items)} 个 (总计 {total_mb:.1f} MB) | 并行线程: {max_workers}")

    downloaded_mb = 0.0
    skipped_mb = 0.0
    errors = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(download_single_file, key, size, output_dir, overwrite)
            for key, size in filtered_items
        ]
        for idx, fut in enumerate(as_completed(futures), 1):
            rel_path, status, sz = fut.result()
            if status == "downloaded":
                downloaded_mb += sz / 1e6
                if idx % 10 == 0 or sz > 10_000_000:
                    elapsed = max(0.1, time.time() - t0)
                    print(f"  [{idx}/{len(filtered_items)}] ✓ 下载: {rel_path} ({sz/1e6:.1f} MB) | 速率: {downloaded_mb/elapsed:.1f} MB/s")
            elif status == "skipped":
                skipped_mb += sz / 1e6
            else:
                errors.append((rel_path, status))
                print(f"  [{idx}/{len(filtered_items)}] ✗ 失败: {rel_path} ({status})")

    elapsed = max(0.1, time.time() - t0)
    print("\n" + "=" * 75)
    print(f"同步完成！耗时: {elapsed:.1f}s | 新增下载: {downloaded_mb:.1f} MB | 已存在跳过: {skipped_mb:.1f} MB | 错误: {len(errors)}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Pure-Python Multi-Threaded OpenNeuro ds004902 Downloader")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--start-sub", type=int, default=1)
    parser.add_argument("--end-sub", type=int, default=71)
    parser.add_argument("--include-eyesclosed", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    download_cohort(
        Path(args.output),
        start_sub=args.start_sub,
        end_sub=args.end_sub,
        include_eyesclosed=args.include_eyesclosed,
        max_workers=args.workers,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
