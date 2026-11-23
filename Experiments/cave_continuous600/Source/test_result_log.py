"""Readable, checkpoint-local test summaries for CAVE and Pavia JSON reports."""
from pathlib import Path
import math
import os
import tempfile


def format_result(report, source_report=None):
    rows = report.get('per_scene', report.get('patches'))
    if not rows:
        raise ValueError('No per-image/patch test results')
    def values(row):
        result = (row['psnr'], row.get('sam', row.get('sam_degrees')), row['ergas'])
        if not all(v is not None and math.isfinite(float(v)) for v in result):
            raise ValueError('Missing or non-finite metric')
        return result
    lines = [f"Loading checkpoint: {report['checkpoint']}", f"训练轮数: {report['epoch']}"]
    if source_report is not None:
        lines.append(f'日志来源：已保存的测试报告 {source_report}（未重新推理）')
    for index, row in enumerate(rows, 1):
        psnr, sam, ergas = values(row)
        lines.append(f'[{index}/{len(rows)}] PSNR={psnr:.4f} SAM={sam:.4f} ERGAS={ergas:.4f}')
    psnr, sam, ergas = values(report['mean'])
    lines.extend(['', '========== Test Results ==========', f"Scale: {report['sf']}",
                  f'Test Images: {len(rows)}', f'Avg PSNR = {psnr:.4f}',
                  f'Avg SAM = {sam:.4f}', f'Avg ERGAS = {ergas:.4f}',
                  '=================================='])
    return '\n'.join(lines)+'\n'


def write_result_log(report, directory=None, source_report=None):
    directory = Path(directory) if directory is not None else Path(report['checkpoint']).resolve().parent
    name = 'test.log' if int(report['sf']) == 4 else f"test_x{int(report['sf'])}.log"
    path = directory/name
    # Existing logs may be the active stdout destination: never truncate them.
    if path.exists():
        return path
    content = format_result(report, source_report)
    directory.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+name, dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
        # Hard-link publication is atomic and never overwrites a concurrent log.
        try:
            os.link(temp, path)
        except FileExistsError:
            pass
    finally:
        os.unlink(temp)
    return path
