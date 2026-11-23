"""Build corrected comparison CSV/Markdown/XLSX from recorded measurements.

The user's workbook was supplied as a screenshot; this creates a new workbook.
No existing spreadsheet, checkpoint or experiment report is changed.
"""
import csv
import hashlib
import json
import math
import unicodedata
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
WORK = PROJECT.parent
OLD = PROJECT / 'Experiments/Results/model_complexity_20261116/complete_rules'
DATASETS = [('cave', 'CAVE'), ('chikusei', 'Chikusei'), ('pavia', 'Pavia Centre')]
MODELS = ['SSR-Net', 'MHF-Net', 'SFNO', 'V3-stage3']


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def model_key(name):
    if name == 'SSR-NET':
        return 'SSR-Net'
    if name.startswith('SPGFNet'):
        return 'V3-stage3'
    return name


def results_path(dataset, model, sf):
    if model in ('SFNO', 'V3-stage3'):
        tag = 'sfno' if model == 'SFNO' else 'v3_stage3'
        experiment = 'cave_continuous600' if dataset == 'cave' else f'{tag}_restart_20261111'
        directory = PROJECT / f'Experiments/{experiment}/Checkpoint/{dataset}_{tag}_x4_seed42'
        filename = ('metrics.json' if sf == 4 else 'metrics_x8.json') if dataset == 'cave' else f'last_test_x{sf}.json'
    else:
        repository, tag = ('SSRNET-master', 'ssrnet') if model == 'SSR-Net' else ('MHF-net-master', 'mhf')
        directory = WORK / f'{repository}/comparison/Checkpoint/{dataset}_{tag}_x4_seed42'
        filename = f'last_test_x{sf}.json'
    return directory / filename


def residual_params(channels, guide=0):
    a, b, c, d = channels + guide, channels + guide + 3, channels + guide + 3, channels
    return 9 * (a*b + b*c + c*d) + 3 * (b+c+d)


def mhf_params(channels, msi, rank=12):
    up = 18*channels**2 + residual_params(channels, msi) + 16*channels
    return channels*msi + channels*rank + 36 + 19*(up + residual_params(rank)) + 4*residual_params(channels)


def collect():
    sources = {}
    for relative, expected in json.loads((OLD / 'source_hashes.json').read_text()).items():
        actual = digest(WORK / relative)
        assert actual == expected, relative
        sources[relative] = actual
    values = {}
    old_rows = json.loads((OLD / 'details.json').read_text())
    for row in old_rows:
        assert not row['unaccounted_ops']
        values[(row['dataset'], model_key(row['model']), row['scale'])] = row
    for checkpoint, expected in {(r['checkpoint'], r['checkpoint_sha256']) for r in old_rows}:
        assert digest(Path(checkpoint)) == expected, checkpoint
    mhf = json.loads((OUT / 'mhf_details.json').read_text())
    assert digest(WORK / 'MHF-net-master/comparison/adapter.py') == mhf['adapter_sha256']
    sources['MHF-net-master/comparison/adapter.py'] = mhf['adapter_sha256']
    for row in mhf['results']:
        assert not row['unaccounted_ops']
        assert row['params'] == mhf_params(row['hsi_channels'], row['msi_channels'])
        values[(row['dataset'], 'MHF-Net', row['scale'])] = row
    table, evidence = [], []
    for dataset, display in DATASETS:
        for model in MODELS:
            x4, x8 = values[(dataset, model, 4)], values[(dataset, model, 8)]
            assert x4['params'] == x8['params']
            metrics = []
            for sf in (4, 8):
                path = results_path(dataset, model, sf)
                if path.exists():
                    mean = json.loads(path.read_text())['mean']
                    metrics += [mean['psnr'], mean.get('sam_degrees', mean.get('sam')), mean['ergas']]
                    evidence.append(dict(dataset=dataset, model=model, sf=sf, report=str(path), sha256=digest(path)))
                else:
                    metrics += [None] * 3
            table.append(dict(dataset=dataset, display=display, model=model,
                              params=x4['params'], params_M=x4['params_M'],
                              FLOPs_G_x4=x4['algorithmic_GFLOPs'], FLOPs_G_x8=x8['algorithmic_GFLOPs'],
                              metrics=metrics,
                              note='architecture-only; Chikusei MHF training config/report not present' if dataset == 'chikusei' and model == 'MHF-Net' else ''))
    (OUT / 'table_evidence.json').write_text(json.dumps(dict(source_hashes=sources, metric_reports=evidence,
                                                           training_seed=42, test_seed=1, results=table), indent=2) + '\n')
    return table


def column_name(number):
    result = ''
    while number:
        number, rem = divmod(number-1, 26)
        result = chr(65+rem) + result
    return result


def sheet_xml(rows, widths, merges=(), header_rows=1, comparison=False):
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
             '<sheetViews><sheetView workbookViewId="0"><pane xSplit="2" ySplit="%d" topLeftCell="C%d" activePane="bottomRight" state="frozen"/></sheetView></sheetViews>' % (header_rows, header_rows+1),
             '<cols>']
    parts += [f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>' for i, width in enumerate(widths, 1)]
    parts += ['</cols><sheetData>']
    for ri, row in enumerate(rows, 1):
        lines = max([1] + [math.ceil(sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(value)) / widths[ci])
                           for ci, value in enumerate(row) if value is not None and isinstance(value, str)])
        height = max(26, 16 * lines + 8)
        parts.append(f'<row r="{ri}" ht="{height}" customHeight="1">')
        is_highlight = comparison and ri > header_rows and row[1] == 'V3-stage3'
        for ci, value in enumerate(row, 1):
            if value is None:
                continue
            address = f'{column_name(ci)}{ri}'
            if ri <= header_rows:
                style = 1
            elif isinstance(value, (int, float)):
                style = 6 if ci == 3 and is_highlight else 5 if ci == 3 else 4 if is_highlight else 3
            else:
                style = 2 if is_highlight else 0
            if isinstance(value, (int, float)):
                assert math.isfinite(value)
                parts.append(f'<c r="{address}" s="{style}"><v>{value}</v></c>')
            else:
                parts.append(f'<c r="{address}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>')
        parts.append('</row>')
    parts.append('</sheetData>')
    if merges:
        parts.append(f'<mergeCells count="{len(merges)}">')
        parts += [f'<mergeCell ref="{ref}"/>' for ref in merges]
        parts.append('</mergeCells>')
    parts.append('<pageMargins left="0.25" right="0.25" top="0.4" bottom="0.4" header="0.2" footer="0.2"/>')
    parts.append('</worksheet>')
    return ''.join(parts)


def write_workbook(table, notes):
    comparison = [
        ['Dataset', 'Methods', 'Params(M)', 'FLOPs(G) @×4', 'In-distribution', None, None, None, None, None],
        [None, None, None, None, '4×', None, None, '8×', None, None],
        [None, None, None, None, 'PSNR↑', 'SAM↓', 'ERGAS↓', 'PSNR↑', 'SAM↓', 'ERGAS↓'],
    ]
    comparison += [[r['display'], r['model'], r['params_M'], r['FLOPs_G_x4'], *r['metrics']] for r in table]
    merges = ['A1:A3', 'B1:B3', 'C1:C3', 'D1:D3', 'E1:J1', 'E2:G2', 'H2:J2', 'A4:A7', 'A8:A11', 'A12:A15']
    for start in (4, 8, 12):
        for row in range(start+1, start+4):
            comparison[row-1][0] = None
    detail = [['Dataset', 'Methods', 'Params(M)', 'FLOPs ×4 (G)', 'FLOPs ×8 (G)', '说明']]
    detail += [[r['display'], r['model'], r['params_M'], r['FLOPs_G_x4'], r['FLOPs_G_x8'], r['note']] for r in table]
    sheets = [('已核对比较表', comparison, [19, 19, 16, 20, 14, 14, 14, 14, 14, 14], merges, 3, True),
              ('复杂度明细', detail, [19, 21, 16, 20, 20, 75], [], 1, False),
              ('统计说明', [['项目', '说明']] + [[key, value] for key, value in notes], [28, 140], [], 1, False)]
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="2"><numFmt numFmtId="164" formatCode="0.0000"/><numFmt numFmtId="165" formatCode="0.000000"/></numFmts>
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFE4ECF7"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFFBE4D5"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left style="thin"><color rgb="FFD0D0D0"/></left><right style="thin"><color rgb="FFD0D0D0"/></right><top style="thin"><color rgb="FFD0D0D0"/></top><bottom style="thin"><color rgb="FFD0D0D0"/></bottom><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="7">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
<xf numFmtId="0" fontId="1" fillId="3" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
<xf numFmtId="164" fontId="1" fillId="3" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
<xf numFmtId="165" fontId="1" fillId="3" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    xmlns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    mainns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    relns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    overrides = ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets)+1))
    content = '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + overrides + '</Types>'
    workbook = f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{mainns}" xmlns:r="{relns}"><sheets>' + ''.join(f'<sheet name={quoteattr(name)} sheetId="{i}" r:id="rId{i}"/>' for i, (name, *_) in enumerate(sheets, 1)) + '</sheets><calcPr calcId="191029"/></workbook>'
    rels = f'<Relationships xmlns="{xmlns}">' + ''.join(f'<Relationship Id="rId{i}" Type="{relns}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets)+1)) + f'<Relationship Id="rId{len(sheets)+1}" Type="{relns}/styles" Target="styles.xml"/></Relationships>'
    with zipfile.ZipFile(OUT / 'comparison_completed.xlsx', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content)
        z.writestr('_rels/.rels', f'<Relationships xmlns="{xmlns}"><Relationship Id="rId1" Type="{relns}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', workbook)
        z.writestr('xl/_rels/workbook.xml.rels', rels)
        z.writestr('xl/styles.xml', styles)
        for i, (_, rows, widths, merges, headers, highlighted) in enumerate(sheets, 1):
            z.writestr(f'xl/worksheets/sheet{i}.xml', sheet_xml(rows, widths, merges, headers, highlighted))


def main():
    table = collect()
    notes = [
        ('统计日期', '2026-11-19'),
        ('主表FLOPs', '按×4统计；batch=1，HR=64×64，LR=16×16；单位G=10^9；一次乘加=2 FLOPs。×8另列于复杂度明细。'),
        ('通道配置', 'CAVE HSI/MSI=31/3；Chikusei=128/3；Pavia Centre=102/4。'),
        ('计数含义', '明确算法约定下的FLOPs：卷积、矩阵乘、注意力、归一化、激活、FFT、采样。FFT采用radix-2理论公式，特殊函数按单个算法操作计算，不是硬件指令精确数。'),
        ('省略项', '整数索引/几何、访存、随机置换、固定几何与OTF warm缓存预处理、普通插值内部坐标和系数生成；详见原RULES.md。'),
        ('原规则', str(OLD / 'RULES.md')),
        ('MHF新增规则', 'var_mean correction0每组k元素计4k；完整显式MomentNorm为8n+2C；常数padding计零；转置卷积按输入空间和完整实际计算计数，后续crop不扣减既有计算。'),
        ('MHF模型', '当前comparison中PyTorch作者CMHF翻译，rank12/stages20，forward推理路径；×8共享权重适配。按结构统计，无需训练或读取checkpoint。'),
        ('Chikusei MHF', '截至生成表格时未发现该组训练配置/测试报告。复杂度按128/3输入及相同ComparisonMHF结构得到，不代表该组已训练完成。'),
        ('参数量', '精确注册参数总量，包括注册但未启用的参数；V3-stage3前两阶段131128个可变形/门控参数未参与前向。'),
        ('截图更正1', '截图SSR-Net与V3-stage3的Params(M)放反；CAVE分别应为0.026041与0.614036。'),
        ('截图更正2', '截图中标Chikusei的数据块对应Pavia；标Pavia Center的数据块对应Chikusei。新主表按真实数据集归位。'),
        ('截图更正3', 'Chikusei SFNO seed42 ×4 PSNR：原报告43.7923866839，四位小数43.7924；截图43.7942疑为录入错误，新表按报告填入。'),
        ('性能来源', '主表性能从训练seed42权重对应的原测试JSON读取，test seed=1；CAVE为历史连续600轮协议，Pavia/Chikusei为当前comparison协议。缺失的MHF遥感性能保持空白。'),
        ('论文SFNO值', '论文报告CAVE61.20G、Chikusei468.62G、Pavia331.85G；统计尺寸/工具/倍率/乘加口径未确认，且作者序列化模型和本地源码存在差异。仅供引用，不可与本表本地测量混用。'),
        ('文件处理', '输入只有截图，生成新Excel/CSV，未覆盖用户原表、既有复杂度文件、训练代码、权重或测试报告。'),
    ]
    headers = ['Dataset', 'Methods', 'Params(M)', 'FLOPs(G) @x4', 'PSNR_x4', 'SAM_x4', 'ERGAS_x4', 'PSNR_x8', 'SAM_x8', 'ERGAS_x8']
    with (OUT / 'comparison_completed.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in table:
            writer.writerow([r['display'], r['model'], f"{r['params_M']:.6f}", f"{r['FLOPs_G_x4']:.9f}",
                             *['' if x is None else f'{x:.4f}' for x in r['metrics']]])
    with (OUT / 'complexity_all.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Methods', 'Params', 'Params(M)', 'FLOPs_x4(G)', 'FLOPs_x8(G)', 'Notes'])
        for r in table:
            writer.writerow([r['display'], r['model'], r['params'], f"{r['params_M']:.6f}",
                             f"{r['FLOPs_G_x4']:.9f}", f"{r['FLOPs_G_x8']:.9f}", r['note']])
    lines = ['# 比较表复杂度补齐与核对（2026-11-19）', '',
             '主表采用当前实现统一测量的×4 FLOPs：batch=1、HR64×64、LR16×16、乘加=2。×8为HR64×64、LR8×8。', '',
             '| Dataset | Methods | Params(M) | FLOPs ×4(G) | FLOPs ×8(G) |',
             '|---|---|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['display']} | {r['model']} | {r['params_M']:.6f} | {r['FLOPs_G_x4']:.4f} | {r['FLOPs_G_x8']:.4f} |")
    lines += ['', '## 统计说明与截图更正', '']
    lines += [f'- **{key}**：{value}' for key, value in notes]
    lines += ['', '## 文件', '',
              '- [comparison_completed.xlsx](comparison_completed.xlsx)：三个工作表，包含与截图同结构的已核对比较表、双倍率复杂度和说明。',
              '- [comparison_completed.csv](comparison_completed.csv)：含性能的主表；[complexity_all.csv](complexity_all.csv)：双倍率复杂度精确值。',
              '- [profile_mhf.py](profile_mhf.py)、[mhf_details.json](mhf_details.json)：MHF测量与算子分项，所有观测算子均有计数或计零规则。',
              '- [table_evidence.json](table_evidence.json)：原统计源文件与测试报告哈希；已有9个模型checkpoint哈希也已与原测量记录核对。',
              '- [build_tables.py](build_tables.py)：可复核表格生成脚本。', '',
              '主表FLOPs为明确规则下的算法计数，不能声称与未知统计协议的论文数值完全同口径。MHF只做CPU前向，未启动、暂停或修改任何训练。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    write_workbook(table, notes)
    print(OUT / 'comparison_completed.xlsx')
    print('\n'.join(lines[:18]))


if __name__ == '__main__':
    main()
