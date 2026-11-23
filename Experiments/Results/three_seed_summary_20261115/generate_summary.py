"""Aggregate 36 final CAVE/Pavia/Chikusei reports; write CSV, Markdown and XLSX."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENTS = PROJECT / 'Experiments'
DATASETS = ('cave', 'pavia', 'chikusei')
LABELS = {'cave': 'CAVE', 'pavia': 'Pavia', 'chikusei': 'Chikusei'}
MODELS = ('sfno', 'v3_stage3')
MODEL_LABELS = {'sfno': 'baseline', 'v3_stage3': 'V3-stage3'}
KEYS = ('psnr', 'sam', 'ergas')
SEEDS = (42, 43, 44)
S = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ET.register_namespace('', S)
ET.register_namespace('r', R)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def cell_name(column, row):
    letters = ''
    while column:
        column, remainder = divmod(column - 1, 26)
        letters = chr(65 + remainder) + letters
    return f'{letters}{row}'


def xml_bytes(element):
    return ET.tostring(element, encoding='utf-8', xml_declaration=True)


def worksheet(rows, widths, merges=(), freeze=0, filter_rows=False):
    sheet = ET.Element(f'{{{S}}}worksheet')
    ET.SubElement(sheet, f'{{{S}}}sheetPr').append(ET.Element(f'{{{S}}}pageSetUpPr', fitToPage='1'))
    ET.SubElement(sheet, f'{{{S}}}dimension', ref=f'A1:{cell_name(max(map(len, rows)), len(rows))}')
    views = ET.SubElement(sheet, f'{{{S}}}sheetViews')
    view = ET.SubElement(views, f'{{{S}}}sheetView', workbookViewId='0', showGridLines='0')
    if freeze:
        ET.SubElement(view, f'{{{S}}}pane', ySplit=str(freeze), topLeftCell=f'A{freeze+1}', activePane='bottomLeft', state='frozen')
    ET.SubElement(sheet, f'{{{S}}}sheetFormatPr', defaultRowHeight='22')
    cols = ET.SubElement(sheet, f'{{{S}}}cols')
    for index, width in enumerate(widths, 1):
        ET.SubElement(cols, f'{{{S}}}col', min=str(index), max=str(index), width=str(width), customWidth='1')
    data = ET.SubElement(sheet, f'{{{S}}}sheetData')
    for row_index, cells in enumerate(rows, 1):
        row = ET.SubElement(data, f'{{{S}}}row', r=str(row_index))
        if not cells:
            row.set('ht', '9')
            row.set('customHeight', '1')
        for col_index, (value, style) in enumerate(cells, 1):
            if value is None:
                continue
            cell = ET.SubElement(row, f'{{{S}}}c', r=cell_name(col_index, row_index), s=str(style))
            if isinstance(value, (int, float)):
                assert math.isfinite(value)
                ET.SubElement(cell, f'{{{S}}}v').text = repr(value)
            else:
                cell.set('t', 'inlineStr')
                string = ET.SubElement(cell, f'{{{S}}}is')
                ET.SubElement(string, f'{{{S}}}t').text = str(value)
    if filter_rows:
        ET.SubElement(sheet, f'{{{S}}}autoFilter', ref=f'A1:{cell_name(len(rows[0]),len(rows))}')
    if merges:
        merged = ET.SubElement(sheet, f'{{{S}}}mergeCells', count=str(len(merges)))
        for address in merges:
            ET.SubElement(merged, f'{{{S}}}mergeCell', ref=address)
    ET.SubElement(sheet, f'{{{S}}}pageMargins', left='.25', right='.25', top='.4', bottom='.4', header='.2', footer='.2')
    ET.SubElement(sheet, f'{{{S}}}pageSetup', orientation='landscape', paperSize='9', fitToWidth='1', fitToHeight='0')
    return xml_bytes(sheet)


STYLES = '''<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000000000"/></numFmts>
<fonts count="3"><font><sz val="11"/><name val="Microsoft YaHei"/></font><font><b/><sz val="11"/><name val="Microsoft YaHei"/></font><font><sz val="10"/><color rgb="FF666666"/><name val="Microsoft YaHei"/></font></fonts>
<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
<borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left style="thin"><color rgb="FFD9D9D9"/></left><right style="thin"><color rgb="FFD9D9D9"/></right><top style="thin"><color rgb="FFD9D9D9"/></top><bottom style="thin"><color rgb="FFD9D9D9"/></bottom><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="6"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf><xf numFmtId="0" fontId="1" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf><xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def write_xlsx(path, sheets):
    rel_ns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    workbook = ET.Element(f'{{{S}}}workbook')
    sheet_list = ET.SubElement(workbook, f'{{{S}}}sheets')
    relationships = ET.Element('Relationships', xmlns=rel_ns)
    overrides = []
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('_rels/.rels', f'<Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="{R}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        for index, (name, rows, widths, merges, freeze, filters) in enumerate(sheets, 1):
            ET.SubElement(sheet_list, f'{{{S}}}sheet', name=name, sheetId=str(index), attrib={f'{{{R}}}id': f'rId{index}'})
            ET.SubElement(relationships, 'Relationship', Id=f'rId{index}', Type=f'{R}/worksheet', Target=f'worksheets/sheet{index}.xml')
            archive.writestr(f'xl/worksheets/sheet{index}.xml', worksheet(rows, widths, merges, freeze, filters))
            overrides.append(f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        ET.SubElement(relationships, 'Relationship', Id='styles', Type=f'{R}/styles', Target='styles.xml')
        archive.writestr('xl/workbook.xml', xml_bytes(workbook))
        archive.writestr('xl/_rels/workbook.xml.rels', xml_bytes(relationships))
        archive.writestr('xl/styles.xml', STYLES)
        archive.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + ''.join(overrides) + '</Types>')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    filenames = ['summary.md', 'summary.csv', 'per_seed.csv', 'statistics.csv', 'summary.xlsx', 'provenance.json']
    assert not any((output / name).exists() for name in filenames), 'Choose a new output directory'
    raw, source_records = [], []
    checkpoint_digests = {}
    native_migration = json.loads((EXPERIMENTS/'cave_continuous600/migration_manifest.json').read_text())
    native_hashes = {entry['new']: entry['sha256'] for entry in native_migration['files']}
    for dataset in DATASETS:
        for model in MODELS:
            exp = 'cave_continuous600' if dataset == 'cave' else ('sfno_restart_20261111' if model == 'sfno' else 'v3_stage3_restart_20261111')
            root = EXPERIMENTS / exp
            protocol_sha = sha(root/'protocol.json') if dataset != 'cave' else None
            for seed in SEEDS:
                folder = root/'Checkpoint'/f'{dataset}_{model}_x4_seed{seed}'
                checkpoint = folder/'last.pth'
                checkpoint_digests[str(checkpoint.relative_to(PROJECT))] = sha(checkpoint)
                config = json.loads((folder/'config.json').read_text())
                assert config['seed'] == seed and config['epochs'] == 600
                if dataset == 'cave':
                    assert config['sf'] == 4
                    assert checkpoint_digests[str(checkpoint.relative_to(PROJECT))] == native_hashes[str(checkpoint.relative_to(PROJECT))]
                else:
                    complete = json.loads((folder/'complete.json').read_text())
                    assert complete['epoch'] == 600 and complete['scales'] == [4, 8]
                for sf in (4, 8):
                    report_path = folder / (('metrics.json' if sf == 4 else 'metrics_x8.json') if dataset == 'cave' else f'last_test_x{sf}.json')
                    report = json.loads(report_path.read_text())
                    assert report['epoch'] == 600 and report['source_sf'] == 4 and report['sf'] == sf
                    assert len(report['per_scene'] if dataset == 'cave' else report['patches']) == (12 if dataset == 'cave' else (4 if dataset == 'pavia' else 8))
                    if dataset == 'cave':
                        assert report['seed'] == 1
                        assert report['model'] == ('sfno' if model == 'sfno' else 'sfno_prgno_v3')
                        sam_key = 'sam'
                    else:
                        assert report['seed'] == seed and report['test_seed'] == 1
                        assert report['dataset'] == dataset and report['model'] == model
                        assert report['protocol_sha256'] == protocol_sha
                        assert report['checkpoint_sha256'] == checkpoint_digests[str(checkpoint.relative_to(PROJECT))]
                        sam_key = 'sam_degrees'
                    if model == 'v3_stage3':
                        assert report['ablation'] == 'deform_stage3'
                    values = dict(psnr=report['mean']['psnr'], sam=report['mean'][sam_key], ergas=report['mean']['ergas'])
                    assert all(math.isfinite(v) for v in values.values())
                    raw.append(dict(dataset=LABELS[dataset], model=MODEL_LABELS[model], train_seed=seed, test_seed=1, epoch=600, train_scale=4, test_scale=sf, **values, source=str(report_path.relative_to(PROJECT))))
                    source_records.append(dict(path=str(report_path.relative_to(PROJECT)), sha256=sha(report_path), protocol_sha256=protocol_sha))
    assert len(raw) == 36
    groups = {}
    for dataset in DATASETS:
        for model in MODELS:
            for sf in (4, 8):
                rows = [r for r in raw if r['dataset'] == LABELS[dataset] and r['model'] == MODEL_LABELS[model] and r['test_scale'] == sf]
                assert sorted(r['train_seed'] for r in rows) == list(SEEDS)
                group = dict(dataset=LABELS[dataset], model=MODEL_LABELS[model], sf=sf, n=3)
                for key in KEYS:
                    values = [row[key] for row in rows]
                    group[key+'_mean'] = statistics.mean(values)
                    group[key+'_std'] = statistics.stdev(values)
                groups[dataset, model, sf] = group
    # Compare independently recomputed values with both existing summaries.
    with (EXPERIMENTS/'cave_continuous600/Summary/mean_std.csv').open() as stream:
        old_cave = list(csv.DictReader(stream))
    queue = json.loads((EXPERIMENTS/'sfno_v3_queue_20261111/summary.json').read_text())
    assert queue['completed_reports'] == queue['required_reports'] == 24
    for (dataset, model, sf), group in groups.items():
        if dataset == 'cave':
            old = next(r for r in old_cave if r['model'] == ('baseline' if model == 'sfno' else 'deform_stage3') and int(r['sf']) == sf)
            assert all(abs(group[key+suffix]-float(old[key+suffix])) < 1e-12 for key in KEYS for suffix in ['_mean','_std'])
        else:
            old = next(r for r in queue['groups'] if r['dataset'] == dataset and r['model'] == model and r['sf'] == sf)
            for key in KEYS:
                source_key = 'sam_degrees' if key == 'sam' else key
                assert abs(group[key+'_mean']-old['mean'][source_key]) < 1e-12
                assert abs(group[key+'_std']-old['sample_std'][source_key]) < 1e-12

    def text(dataset, model, sf, key):
        group = groups[dataset, model, sf]
        return f"{group[key+'_mean']:.4f} ± {group[key+'_std']:.4f}"

    formatted, display_rows, merges = [], [], []
    markdown = ['训练种子42、43、44；测试种子1。均为最终600轮权重，×4训练、×4/×8测试。每个种子先使用其测试场景/图块的平均指标，再对三个种子计算均值和样本标准差（ddof=1）；以下保留四位小数。','']
    for dataset in DATASETS:
        start = len(display_rows)+1
        display_rows += [[('随机种子',1),(f'{LABELS[dataset]}(×4)',1),(None,1),(None,1),(f'{LABELS[dataset]}(×8)',1),(None,1),(None,1)], [('模型',1),('PSNR ↑',1),('SAM ↓',1),('ERGAS ↓',1),('PSNR ↑',1),('SAM ↓',1),('ERGAS ↓',1)]]
        merges += [f'B{start}:D{start}',f'E{start}:G{start}']
        markdown += [f'**{LABELS[dataset]}**','', '| 模型 | ×4 PSNR ↑ | ×4 SAM ↓ | ×4 ERGAS ↓ | ×8 PSNR ↑ | ×8 SAM ↓ | ×8 ERGAS ↓ |','|---|---:|---:|---:|---:|---:|---:|']
        for model in MODELS:
            row = dict(dataset=LABELS[dataset],model=MODEL_LABELS[model])
            cells, md_values = [(MODEL_LABELS[model],2)], []
            for sf in (4,8):
                for key in KEYS:
                    value = text(dataset,model,sf,key)
                    means = [groups[dataset,m,sf][key+'_mean'] for m in MODELS]
                    best = (max(means) if key == 'psnr' else min(means))
                    winner = groups[dataset,model,sf][key+'_mean'] == best
                    row[f'x{sf}_{key}'] = value
                    cells.append((value,3 if winner else 2))
                    md_values.append(f'**{value}**' if winner else value)
            display_rows.append(cells)
            formatted.append(row)
            markdown.append('| '+' | '.join([MODEL_LABELS[model],*md_values])+' |')
        display_rows += [[],[]]
        markdown.append('')
    for note in ['训练种子：42、43、44；测试种子：1；每组600轮。','均值 ± 样本标准差（ddof=1），保留四位小数。','加粗仅表示同数据集、同尺度下的均值更优。各数据集沿用各自实验评价协议。']:
        display_rows.append([(note,4)])
        merges.append(f'A{len(display_rows)}:G{len(display_rows)}')
    markdown += ['加粗仅表示同数据集、同尺度下的均值更优。各数据集沿用各自实验评价协议。','', 'Excel：summary.xlsx；逐种子原始数值及报告路径：per_seed.csv；未舍入统计：statistics.csv。']
    (output/'summary.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    for name, rows in [('summary.csv',formatted),('per_seed.csv',raw),('statistics.csv',list(groups.values()))]:
        with (output/name).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    raw_headers=['数据集','模型','训练种子','测试种子','训练轮数','训练尺度','测试尺度','PSNR','SAM','ERGAS','报告路径']
    raw_rows=[[(h,1) for h in raw_headers]]+ [[(v,5 if k in KEYS else 2) for k,v in row.items()] for row in raw]
    number_headers=['数据集','模型','测试尺度','种子数','PSNR均值','PSNR样本标准差','SAM均值','SAM样本标准差','ERGAS均值','ERGAS样本标准差']
    number_rows=[[(h,1) for h in number_headers]]+ [[(v,5 if k.endswith(('_mean','_std')) else 2) for k,v in row.items()] for row in groups.values()]
    xlsx=output/'summary.xlsx'
    write_xlsx(xlsx,[('随机种子汇总',display_rows,[19,25,25,25,25,25,25],merges,0,False),('逐种子明细',raw_rows,[16,18,12,12,12,12,12,20,20,20,105],[],1,True),('未舍入统计',number_rows,[16,18,12,12,20,20,20,20,20,20],[],1,True)])
    with zipfile.ZipFile(xlsx) as archive:
        assert archive.testzip() is None
        for member in archive.namelist():
            if member.endswith(('.xml','.rels')):
                ET.fromstring(archive.read(member))
        xml=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        strings={c.attrib['r']:c.find(f'{{{S}}}is/{{{S}}}t').text for c in xml.findall(f'.//{{{S}}}c') if c.attrib.get('t') == 'inlineStr'}
        assert strings['B3']=='51.8920 ± 0.0145' and strings['B4']=='52.0682 ± 0.0460'
        assert strings['E3']=='49.5444 ± 0.0478' and strings['G4']=='0.5253 ± 0.0052'
        for row_index,row in enumerate(display_rows,1):
            for col_index,(value,_) in enumerate(row,1):
                if value is not None:assert strings[cell_name(col_index,row_index)]==str(value)
    provenance=dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),train_seeds=list(SEEDS),test_seed=1,epochs=600,train_scale=4,test_scales=[4,8],report_count=36,experiment_count=18,std='sample standard deviation across three training seeds, ddof=1',rounding='4 decimal places only for presentation; statistics computed from unrounded report means',sources=source_records,checkpoint_sha256=checkpoint_digests,verification=dict(report_metadata_and_checkpoint_hashes=True,all_groups_have_exactly_three_seeds=True,existing_independent_summaries_agree=True,xlsx_zip_and_xml_valid=True,xlsx_cells_match_generated_tables=True,cave_values_match_user_screenshot=True),outputs={name:sha(output/name) for name in filenames if name!='provenance.json'})
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print((output/'summary.md').read_text())
    print('Saved:',output)


if __name__=='__main__':
    main()
