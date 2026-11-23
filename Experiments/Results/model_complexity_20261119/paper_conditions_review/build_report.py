"""Package actual test-geometry profiles without claiming the paper protocol."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[3]
PARENT = OUT.parent
names = {'cave': 'CAVE', 'chikusei': 'Chikusei', 'pavia': 'Pavia Centre'}
labels = {'author': 'SFNO 作者TorchScript', 'sfno': 'SFNO 本地复现', 'v3_stage3': 'V3-stage3 本地'}
paper = {'cave': (0.32, 61.20), 'chikusei': (1.57, 468.62), 'pavia': (1.34, 331.85)}


def read(dataset, model, size):
    row = json.loads((OUT / f'{dataset}_{model}_hr{size}.json').read_text())
    assert not row['unclassified_ops'] and not row['paper_Table1_conditions_confirmed']
    with Path(row['checkpoint']).open('rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == row['checkpoint_sha256']
    return row


def workbook(rows, notes, small):
    spec = importlib.util.spec_from_file_location('table_format_helpers', PARENT / 'build_tables.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sheets = [
        ('按测试尺寸实测', rows, [19, 28, 17, 17, 16, 21, 23], 1),
        ('CAVE统一64对照', small, [19, 28, 17, 21, 23], 1),
        ('条件与限制', [['项目', '说明']] + notes, [30, 145], 1),
    ]
    with zipfile.ZipFile(PARENT / 'comparison_completed.xlsx') as z:
        styles = z.read('xl/styles.xml')
    prefix = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    sns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    pns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    cns = 'http://schemas.openxmlformats.org/package/2006/content-types'
    with zipfile.ZipFile(OUT / 'test_geometry_profiles.xlsx', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', prefix + f'<Types xmlns="{cns}"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in (1, 2, 3)) + '</Types>')
        z.writestr('_rels/.rels', prefix + f'<Relationships xmlns="{pns}"><Relationship Id="rId1" Type="{rns}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', prefix + f'<workbook xmlns="{sns}" xmlns:r="{rns}"><sheets>' + ''.join(f'<sheet name={quoteattr(name)} sheetId="{i}" r:id="rId{i}"/>' for i, (name, *_) in enumerate(sheets, 1)) + '</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', prefix + f'<Relationships xmlns="{pns}">' + ''.join(f'<Relationship Id="rId{i}" Type="{rns}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in (1, 2, 3)) + f'<Relationship Id="rId4" Type="{rns}/styles" Target="styles.xml"/></Relationships>')
        z.writestr('xl/styles.xml', styles)
        for i, (_, content, widths, headers) in enumerate(sheets, 1):
            xml = helper.sheet_xml(content, widths, header_rows=headers)
            if i == 1:
                for row_number in range(2, len(content)+1):
                    xml = xml.replace(f'<c r="E{row_number}" s="3">', f'<c r="E{row_number}" s="5">')
            z.writestr(f'xl/worksheets/sheet{i}.xml', xml)


def main():
    profiles = [read('cave', 'author', 512)]
    profiles += [read(dataset, model, size) for dataset, size in [('cave', 512), ('chikusei', 500), ('pavia', 192)]
                 for model in ('sfno', 'v3_stage3')]
    author64 = read('cave', 'author', 64)
    assert author64['params'] == 318245
    assert author64['core_FLOPs'] == 2532376576 and author64['additional_FLOPs'] == 154277376
    notes = [
        ['严格同口径结论', '无法确认论文Table1 FLOPs的完整统计条件。论文全文和作者全部20个公开提交均未找到全模型profile入口；用户确认没有额外作者说明。因此本次绝不标作已复现论文61.20/468.62/331.85G。'],
        ['本次完成', '8组实际CPU前向计数：作者CAVE模型64/512；本地SFNO/V3分别CAVE512、Chikusei500、Pavia192；没有按面积外推。'],
        ['作者CAVE来源', '官方BestModel_ts.pt，318245个注册参数；作者Test.py默认及导出trace示例均为batch1、HR512、LR128、31/3通道、×4。默认测试/trace输入不是Table1 FLOPs输入的证明。'],
        ['其他两数据集', '图像尺寸取自论文§4.1测试设置；模型采用当前本地实验：Chikusei128/3，Pavia102/4。作者没有公开这两组网络/权重，不能确认与论文配置相同。'],
        ['计数口径', 'batch1、×4、乘加=2；core为PyTorch支持的卷积/矩阵乘，algorithmic为core加明确的其他算法规则。FFT/插值等是理论成本约定，不是实际机器指令。'],
        ['FFT限制', '沿用5*N*log2(N)理论成本；192和500不是2幂，不能声称该公式精确等于FFT库内部执行计数。'],
        ['大图内存优化', '本地模型仅对eval VectorAttention使用既有32行分块实现，保持数学计算，避免高波段大图激活峰值；本次统计包含实际执行的分块路径。作者TorchScript保持原实现。'],
        ['计数器补充', 'TorchScript使用aten._convolution，已补bias加法；equal布尔比较计0；floor_divide仅允许整数shape几何并计0；常数padding计0。8组未分类算子均为空。'],
        ['模型区别', '作者TS PR窗口8/4/2且无随机排列；当前公开/本地SFNO PR窗口4/4/2且有随机排列；作者ADE没有当前源码额外的+x。'],
        ['未启用参数', '作者TS注册但未在顶层前向调用的7个模块共73600参数，解释其0.318245M与源码0.244645M之差；不构成这些未调用模块的前向FLOPs。'],
        ['论文对齐程度', 'CAVE作者参数0.318245M可四舍五入得到论文0.32M；论文Chikusei1.57M/Pavia1.34M因作者配置未公开不能独立重现。作者512实测core162.0721G、algorithmic171.9458G均不等于61.20G。'],
        ['工作区处理', '只读权重并核对前后SHA256；不修改模型/训练配置/已有性能报告，也不启动或暂停训练。此前HR64统一比较表保持不变。'],
    ]
    headings = ['Dataset', 'Methods', 'HR', 'LR', 'Params(M)', 'Core FLOPs(G)', 'Algorithmic FLOPs(G)']
    rows = [headings] + [[names[r['dataset']], labels[r['model']], '×'.join(map(str, r['HR'])),
                         '×'.join(map(str, r['LR'])), r['params_M'], r['core_GFLOPs'], r['algorithmic_GFLOPs']] for r in profiles]
    with (OUT / 'test_geometry_profiles.csv').open('w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(rows)
    old = json.loads((PROJECT / 'Experiments/Results/model_complexity_20261116/complete_rules/details.json').read_text())
    small = [['Dataset', 'Methods', 'Params(M)', 'Core FLOPs(G)', 'Algorithmic FLOPs(G)'],
             ['CAVE', labels['author'], author64['params_M'], author64['core_GFLOPs'], author64['algorithmic_GFLOPs']]]
    for name, label in [('SFNO', labels['sfno']), ('SPGFNet (V3-stage3)', labels['v3_stage3'])]:
        r = next(r for r in old if r['dataset'] == 'cave' and r['model'] == name and r['scale'] == 4)
        small.append(['CAVE', label, r['params_M'], r['core_GFLOPs'], r['algorithmic_GFLOPs']])
    workbook(rows, notes, small)
    text = ['# 论文复杂度条件复核与测试尺寸实测（2026-11-19）', '',
            '**尚不能声称按论文Table1完全同口径复测。** 本次已检查论文全文、作者仓库全部20个提交及预训练模型；作者没有公开全模型统计脚本、Table1输入几何、计数工具/MAC口径及遥感模型配置，用户也没有额外作者说明。', '',
            '已完成可核实的工作：作者CAVE参数可重现0.32M；8组CPU实际前向计数、无未分类算子；按作者默认CAVE测试几何和论文遥感测试图像尺寸得到如下结果。**这些测试图像尺寸不等于已确认的Table1 FLOPs输入。**', '',
            '| 数据集 | 模型来源 | HR / LR（×4） | Params(M) | 核心FLOPs(G) | 约定下算法FLOPs(G) |',
            '|---|---|---|---:|---:|---:|']
    for r in profiles:
        text.append(f"| {names[r['dataset']]} | {labels[r['model']]} | {r['HR'][0]} / {r['LR'][0]} | {r['params_M']:.6f} | {r['core_GFLOPs']:.4f} | {r['algorithmic_GFLOPs']:.4f} |")
    text += ['', 'HR/LR均为正方形，batch1；core采用乘加2，algorithmic添加归一化、激活、FFT、插值等明确规则。FFT及插值部分是算法估算约定。', '',
             '## 与论文数值的边界', '',
             '| 数据集 | 论文Params(M) | 论文FLOPs(G) | 本次确认程度 |',
             '|---|---:|---:|---|',
             '| CAVE | 0.32 | 61.20 | 作者模型注册参数0.318245M可重现；FLOPs完整统计口径无法确认 |',
             '| Chikusei | 1.57 | 468.62 | 作者该数据集模型/权重未公开，Params/FLOPs均不能声称已对齐 |',
             '| Pavia Centre | 1.34 | 331.85 | 作者该数据集模型/权重未公开，Params/FLOPs均不能声称已对齐 |', '',
             '作者CAVE模型512输入下core162.072100864G，即只对卷积/矩阵乘换算为1 MAC也为81.036050432 GMACs，仍不是61.20。因此不能只把输入改512或只除2就宣称复现论文。其他算术操作也不应统一除2。', '',
             '## 统一64输入的实际对照', '',
             '| 模型 | Params(M) | 核心FLOPs(G) | 约定下算法FLOPs(G) |',
             '|---|---:|---:|---:|']
    for row in small[1:]:
        text.append(f'| {row[1]} | {row[2]:.6f} | {row[3]:.4f} | {row[4]:.4f} |')
    text += ['', '作者模型与当前源码存在PR窗口、排列和ADE残差差别，不能将它们视为完全同一实现。未调用参数只影响注册参数统计，不产生对应前向FLOPs。', '',
             '## 条件、证据与限制', '']
    text += [f'- **{key}**：{value}' for key, value in notes]
    text += ['', '## 来源', '',
             '- [原论文](../../../../Li%20等%20-%202026%20-%20Arbitrary-scale%20spatial-spectral%20fusion%20using%20kernel%20integral%20and%20progressive%20resampling.pdf)：§4.1测试尺寸、Table1复杂度及Table5消融。',
             '- [作者测试入口固定版本](https://github.com/weili419/SFNO/blob/156fecb52c230e554d46e9c3be8533263fa9eb75/CAVE/Test.py#L15)：默认512、batch1、×4；不是复杂度脚本。',
             '- [官方历史核查证据](sfno_author_history_20261119_evidence.json)：20提交、91历史路径、23文本版本，未发现完整profile入口或遥感模型。',
             '- [作者模型独立64核验](sfno_author_ts64_review.json)：含实际forward代码、PR窗口、参数及算子分项。',
             '- [既有明确计数规则](../../model_complexity_20261116/complete_rules/RULES.md)；本次新增TorchScript packet处理见[profile_geometry.py](profile_geometry.py)。', '',
             '## 文件', '',
             '- [test_geometry_profiles.xlsx](test_geometry_profiles.xlsx)：三个工作表，测试尺寸实测、64控制组、条件与限制。',
             '- [test_geometry_profiles.csv](test_geometry_profiles.csv)：本次实际大图数值。',
             '- 各`*_hr*.json`：每次前向的形状、权重哈希、完整算子分项和未分类清单。',
             '- [profile_geometry.py](profile_geometry.py)：可复跑实际前向；[build_report.py](build_report.py)：本报告与工作簿生成。', '',
             '此前的HR64统一算法计数表仍可用于明确条件下的本地模型比较；本次实测不是对该表的覆盖，也不应替换论文引用值后声称全部为论文同口径。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(text))
    status = dict(exact_paper_FLOPs_replication='not_established_missing_author_protocol',
                  author_CAVE_Params_reproduced=318245, measured_forward_profiles=8,
                  all_unclassified_empty=True, user_has_additional_author_material=False,
                  reviewed_upstream_commits=20, measured_results=profiles,
                  controlled_author64=author64)
    (OUT / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print('\n'.join(text[:18]))
    print(OUT / 'test_geometry_profiles.xlsx')


if __name__ == '__main__':
    main()
