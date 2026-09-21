"""Write a Chinese summary from the validated, exported measurements."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'summary.json').read_text())
manifest = json.loads((ROOT / 'manifest.json').read_text())
assert manifest['independent_builds_per_variant'] == 1
builds = {r['variant']: r for r in data['builds']}
points = {(r['variant'], r['threads'], r['k'], r['ef']): r for r in data['curves']}
original, current = builds['upstream'], builds['current']
ratios = {
    threads: [row['qps'] / points['upstream', threads, row['k'], row['ef']]['qps']
              for row in data['curves'] if row['variant'] == 'current' and row['threads'] == threads]
    for threads in [1, 16]
}
lines = [
    '# HVS 与原版 HVS：完整 SIFT1M 性能对比',
    '',
    '测试日期：2026-09-21。按要求，每个版本只构建一次索引。',
    '',
    '**历史性能测试配置：16 个物理核心、单 NUMA 插槽、未启用 interleave。** '
    '本次测试早于“系统最大线程数 + NUMA interleave”的要求；后续 64 线程一致性审计单独记录，'
    '不能将以下性能数据视为 64 线程测试结果。',
    '',
    f"本次构建 API 时间减少 {(1-current['build_api_s']/original['build_api_s'])*100:.2f}%。"
    f"相同 k/ef 下，单线程 QPS 在 {sum(r>1 for r in ratios[1])}/33 个点上更高，差异范围为 "
    f"{(min(ratios[1])-1)*100:+.2f}% 至 {(max(ratios[1])-1)*100:+.2f}%；"
    f"16 线程 QPS 在 {sum(r>1 for r in ratios[16])}/33 个点上更高，差异范围为 "
    f"{(min(ratios[16])-1)*100:+.2f}% 至 {(max(ratios[16])-1)*100:+.2f}%。"
    '因此，本次结果支持构建耗时减少和单线程小幅改善，但不支持并发吞吐全面提升。',
    '',
    '数据为完整 SIFT1M：1,000,000 条 128 维基向量、全部 10,000 条查询。'
    '两版均使用 T=1、M=16、efConstruction=500、delta=0.5、10,000 个训练样本和种子 100。'
    '图构建开启 16 线程；查询分别测试 1 线程和 16 线程并发。CPU 绑定到同一插槽的 16 个物理核心，OpenCV 内部使用 1 线程。',
    '',
    f"当前 HVS：`{manifest['current_commit']}`；原版：`{manifest['upstream_commit']}`。"
    '两版均使用 GCC 14.3、相同的 -O3 优化设置，以及完全相同的 OpenCV Core / Eigen / zlib 静态依赖。',
    '',
    '## 构建时间',
    '',
    '| 计时范围 | 原版 HVS | 当前 HVS | 当前 / 原版 |',
    '| --- | ---: | ---: | ---: |',
]
for key, label in [('build_api_s', '构建 API'), ('ready_s', '从输入到索引可查询')]:
    lines.append(f"| {label} | {original[key]:.3f} s | {current[key]:.3f} s | {current[key]/original[key]:.3f}× |")
lines += [
    '',
    '原版构建 API 包含其原生文件读写；当前 API 为内存构建。第二行还计入当前版本的基向量读取，'
    '以及原版在新进程中载入索引和查询元数据的时间。两版均预热数据文件缓存。'
    '这比较的是实际集成流程，不是剥离 I/O 后的纯构建内核。',
    '两版的构建 API 时间均包含量化器训练和全量向量编码。',
    '',
    '## 相同查询参数下的实测结果',
    '',
    '| k | ef | 原版 recall | 当前 recall | 原版单线程 QPS | 当前单线程 QPS | 原版 16 线程 QPS | 当前 16 线程 QPS |',
    '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for k, ef in [(1, 12), (1, 96), (10, 16), (10, 24), (10, 96), (10, 128), (100, 128), (100, 256)]:
    u1, c1 = points['upstream', 1, k, ef], points['current', 1, k, ef]
    u16, c16 = points['upstream', 16, k, ef], points['current', 16, k, ef]
    lines.append(f"| {k} | {ef} | {u1['recall']:.4%} | {c1['recall']:.4%} | {u1['qps']:,.0f} | {c1['qps']:,.0f} | {u16['qps']:,.0f} | {c16['qps']:,.0f} |")
differences = [
    (abs(row['recall'] - points['upstream', 1, row['k'], row['ef']]['recall']), row['k'], row['ef'])
    for row in data['curves'] if row['variant'] == 'current' and row['threads'] == 1
]
maximum_difference, maximum_k, maximum_ef = max(differences)
lines += [
    '',
    f'在本次 33 组相同 k/ef 参数中，两版 recall 的最大绝对差为 {maximum_difference * 100:.4f} 个百分点'
    f'（k={maximum_k}、ef={maximum_ef}）。',
]
lines += [
    '',
    '## 相同目标 recall 下的曲线比较',
    '',
    '以下 QPS 在实测曲线点之间对 log(QPS) 做线性插值，便于对齐 recall；'
    '它们是估算值，不是新增测量点。仅列出两版实测曲线共同覆盖的目标。',
    '',
    '| 查询线程 | k | 目标 recall | 原版估算 QPS | 当前估算 QPS | 当前 / 原版 QPS |',
    '| ---: | ---: | ---: | ---: | ---: | ---: |',
]
for row in data['matched_recall']:
    lines.append(f"| {row['threads']} | {row['k']} | {row['target_recall']:.0%} | {row['upstream_qps']:,.0f} | {row['current_qps']:,.0f} | {row['speedup']:.3f}× |")
lines += [
    '',
    '## 验证和适用范围',
    '',
    '- 两版各完成 66 个参数配置，共 132 个计时点。每个配置预热 1,000 条查询，再计时全部 10,000 条查询；'
    '若一批不足 0.12 秒，则在同一个计时样本中重复完整查询批次。',
    '- 两版分别在全部 33 组 k/ef 参数下通过单线程与 16 线程 recall 一致性检查；所有结果 ID 均合法且每条查询无重复 ID。',
    '- 适配器事先与原版直接运行的 10,000 条基向量结果比较：14 组参数、257,000 个有序结果 ID，'
    '两版各自均为零差异。此小规模验证用于检查适配器的语义，不属于完整 SIFT1M 的性能测量。',
    '- Recall@k 采用返回 ID 与真实前 k 个邻居的集合交集比例。QPS 包含查询复制/填充、旋转、距离表计算、路由和图搜索。',
    '- 原版 16 线程结果由适配器并发调用原生查询函数得到；原版示例程序本身是串行查询。',
    '- 只测试默认 T=1。每版只有一次并行构建和每配置一次计时样本，不估计置信区间或运行间波动；'
    '并行插入顺序也可能造成两版 recall 的小幅差异，因此不能将小幅 QPS 差异直接归因于重构。',
    '',
    '## 图和原始数据',
    '',
    '![Recall vs QPS](recall_vs_qps.png)',
    '',
    '![相同 k/ef 下的 recall 差异：当前减原版，单位为百分点](recall_difference.png)',
    '',
    '![Build time](build_time.png)',
    '',
    '- [Recall vs QPS：PDF](recall_vs_qps.pdf) / [SVG](recall_vs_qps.svg) / [CSV](recall_vs_qps.csv)',
    '- [Recall 差异：PDF](recall_difference.pdf) / [SVG](recall_difference.svg) / [CSV](recall_difference.csv)',
    '- [构建时间 CSV](build_times.csv)、[全部计时样本](query_samples.csv)、[对齐 recall 的估算](matched_recall.csv)',
    '- [复现方法和计时边界](README.md)、[版本与数据指纹](manifest.json)、[原始日志](raw/)',
    '',
]
(ROOT / 'REPORT.md').write_text('\n'.join(lines))
print('Saved REPORT.md')
