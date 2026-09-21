from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'audit-results.json').read_text())
assert len(rows)==8
assert all(r['same_index_different_ids']==0 for r in rows)
assert all(all(r['training_fields_equal'].values()) for r in rows)
source=json.loads((ROOT/'source-comparison.json').read_text())
assert all(source['previously_validated_sources'].values())
total=sum(r['same_index_compared_ids'] for r in rows)
configs=sum(r['query_configurations'] for r in rows)
worst=max(((abs(q['recall_delta_pp']),r['case'],q) for r in rows for q in r['queries']), key=lambda item:item[0])
lines=[f'''# 当前 HVS 与原版实现一致性审计

审计日期：2026-09-21。

**结论：核心算法流程得到保留；相同索引状态上的查询在本次测试中完全一致。不能宣称默认运行、独立并行建图或所有输入条件下都严格一致。**

- 原版：Kejing-Lu/hvs `f554268f8c6abcbc23ab35c3562703d7df8ae7b5`，本次通过 GitHub `git ls-remote` 确认仍是 main/HEAD。
- 当前版：WittenYeh/hvs `33de94f523fa46fc8703411148b50fc89ada3f8c`。
- 13 个当前算法源码/头文件与先前完成受控等价性验证时的 SHA-256 全部相同。
- 本次新增测试全部使用系统最大 64 个 OpenMP 线程及 `numactl --interleave=all`，没有固定到 16 线程或单个插槽。
- 两边 OpenCV 内部线程固定为 1，以保持训练数值后端一致。本次测试关注正确性，不用进程时间比较性能。

## 逐项源码核对

| 部分 | 原版位置 | 当前位置 | 结论 |
| --- | --- | --- | --- |
| 维度补零、等间隔训练采样 | sift_1b.cpp:996、1142 | hvs.cpp:56、74 | 默认 10,000 训练样本且 n>=10,000 时规则相同；当前支持配置样本数并对 n 截断 |
| 原始图与量化图插入 | sift_1b.cpp:1198、2068；hnswalg.h:addPoint | hvs.cpp:94、387 | 插入算法函数体未变；外层并行调度不同 |
| PCA 初始化、子空间平衡 | sift_1b.cpp:1263 | hvs.cpp:105 | 相同特征值乘积平衡和旋转构造；C API 改为 C++ API |
| OPQ 更新 | sift_1b.cpp:1354 | hvs.cpp:143 | 保留两轮、每轮十次中心更新和中间一次 SVD 旋转更新 |
| K-means | sift_1b.cpp:504、589 | quantizer.cpp:9 | 保留等间隔初始化、五轮 Lloyd 更新、空簇保持原中心和严格小于的距离比较 |
| 子空间配对、量化器合并 | sift_1b.cpp:1462、689 | hvs.cpp:180、200；quantizer.cpp:33 | 有效配对评分和合并规则保留；排序只取前五项；无效配对项新增初始化 |
| 代表点筛选 | sift_1b.cpp:1790 | hvs.cpp:264 | 密度排序、delta、编码去重、最小量化误差规则相同；等误差时保留先进入临界区的点 |
| 层间跳转 | sift_1b.cpp:1912、1970、2101 | hvs.cpp:324、394 | 保留量化层/原始层标志及内部 ID 映射 |
| 入口路由 | sift_1b.cpp:2132 | hvs.cpp:406 | 仍是 4 组 × 16 中心、65,536 单元、每单元 10 入口、ef=200；当前并行调用同一个 connect |
| 查询距离表与路由选择 | sift_1b.cpp:143、363；hnswalg.h:restore_index | hvs.cpp:483 | 同样的旋转、查表、合并求和和最小中心选择；临时内存改为复用 |
| 五个图查询函数 | hnswalg.h:SearchWith* | 同名函数 | 除候选缓冲区存储方式和未使用变量外，函数体一致 |
| 最终原始向量距离和 ID | hnswalg.h:SearchWithOptGraph | hnswalg.h:1786 | 同样比较 norm(x)-2*q·x，并映射为原始 ID |

机械检查覆盖 14 组图函数，包括 addPoint、searchBaseLayer、邻居筛选、连接、密度、路由等；去除注释和空白后函数体相同。五个查询函数只规范化候选内存分配和原版未使用的 init_ids，不忽略算术、比较或遍历语句。其余五个图头文件逐字节相同。详见 [source-comparison.json](source-comparison.json)。

当前 OpenCV 源码中，旧 cvCalcPCA/cvSVD 本身调用 cv::PCA/cv::SVD。因此使用相同 OpenCV 构建时，API 迁移没有替换数学后端；但不同 OpenCV 后端和编译参数仍可能改变浮点结果。

## 不能笼统声称完全一致的地方

### 1. 独立并行构建无法保证相同索引

当前 hvs.cpp:95、388 直接并行处理内存向量；原版外层循环还带有共享文件读取/进度统计临界区。这改变执行时序。原版最终传入 addPoint 的标签仍是循环下标 i/l，并不是临界区里的 j2；不能把差异错误描述成外部标签生成规则变了。

插入完成顺序、随机层分配与邻接关系会随调度变化，密度和代表点也可能随之变化。同编码且量化误差相等时，两边都保留先获得桶锁的代表点，代表 ID 也可能不确定。

### 2. 两版共同继承共享 RNG 的并发数据竞争风险

当前 [hnswalg.h:289](../../hnsw/hnswlib/hnswalg.h#L289) 的 getRandomLevel 修改共享 level_generator_；addPoint 在节点各自的锁下调用它（当前第 1040–1041 行），而公共计数锁已经释放，global 锁尚未获取。不同节点的锁无法保护同一个随机引擎。原版函数和调用位置相同。

这是源码检查发现的共享状态无统一同步问题，本次没有运行 ThreadSanitizer。它由上游继承，不是本次重构引入；不能仅用“固定 seed”或“查询测试通过”保证并行构建的确定性与无数据竞争。

### 3. 默认随机种子、训练配置和构建环境不同

- 原版 sift_1b.cpp:1015 使用 srand(time(NULL))；当前 hvs.cpp:78 使用 srand(config.seed)，默认 100。多层合并的重复码本重采样使用这个 C RNG。图本身的默认种子两版均为 100。
- 原版 M=16、efConstruction=500、训练样本宏 size_n=10,000；当前将其公开为参数，样本数取 min(training_samples,n)。超出原版默认配置后，不能声称是在比较同一配置。
- 原版 CMake 使用 -Ofast 并寻找外部 OpenCV；当前 benchmark 通常使用 -O3，并固定 OpenCV/Eigen/zlib。历史测试已经证明 -Ofast 与 -O3 能改变同一上游源码的量化图。
- 当前增加参数/容量/退化数据检查，对原版可能越界、卡住或无效计算的输入会抛异常。这是有意增加的行为。

### 4. 原版存在未初始化的排序字段，当前补了初始化

原版 sift_1b.cpp:1518分配 k_elem 配对数组后只为全部元素设置 id=-1；只有有效配对才设置 dist。随后 QsortComp 会读取所有元素的 dist。无效项存在未初始化读取。

当前 hvs.cpp:205 使用 vector<RankedPoint>(count*count, {{-1,0}})，为无效项提供确定的值。有效配对仍按分数、ID 排序并跳过 id<0 项，但原版未定义行为本身无法被要求逐字节复现。

### 5. benchmark 的指标和计时边界有差别

原版 demo 的 Recall@1 按 ID 计算，当前通用 bench-hvs 在 k=1 时使用基于距离的 Soft Recall@1。原版 demo 的打印 QPS 将距离表预计算放在计时之外；当前库调用的完整查询路径包含这部分。它们不能不经统一定义就直接比较。此次 correctness audit 根据统一子集 GT 计算 ID Recall@k。

## 确认为工程调整的部分

- 用 cv::Mat、vector、unique_ptr 管理训练和索引资源，并补全量化层释放。
- 新增 finishLayer，使原始向量图无需写盘/释放再加载。
- 每线程 QueryScratch 复用旋转空间、距离表、候选队列和访问标记。
- 同一比较规则下，只排序实际读取的前五个码本候选。
- 密度排序复用；旋转置换由稠密置换矩阵乘法改为行复制。
- 入口单元并行构建，保留 connect 算法及每个单元的独立输出。

这些改动的资源使用、I/O 和执行时序不同，但没有在审查及测试中发现其改变受控正常输入的查询语义。

还发现原版合并采样中的 `1 / n / 2.0f` 均匀概率项因整数除法成为 0；当前 quantizer.cpp:93 有意保留等价写法 `1 / pairs / 2.0f`。这属于两版共同存在的上游行为，没有在本次一致性审计中修正。

## 已有受控证据与本次新增证据

之前在单线程、相同 seed/编译参数/依赖下，8 组输入、112 组查询配置、974,544 个有序返回 ID 完全相同；有效图边顺序、向量/范数、量化码、旋转、码本、合并表、层间映射和全部入口路由也精确相同。此次重新核对了当前算法文件的哈希，确认这些历史证据对应当前代码；没有重新执行历史单线程测试。

本次使用 64 线程、NUMA interleave，分别测试两件事：

1. **固定索引比较查询实现**：当前构建出的同一个索引，交给当前查询与原版查询分别执行，{configs} 组配置、{total:,} 个有序 ID，差异为 0。
2. **两版独立并行构建**：比较训练产物、索引字段和统一定义的 recall。8 组训练旋转/码本/合并表全部精确一致；独立图结构及查询 ID 不是全部相同。

| 数据 / T | 基向量 | 查询 | 同索引比较 ID 数 | 同索引不同 ID | 独立构建训练字段 | 独立构建最大 recall 差（百分点） |
| --- | ---: | ---: | ---: | ---: | --- | ---: |''']
for r in rows:
 lines.append(f"| {r['case']} | {r['n']:,} | {r['nq']:,} | {r['same_index_compared_ids']:,} | {r['same_index_different_ids']} | 全部一致 | {max(abs(q['recall_delta_pp']) for q in r['queries']):.4f} |")
lines += ['\n![同索引查询与独立并行构建的最大绝对 recall 差异](recall_difference.png)',
          '\n图中柱形表示每组 k/ef 配置的最大绝对差；同索引查询差异均为零。'
          '图表数据见 [recall_difference.csv](recall_difference.csv)。']
lines += [f'''
本次独立构建的最大绝对差出现在 {worst[1]}、k={worst[2]['k']}、ef={worst[2]['ef']}：原版 recall={worst[2]['upstream_recall']:.6f}，当前={worst[2]['current_recall']:.6f}，相差 {worst[0]:.4f} 个百分点。每版每组只独立构建一次，不能据此判定哪个版本在统计上更好或更差，也不能把差异全部归因于某一项重构。

每个查询进程还执行了非单调 ef 切换与重复 ef=256 的检查，所有结果 ID 合法、每条结果无重复 ID、复用 scratch 后重复结果一致。

限制：这些是有限数据上的回归证据，不是所有输入的形式化等价证明；本次数据量是每组 10,000 条，不是完整 SIFT1M；原版 64 线程查询使用逐线程 scratch 适配器调用原生查询函数，原始 demo 的外层循环仍是串行的。

详细过程见 [README.md](README.md)，逐配置数据见 [audit-results.json](audit-results.json)，编译命令及原始日志均保存在本目录。生产算法源码未修改。
''']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print('Saved REPORT.md; compared IDs:',total,'max independent recall delta pp:',worst[0])
