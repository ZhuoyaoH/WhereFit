# test_long_term_air_quality.py 代码逻辑

## 职责

验证 ACAG 长期 PM2.5 生成数据的覆盖、运行时聚合、历史天气合并、缺失回退和提取辅助函数。

## 输入与输出

- 输入：项目种子表、ACAG 770 行年度表、合成历史天气、pytest 临时路径。
- 输出：pytest 断言；不保留测试产物。

## 函数与类

| 名称 | 职责 | 关键输入/输出 |
| --- | --- | --- |
| `_score` | 构造最小评分对象 | 无 / `ScoreResult` |
| `test_bundled_*` | 校验 77 城市十年完整覆盖 | 固定 CSV / 覆盖断言 |
| `test_manifest_*` | 校验版本、许可、十个源文件及输出哈希 | manifest、CSV / 溯源断言 |
| `test_acag_summary_*` | 校验连续浓度与来源状态 | 种子表、ACAG 表 / `ClimateMetrics` |
| `test_historical_*` | 校验历史天气不会丢失 ACAG 来源 | 合成天气 / 合并指标 |
| `test_missing_*` | 校验缺文件显式回退 | 临时路径 / fallback 指标 |
| `test_acag_grid_*` | 校验网格索引和 HDF5 文本解析 | 坐标、文本 / 确定性结果 |

## 数据流与主要步骤

1. 读取固定数据表并检查城市、年份、数值和格点完整度。
2. 合并种子表并验证北京十年均值、趋势、年份和非回退状态。
3. 将 ACAG 信息复制到合成历史天气指标并检查字段保留。
4. 模拟缺少数据文件，确认应用退回人工等级且明确标记。

## 依赖、配置与运行方式

运行 `python -m pytest -q tests/test_long_term_air_quality.py`。

## 假设、边界与注意事项

- 固定值断言对应 manifest 中的 V6.GL.03、2015–2024 和 3×3 提取方法。
- `tmp_path` 仅由 pytest 管理，测试结束后不进入项目产物。
