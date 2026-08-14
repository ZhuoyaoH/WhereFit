# models.py 代码逻辑

## 职责

定义位置、气候、偏好、评分、预报、空气质量、数据质量和灾害记录的数据契约。

## 关键约束

- `ClimateMetrics.period_months` 区分月度与全年聚合，供日数指标归一化。
- `estimated_fields` 与 `fallback_fields` 保留字段级来源边界。
- `pm25_*` 字段单独保存长期空气质量的来源、年份、样本数、趋势和空间提取方法，避免被总体气候状态覆盖。
- `DataQualityRecord.affects_score` 明确展示数据是否进入主排名。
- `CityResult.air_quality` 仅承载短期空气质量，不替换长期气候基线。
