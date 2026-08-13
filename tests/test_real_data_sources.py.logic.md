# test_real_data_sources.py 代码逻辑

## 职责

验证空气质量、EONET、洪水流量和 USGS 地震数据源的响应规范化与分页边界。

## 输入与输出

- 输入：合成 API 响应、城市坐标和 pytest 隔离缓存目录。
- 输出：无持久产物；断言规范化摘要、证据文本和状态。

## 函数与类
| 名称 | 职责 | 关键输入/输出 |
| --- | --- | --- |
| `test_summarize_air_quality_*` | 验证近期 PM2.5/AQI 聚合 | 小时表 / 摘要 |
| `test_eonet_*` | 验证事件半径过滤 | GeoJSON / 事件表 |
| `test_flood_*` | 验证模型网格流量文字 | 日流量 / 记录 |
| `test_usgs_*` | 验证 20,000 条上限下的 offset 分页 | 两页 feature / 完整计数 |
| `_earthquake_feature` | 构造最小 USGS feature | ID、震级、时间 / GeoJSON 字典 |

## 数据流与主要步骤
1. 以 monkeypatch 替代外部网络，保留真实参数契约。
2. 将响应送入项目数据源函数并检查状态、聚合值和证据类型。
3. 将 USGS 测试页大小降为 2，确认满页后使用 offset=3 拉取下一页且未截断。

## 依赖、配置与运行方式

使用 pytest、pandas 和项目数据源模块；所有外部请求均模拟。

## 假设、边界与注意事项

pytest 隔离目录为框架临时产物，不保存到仓库。
