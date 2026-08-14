# build_nasa_power_climate.py 代码逻辑

## 职责

从 NASA POWER Daily Point API 读取 2000—2025 年 POWER/MERRA-2 逐日模型网格数据，为种子表中的 77 个城市生成可版本化的月度、年度气候基线和来源清单。

## 输入与输出

- 输入：`data/city_seed.csv` 的城市中心坐标；命令行可覆盖起止日期、并发数、超时和重试次数。
- 输出：`data/climate/nasa_power_merra2_city_monthly_2000_2025.csv`、年度 CSV 和 manifest JSON。

## 函数与类

| 名称 | 职责 | 关键输入/输出 |
| --- | --- | --- |
| `CityBuildResult` | 保存单城市聚合结果与响应来源 | 月度行、年度行、网格元数据 |
| `_fetch_json` | 用标准库 HTTPS 客户端限次重试 | URL / JSON 与响应哈希 |
| `_fetch_city` | 请求并校验一个 NASA POWER 点位 | 种子行 / `CityBuildResult` |
| `_apparent_temperature` | 用温度、湿度和 10 米风计算遮阴体感温度 | 三个逐日序列 / 派生序列 |
| `_write_resume_result` / `_read_resume_result` | 保存或读取已完成的单城聚合 | 项目缓存 JSON / `CityBuildResult` |
| `_validate_daily` | 拒绝缺日期、缺字段和缺测值 | 日表 / 无返回 |
| `_aggregate_period` | 计算温湿均值和平均年度阈值日数 | 日表、月份 / 聚合行 |
| `build_tables` | 并发处理全部城市并写出稳定产物 | 参数 / 三个输出路径 |

## 数据流与主要步骤

1. 读取并验证城市种子表，不允许城市名重复。
2. 每城请求 T2M、T2M_MAX、T2M_MIN、RH2M、PRECTOTCORR、WS10M 和 WS10M_MAX，最多使用五个并发请求。
3. 将 POWER 的 `YYYYMMDD -> value` 字典转成连续日表，并拒绝 `-999` 缺测、缺日或乱序响应。
4. 用 T2M、RH2M 和日均 10 米风速 WS10M 按 BOM/Steadman 公式派生遮阴体感温度；WS10M_MAX 只用于强风日阈值；用“日降水 ≥1 mm 且日均温 ≤0 °C”形成明确标记的雪日代理。
5. 每个完成城市先写入 `data/cache/nasa-power-climate-build/`；中止后可续跑，不保存逐日响应。
6. 对十二个月和全年分别聚合；阈值日数先按年统计，再取 26 年均值。
7. 全部城市完成后原子写出 CSV 和 manifest，manifest 固定参数、公式、文件哈希、响应哈希及模型网格元数据，随后清除续传缓存。

## 依赖、配置与运行方式

使用项目已有 numpy 和 pandas。运行 `python scripts/build_nasa_power_climate.py`；默认最多五并发、单请求 180 秒超时、失败重试四次。

## 假设、边界与注意事项

- POWER/MERRA-2 为模型网格数据，不是城市站点观测或行政区平均。
- 体感温度会参与舒适度评分，因此在数据质量与页面方法中标为派生字段；雪日代理不冒充实测降雪。
- 脚本不保留逐日响应，只在 manifest 中保存每城响应字节哈希；上游重处理后重跑结果可能变化。
