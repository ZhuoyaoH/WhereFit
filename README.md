# WhereFit AI

WhereFit AI 是一个个性化气候地点推荐 MVP。用户输入候选城市和气候偏好后，应用会基于静态城市气候 proxy、历史天气、未来预报和灾害/极光数据入口，计算旅行舒适度、长期风险 proxy 和个性化匹配度，并展示排名、地图、图表和中文解释。

## 当前版本

- Streamlit 单页应用
- 47 个国内城市和高纬极光相关城市静态种子数据
- 支持中英文城市名输入
- 支持旅行、长期居住、城市对比三种模式
- 支持温度、湿度、降水、空气质量、极端天气偏好
- 输出城市卡片、排名表、地图、雷达图、中文解释报告
- V01 增加历史气候常态模式，支持 Open-Meteo Historical Weather API 和本地 CSV 缓存
- V01 增加未来天气预报 tab，支持 Open-Meteo Forecast 1-16 天短期预报
- V01 增加历史灾害档案 tab，支持按需查询 USGS 地震历史和 IBTrACS 西北太平洋台风路径统计；洪涝/滑坡先用 proxy 摘要
- V01 增加极光机会 tab，支持 NOAA SWPC OVATION 短临预报，失败时回退纬度启发式
- 核心评分函数有单元测试

## 运行方式

从 GitHub 克隆后，进入应用目录安装依赖并启动：

```bash
cd WhereFit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

如果偏好 conda/mamba：

```bash
cd WhereFit
conda create --prefix ./.conda-env python=3.11 pip -y
conda activate ./.conda-env
pip install -r requirements.txt
streamlit run app.py
```

本地 API 缓存会自动生成在 `data/cache/`，该目录不需要提交到 Git。

## 测试

```bash
cd WhereFit
pytest -q
```

## V01 数据模式

页面侧边栏提供：

- `Demo 静态数据`：使用 `data/city_seed.csv`，稳定、不联网。
- `历史气候常态（2000 至今）`：请求 Open-Meteo Historical Weather API，并缓存到 `data/cache/weather/history/`。如果 API 失败，会自动回退到静态数据。

历史灾害页当前提供：

- USGS 地震历史查询入口；
- IBTrACS 西北太平洋台风路径统计，按城市周边 100/200/500km 统计接近次数；
- 洪涝/滑坡先用强降水和区域 proxy 摘要，后续接事件库；
- 所有灾害信息仅作历史记录与暴露度参考，不构成安全评级。

未来预报页当前提供：

- Open-Meteo Forecast 1-16 天预报；
- 统计最高温、最高体感温度、降水概率、强降水日和大风日；
- 结果缓存到 `data/cache/weather/forecast/`。

极光机会页当前提供：

- NOAA SWPC OVATION Aurora 30 Minute Forecast；
- 显示最近网格概率、距离和预报时间；
- 请求失败或未勾选时回退纬度启发式。

## 数据口径说明

这里的“数据口径”指评分使用什么来源、什么粒度、什么定义的数据。当前应用支持两类口径。

静态 fallback 使用 `data/city_seed.csv` 中的人工等级数据：

- `summer_heat_level`
- `humidity_level`
- `air_quality_level`
- `precipitation_level`
- `winter_cold_level`
- `coastal`
- `typhoon_region`

这些字段是 1-5 级 proxy。程序会把它们转换成近似的温度、湿度、降水天数、PM2.5、热天数和强降水天数。这样做的优点是稳定、可控、无需 API key，适合作为 fallback。缺点是它不是权威气候数据。

历史气候常态使用 Open-Meteo Historical Weather API：

- 默认历史窗口：2000-01-01 至最近可用完整日；
- 按城市经纬度请求日尺度天气；
- 聚合目标月份的温度、体感温度、降水日、强降水日、高温日、大风日等指标；
- 数据写入本地缓存，避免重复请求。

## 如果要接真实气候数据

建议分三步升级，不要一开始就把 UI 和 API 绑死：

1. 历史气候数据：接 Open-Meteo Archive API，按城市经纬度和月份拉取过去多年日尺度数据，聚合出平均最高温、平均湿度、降水天数、强降水天数、热天数。
2. 空气质量数据：接 OpenAQ 或其他空气质量数据源，按城市或最近监测站聚合 PM2.5。不同国家的 AQI 标准不同，建议统一用 PM2.5 浓度。
3. 长期风险数据：接 ERA5-Land、WorldClim、CMIP6 或预处理后的气候栅格数据，计算热浪、暴雨、未来变暖趋势等长期 proxy。

需要额外准备：

- 稳定的数据源选择和 API 使用限制评估
- 城市名到经纬度的 geocoding 表
- 本地缓存层，避免每次刷新 Streamlit 都打 API
- 明确时间范围，例如近 10 年、近 30 年或未来情景
- 单位和缺失值处理规则

当前代码已经把数据读取、指标转换、评分函数和 UI 分开，后续可以新增 `src/wherefit/data_sources/`，让真实 API 输出同样的 `ClimateMetrics`，评分层无需大改。

## 英文版预留

当前页面文案以中文为主。后续要改成全英文，优先抽离：

- `app.py` 中的 UI 文案
- `src/wherefit/report/template_report.py` 中的报告模板
- `src/wherefit/config.py` 中的模式和标签

可以新增 `src/wherefit/i18n.py`，按 `zh` / `en` 管理文案。

## 免责声明

本项目是气候适配推荐原型，评分仅用于信息展示和作品集演示，不应用于房产、保险、医疗或灾害避险等高风险决策。
