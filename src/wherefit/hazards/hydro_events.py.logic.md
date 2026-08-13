# hydro_events.py 代码逻辑

## 职责

获取 NASA EONET 公开事件记录和 Open-Meteo Flood 提供的城市附近区域河流流量估算。

## 数据契约

- EONET v3 `bbox` 按 `west,north,east,south` 发送，不能使用常见的 `west,south,east,north` 顺序。
- GeoJSON 事件日期优先读取 `properties.date`；旧式或兼容响应缺少该字段时才读取 `geometryDates` 最后一个值。
- 复杂几何使用所有坐标的代表性均值定位，再按球面距离筛选城市半径。

## 边界

目录事件不是城市实测灾情，城市附近区域河流流量估算也不是城市洪水概率；页面和结果说明必须保留这一空间与含义限制。
