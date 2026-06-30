# WhereFit 本地实现操作日志

记录范围：仅记录本次在 `/Users/zhuoyao/Documents/WhereFit/WhereFit` 内进行的实现操作、状态和修改。

约束：
- 从 2026-06-29 18:32 CST 起，所有写入操作只允许发生在 `/Users/zhuoyao/Documents/WhereFit/WhereFit`。
- 不修改 `/Users/zhuoyao/Documents/WhereFit/WhereFit` 以外的任何文件。
- 暂不考虑 Git 仓库操作，先完成本地可运行项目。

## 操作记录

| 时间 | 操作 | 状态 | 影响 |
|---|---|---|---|
| 2026-06-29 18:32 CST | 确认用户更正的项目路径为 `/Users/zhuoyao/Documents/WhereFit/WhereFit` | 完成 | 后续实现路径切换到该目录 |
| 2026-06-29 18:32 CST | 检查 `/Users/zhuoyao/Documents/WhereFit/WhereFit` 当前文件 | 完成 | 未发现已有文件 |
| 2026-06-29 18:32 CST | 创建本操作日志 `OPERATION_LOG.md` | 完成 | 开始记录后续所有本地实现操作 |
| 2026-06-29 18:33 CST | 创建项目目录结构 | 完成 | 新增 `data/`、`src/wherefit/`、`src/wherefit/scoring/`、`src/wherefit/report/`、`src/wherefit/visualization/`、`tests/` |
| 2026-06-29 18:35 CST | 写入核心源码和静态数据 | 完成 | 新增配置、模型、数据读取、评分、报告、图表、地图模块，以及 `data/city_seed.csv` |
| 2026-06-29 18:36 CST | 写入应用入口、测试和 README | 完成 | 新增 `app.py`、`tests/test_scoring.py`、`README.md`、`requirements.txt`、`.env.example`、`pytest.ini` |
| 2026-06-29 18:37 CST | 尝试用系统 Python 运行测试 | 未完成 | 系统 Python 缺少 `pytest`，未运行测试 |
| 2026-06-29 18:38 CST | 创建项目内临时 `.venv` 并安装依赖 | 已撤销 | 依赖装在项目内 `.venv`，但不符合用户要求的 conda/mamba 环境 |
| 2026-06-29 18:55 CST | 按用户确认删除 `.venv` | 完成 | 移除项目内临时虚拟环境，准备改用项目内 `.conda-env` |
| 2026-06-29 18:56 CST | 尝试用 mamba 创建项目内 `.conda-env` | 未完成 | 首次被网络沙箱拦截，联网重试后长时间无输出，手动中断，退出码 139 |
| 2026-06-29 18:59 CST | 检查 mamba 创建残留 | 完成 | 未生成 `.conda-env`，仅生成项目内 `.conda-pkgs/urls.txt` |
| 2026-06-29 19:00 CST | 尝试用 conda 创建项目内 `.conda-env` | 未完成 | conda-forge 元数据解析过慢，手动中断，未生成 `.conda-env` |
| 2026-06-29 19:02 CST | 调整环境策略 | 进行中 | 按用户建议，conda 只创建 Python/pip 环境，项目依赖改由 pip 安装 |
| 2026-06-29 19:03 CST | 再次尝试用 mamba 创建精简 `.conda-env` | 未完成 | 网络较慢且命令无输出，按用户要求停止，退出码 139 |
| 2026-06-29 19:04 CST | 检查当前项目状态 | 完成 | 未生成 `.conda-env`；代码文件已写好；依赖安装和测试尚未完成 |
| 2026-06-29 19:48 CST | 恢复环境创建和验证流程 | 进行中 | 确认 `.conda-env` 不存在，准备重新用 conda 创建项目内环境并安装依赖 |
| 2026-06-29 19:49 CST | 创建项目内 conda 环境 `.conda-env` | 完成 | 使用 conda 安装 Python 3.11 和 pip 到项目目录内 |
| 2026-06-29 19:50 CST | 用 conda 安装应用依赖 | 完成 | 安装 `streamlit`、`pandas`、`numpy`、`plotly`、`pydeck`、`requests`、`python-dotenv`、`pytest` |
| 2026-06-29 19:52 CST | 运行单元测试 | 完成 | `pytest` 收集 8 个测试，结果 8 passed，用时 87.50 秒 |
| 2026-06-29 19:59 CST | 启动 Streamlit 本地服务 | 完成 | 服务启动在 `http://127.0.0.1:8501`；普通沙箱不允许绑定端口，使用授权方式完成烟雾验证 |
| 2026-06-29 20:00 CST | 验证 Streamlit HTTP 可访问 | 完成 | 本地请求返回 HTTP 200；随后停止 Streamlit 服务 |
| 2026-06-29 20:00 CST | 更新 README 运行方式 | 完成 | README 改为项目内 conda prefix 环境和 conda 安装依赖的命令 |
| 2026-06-29 20:01 CST | 运行业务烟雾测试 | 完成 | 默认输入识别 5 个城市，无缺失；当前偏好下第一名为 Sapporo，匹配度 82.5 |
| 2026-06-29 20:02 CST | 最终测试复核 | 完成 | `pytest -q` 结果 8 passed in 1.35s；Streamlit 会话已停止 |

## 当前状态

- 已完成 MVP 代码初稿。
- 已完成 conda 环境创建和依赖安装。
- 单元测试已通过。
- Streamlit 启动和 HTTP 访问烟雾验证已通过。
- 业务评估烟雾测试已通过。
- 当前项目目录占用约 1.9G，主要来自 `.conda-env` 和 `.conda-pkgs`。
