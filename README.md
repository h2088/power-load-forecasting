# 电力负荷预测系统

## 项目定位

这是一个面向实际电力负荷预测场景的端到端项目。系统以总负荷数据
total_consumption.csv 为主要输入，预测目标日 D 的全天 96 个 15 分钟时段负荷，并提供
回测、HTTP API、可视化页面和容器化部署能力。

当前生产服务路径只依赖总负荷数据，不要求把分表数据打包进模型或镜像。

## 业务约束

### D-6 数据可用性规则

预测目标是日期 D。由于实际业务存在数据延迟，预测时只允许使用截至 D-6 的实际数据：

- D 是需要预测的目标日，不能使用；
- D-1 至 D-5 的数据视为不可用，不能进入特征；
- D-6 是最新允许使用的实际日期；
- D-7、D-14、D-28 等更早日期可作为历史参考。

每次预测默认使用以 D-6 为最后一天的 28 个日历日训练窗口。这个约束同时应用于训练、
滚动回测和在线预测，避免把未来数据泄漏到特征中。

### 输入数据格式

原始数据不会上传到公共仓库。运行前需要将私有数据文件放到
data/total_consumption.csv。

文件至少需要包含以下列：

| 列名 | 类型 | 含义 |
|---|---|---|
| timestamp | datetime | 15 分钟粒度的时间戳 |
| consumption | numeric | 对应时段的总负荷，单位为 MWh |

系统会按时间排序、去除重复时间戳，并根据时间戳计算日期和时段编号。完整的一天
应包含 96 条记录，时段编号为 0 至 95。

## 方案1建模方法

方案1在 model.py 中实现，入口是 TwoStageLoadModel。它不是直接对 96 个时段使用
同一个未经处理的历史值，而是先预测当天的整体负荷水平，再预测一天内部的相对形状，
最后重新组合成 96 个时段的预测曲线。

### 模型结构

1. **日均水平模型**：以每天为粒度预测目标日的日均负荷，输出当天整体水平。
2. **日内形状模型**：将每个时段负荷除以当天日均负荷，学习该时段相对于日均水平
   的比例。
3. **曲线重组**：对预测的形状比例进行日内归一化和波动收缩，再乘以预测日均水平，
   得到目标日 96 个时段的负荷预测。

水平模型和形状模型均使用 sklearn.ensemble.HistGradientBoostingRegressor。模型参数
在 model.py 中统一定义，包含 500 次迭代、learning_rate=0.05、max_leaf_nodes=31
和固定随机种子 42。

### 特征工程

特征由 build_train_features() 和 build_predict_features() 统一构造，训练和预测
使用同一套字段逻辑。

**日历与时段特征**

- slot：当天第几个 15 分钟时段，范围为 0 至 95；
- dayofweek：星期编号；
- slot_parity：时段奇偶性，用于表达明显的高低交替形态。

**D-6 锚定的滞后负荷特征**

- lag_6d：D-6 同一时段的负荷；
- lag_7d：D-7 同一时段的负荷；
- lag_14d：D-14 同一时段的负荷；
- lag_28d：D-28 同一时段的负荷。

每个滞后日期还会附带日均值、日内标准差、相邻时段变化、局部均值/中位数/范围、
异常标记和所属数据阶段等统计信息。

**稳健历史统计特征**

- 近 7 天和近 28 天的同一时段中位数；
- 按数据阶段计算的时段中位数；
- 近 7 天和近 28 天的日均负荷、日内标准差中位数；
- 各阶段的日均负荷和波动水平统计；
- 日内形状比例及其阶段化历史锚点。

**趋势、波动和阶段判断特征**

- 近期 28 天的均值/标准差比例；
- D-6 与 D-7、D-14、D-28 的日均值差异；
- 近期局部波动范围；
- 历史滞后日期之间的阶段不一致程度；
- 预期阶段、阶段置信度、转折风险；
- 目标日的预期日均值和日内波动锚点。

水平模型重点使用日均负荷和阶段统计，形状模型额外使用 slot、同一时段历史信息和各
滞后日的形状比例。

### 数据质量处理

系统不会简单地把所有异常记录永久删除，而是在特征构造阶段尽量降低异常历史对预测
的影响：

- 缺失负荷优先使用该时段过去一周的可用历史均值填补；
- 如果过去一周没有可用值，则使用该时段的总体均值填补；
- 通过日内波动、相邻时段变化和阶段规则识别异常日；
- 稳健中位数和阶段统计在计算时尽量排除异常日；
- 对特征中的无穷值、全空列和常量列进行安全处理；
- 训练窗口极端稀疏时使用安全的常量回退，保证服务不会因特征分箱失败而崩溃；
- 形状预测会进行比例裁剪、日内归一化和波动收缩，防止异常值无限放大。

这种做法保留原始数据的可追溯性，同时减少异常区间、水平突变和锯齿波动对模型训练
的影响。

### 五个数据阶段

当前数据分析识别出以下五个阶段。阶段信息用于构造稳健历史统计、目标阶段锚点和
转折风险特征，而不是简单地把所有日期视为同一种分布。

| 日期范围 | 阶段标识 | 数据特征 |
|---|---|---|
| 2026-01-01 至 2026-01-08 | early_low_vol | 低波动起始/桥接段 |
| 2026-01-09 至 2026-03-13 | high_zigzag | 主高位锯齿震荡段 |
| 2026-03-14 至 2026-03-19 | high_zigzag 子段 | 高位锯齿段，日内形状存在偏移 |
| 2026-03-20 至 2026-03-23 | transition_drop | 整体下台阶的过渡段 |
| 2026-03-24 至 2026-03-29 | smooth_high_level | 高位且相对平滑的阶段 |

阶段识别和异常处理是基于历史数据统计的工程规则。接入新业务数据时，应重新检查
日期范围、阶段边界和特征稳定性，不应直接假定上述日期规则永久适用。

## 训练与预测流程

### 离线训练/回测

1. 读取并清洗 total_consumption.csv；
2. 按 15 分钟时间戳生成 date、slot、星期和奇偶特征；
3. 根据历史日曲线计算异常标记和数据阶段；
4. 构造 D-6、D-7、D-14、D-28 滞后特征及稳健历史统计；
5. 按目标日构造训练样本，分别训练水平模型和形状模型；
6. 在滚动回测中，每个目标日只使用该日 D-6 及更早的数据；
7. 输出 15 分钟预测、日汇总、每日 MAPE 和总体评估结果。

离线代码主要位于：

- model.py：特征工程、方案1模型和预测组合逻辑；
- src/pipeline.py：train()、predict() 封装；
- src/backtest.py：滚动回测、MAPE 计算和图表输出；
- src/config.py：D-6、28 天窗口和 96 时段等参数；
- run.ipynb：本地分析和回测入口。

### 单次服务预测

对于目标日期 D，后端会：

1. 校验目标日期是否满足 D-6 可用性约束；
2. 取 D-6 结束的最近 28 天训练窗口；
3. 调用 src.pipeline.train() 训练方案1；
4. 调用 src.pipeline.predict() 生成 96 个 15 分钟预测值；
5. 若目标日期已经存在实际数据，则同时返回实际值和 MAPE；
6. 返回训练覆盖率、窗口范围和异常阶段提示等数据质量信息。

当前服务支持按目标日期动态训练，适合内部分析和小规模使用。生产环境更推荐每天
定时生成一次预测文件，再由 API 读取最新结果，以降低延迟并保证结果可复现。

## 技术栈

| 层次 | 技术 | 用途 |
|---|---|---|
| 数据处理 | Python、pandas、NumPy | 时间序列整理、统计特征和数据质量处理 |
| 机器学习 | scikit-learn | HistGradientBoostingRegressor 回归模型 |
| 回测评估 | pandas、NumPy、Matplotlib | 滚动回测、MAPE、日汇总和分析图 |
| 后端 | Flask | 健康检查、模型信息和预测 API |
| 前端 | Streamlit、Plotly | 96 时段曲线、日总量柱状图和结果下载 |
| 服务通信 | HTTP/JSON、Python requests | 前端调用后端接口 |
| 容器 | Docker、Docker Compose | 前后端隔离、环境复现和本地编排 |
| 云部署 | AWS EC2 / ECS、ECR、S3、CloudWatch | 计算、镜像、私有数据和日志管理 |

## 项目结构

~~~text
.
|-- model.py                    # 方案1模型和特征工程
|-- src/
|   |-- config.py               # 核心参数
|   |-- pipeline.py             # 训练和预测封装
|   |-- backtest.py             # 滚动回测和评估
|-- backend/
|   |-- app.py                  # Flask API 路由
|   |-- data_store.py           # 数据读取和数据集元信息
|   |-- forecast_logic.py       # 预测业务逻辑
|   |-- batch_forecast.py       # 批量预测命令
|   |-- requirements.txt        # 后端依赖
|-- frontend/
|   |-- frontend.py             # Streamlit 页面
|   |-- requirements.txt        # 前端依赖
|-- data/
|   |-- README.md               # 私有数据放置说明，不包含原始数据
|-- docker-compose.yml          # 两个服务的本地编排
|-- backend/Dockerfile          # 后端镜像
|-- frontend/Dockerfile         # 前端镜像
|-- start.ps1                   # Windows 本地启动脚本
|-- stop.ps1                    # Windows 本地停止脚本
|-- DEPLOYMENT_PLAN.md          # 更完整的部署规划
~~~

回测结果、预测 CSV/JSON、运行时文件和原始数据均通过 .gitignore 排除，不应提交到
公共仓库。

## 本地运行

### Python 方式

在项目根目录执行：

~~~powershell
python -m pip install -r backend/requirements.txt
python -m pip install -r frontend/requirements.txt
python -m backend.app
~~~

另开一个终端启动前端：

~~~powershell
streamlit run frontend/frontend.py
~~~

默认访问地址：

- 前端页面：http://127.0.0.1:8501
- 后端健康检查：http://127.0.0.1:5000/health

Windows 也可以使用：

~~~powershell
./start.ps1
~~~

停止两个本地服务：

~~~powershell
./stop.ps1
~~~

### Docker Compose

确保 data/total_consumption.csv 已经存在，然后执行：

~~~powershell
docker compose up --build
~~~

默认访问地址与 Python 方式相同。后端容器通过只读卷读取
/app/data/total_consumption.csv，数据不会被复制到镜像层中。前端容器通过 Compose
内部服务名 backend 访问后端的 http://backend:5000，浏览器只需要访问前端端口。

后台运行并查看日志：

~~~powershell
docker compose up --build -d
docker compose logs -f backend frontend
~~~

可通过环境变量修改宿主机端口。例如宿主机的 5000 和 8501 已被其他服务占用时：

~~~powershell
$env:BACKEND_HOST_PORT = "5001"
$env:FRONTEND_HOST_PORT = "8502"
docker compose up --build -d
~~~

此时前端仍通过 Compose 内部网络访问后端 `http://backend:5000`；`5001` 和 `8502`
只是宿主机映射端口，不会改变容器内部端口。

## API 接口

### GET /health

返回服务存活状态：

~~~json
{"status": "ok"}
~~~

### GET /api/model-info

返回模型名称、D-6 延迟、训练窗口、时段数量、特征组、五个数据阶段和数据集元信息。

### GET /api/forecast/latest

使用当前数据集允许的最新可预测目标日生成预测。

### POST /api/predict

请求体：

~~~json
{
  "target_date": "2026-04-05"
}
~~~

返回内容包括：

- 目标日期和 D-6 截止日期；
- 28 天训练窗口的起止日期和覆盖率；
- 96 条预测记录，每条包含 timestamp、slot、predicted 和可选的 actual；
- 预测日总量、均值、峰值、谷值；
- 目标日已有实际数据时的 MAPE、MAE 和偏差；
- 数据质量警告。

示例调用：

~~~powershell
$body = @{ target_date = "2026-04-05" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/predict -ContentType "application/json" -Body $body
~~~

## AWS 部署

### EC2 + Docker Compose

EC2 适合当前项目的低成本部署和验证。基本流程如下：

1. 创建 Amazon Linux EC2 实例并配置安全组；
2. 安装 Git、Docker 和 Docker Compose 插件；
3. 克隆本仓库；
4. 通过安全方式将私有的 total_consumption.csv 放到服务器的 data/ 目录；
5. 在项目根目录构建并启动 Compose 服务；
6. 在安全组中只开放必要端口，优先使用 HTTPS 反向代理，不直接暴露后端管理接口；
7. 使用 docker compose ps、docker compose logs 和 /health 检查运行状态。

示例命令：

~~~bash
sudo dnf install -y git docker
sudo systemctl enable --now docker
git clone https://github.com/h2088/power-load-forecasting.git
cd power-load-forecasting
mkdir -p data
# 将私有 total_consumption.csv 上传到当前目录的 data/ 下
BACKEND_HOST_BIND=0.0.0.0 BACKEND_HOST_PORT=5001 FRONTEND_HOST_PORT=8502 docker compose up --build -d
~~~

如果默认端口没有冲突，可以不设置端口变量，使用后端 5000 和前端 8501。EC2 安全组
需要与实际访问方式匹配；调试阶段可临时开放前端端口，长期运行建议在 ALB 或 Nginx
后面提供 HTTPS，并限制后端端口仅允许内部访问。

### 当前 EC2 实例

当前部署实例使用非默认宿主机端口，因为实例上的其他服务已经占用 5000 和 8501：

| 服务 | 宿主机端口 | 容器端口 | 访问方式 |
|---|---:|---:|---|
| Streamlit 前端 | 8502 | 8501 | 公网访问 |
| Flask 后端 | 5001 | 5000 | Docker 内部访问，公网不开放 |

当前前端地址：

`http://57.183.6.6:8502`

当前实例的安全组只需要按需开放 TCP `8502`，建议来源设置为 `My IP`。后端 TCP
`5001` 不应开放给公网；前端容器会通过 Docker 网络中的 `backend:5000` 调用后端。
如果实例绑定的公网 IPv4 发生变化，应更新访问地址；长期运行建议为 EC2 分配并绑定
Elastic IP，避免实例停止后公网地址变化。

### 更适合长期运行的架构

长期生产环境建议将应用拆分为以下组件：

~~~text
用户 -> HTTPS/ALB -> Streamlit 前端
                  -> Flask 后端 -> 最新预测结果

私有数据 S3 -> 定时任务 -> 方案1训练/预测 -> 预测 CSV/JSON 写回 S3
                                      -> CloudWatch 日志和告警
~~~

- 使用 ECR 保存后端和前端镜像；
- 使用 ECS Fargate 运行前后端服务；
- 使用 S3 保存私有输入数据和预测产物；
- 使用 EventBridge 每日触发批量预测任务；
- 使用 CloudWatch 采集日志并配置失败、延迟和数据缺失告警；
- 使用 IAM Role 授权任务访问指定 S3 对象，不在代码或镜像中写入密钥。

生产运行建议采用“每日定时预测 + API 读取最新结果”，而不是每次用户请求都重新训练。
当前 API 的按请求训练模式保留用于内部分析、回测和人工指定日期预测。

## 评估与监控

主要评估指标是 MAPE，同时建议配合 MAE 和预测偏差一起观察：

- 15 分钟粒度：逐时段计算绝对百分比误差，再按有效时段汇总；
- 小时粒度：将连续 4 个 15 分钟时段聚合为小时后计算误差；
- 日粒度：将 96 个时段求和后比较预测日总量与实际日总量；
- 运营监控：跟踪训练窗口覆盖率、缺失率、异常阶段、MAPE、MAE 和持续性偏差。

当实际负荷接近零时，MAPE 对微小绝对误差也可能非常敏感，应同时查看 MAE 和日总量
误差，避免只根据单一指标判断模型质量。

## 数据隐私

total_consumption.csv、by_meter.csv 和 total_consumption.xlsx 等原始数据可能包含
业务敏感信息，因此默认不进入公共 Git 仓库，也不复制进 Docker 镜像。部署时应：

- 使用私有传输或受控对象存储上传数据；
- 仅授予服务读取所需数据的权限；
- 保持容器中的数据目录只读；
- 不把数据、预测明细和日志中的敏感字段写入公共 issue 或代码仓库；
- 定期检查 git status 和 git ls-files，确认原始数据没有被跟踪。
