# 车辆在途时间预测功能使用说明

## 功能概述

本功能在客户选择模型中增加了车辆在途时间作为确定性效用项。使用CNN模型根据VRP历史数据，针对每个顾客到达t时刻，预测推荐的OOH点和HOME点分别到达终点（depot）的时间，并显示给乘客。

**关键概念**：在途时间是从HOME点或OOH点到终点（depot）的累计时间。系统会预测这几个点在整个VRP的服务顺序（根据历史的运行情况），然后计算从该点到终点（depot）的时间。

## 实现内容

### 1. CNN预测模型

- 在 `Src/Utils/Predictors.py` 中添加了 `CNN_TravelTime` 类
- 复用 `CNN_2d` 的结构，专门用于预测车辆在途时间

### 2. 客户选择模型修改

- 在 `Environments/OOH/customerchoice.py` 中：
  - 添加了 `travel_time_weight` 参数（在途时间的权重系数，应为负值）
  - 修改了 `mnl_euclid` 和 `mnl_distmat` 方法，支持在途时间参数
  - 修改了 `customerchoice_offer` 和 `customerchoice_pricing` 方法，支持接收在途时间字典

### 3. DSPO算法集成

- 在 `Src/Algorithms/DSPO.py` 中：
  - 添加了 `use_travel_time_prediction` 配置选项
  - 添加了 `get_travel_time_prediction` 方法用于预测在途时间
  - 初始化了 `travel_time_predictor` CNN模型（如果启用）

### 4. DSPO_Revenue集成

- 在 `Src/Algorithms/DSPO_Revenue.py` 中：
  - 在 `get_action_pricing` 方法中调用在途时间预测
  - 将预测结果存储到环境中

### 5. 环境支持

- 在 `Environments/OOH/Parcelpoint_py.py` 中：
  - 添加了 `current_travel_times` 属性存储当前预测
  - 添加了 `set_travel_times` 和 `clear_travel_times` 方法
  - 修改了 `get_delivery_loc_pricing` 和 `get_delivery_loc_offer` 方法，传递在途时间给客户选择模型

## 使用方法

### 1. 配置参数

在配置文件中添加以下参数：

```python
# 启用在途时间预测
config.use_travel_time_prediction = True

# 在途时间的权重系数（负值，因为时间越长效用越低）
# 建议值：-0.001 到 -0.01（根据时间单位调整）
config.travel_time_weight = -0.001  # 例如：-0.001 表示每增加1秒，效用减少0.001

# 在途时间预测器的学习率（可选，默认使用主学习率）
config.travel_time_learning_rate = 0.001

# 在环境初始化时传递travel_time_weight
env = Parcelpoint_py(
    ...,
    travel_time_weight=config.travel_time_weight
)
```

### 2. 训练在途时间预测模型

**训练已集成到主训练流程中！** 在途时间预测模型会自动训练，无需单独训练。

训练流程：

- **数据收集**：在每个episode结束时，从最终优化路线中自动收集真实的在途时间数据
- **自动训练**：在`initial_phase_training`和`optimize`方法中自动训练在途时间预测模型
- **训练数据**：包括状态特征（车辆位置、客户位置、OOH点位置等）和真实的在途时间标签

训练过程与成本预测模型类似，但标签是在途时间而不是成本。

### 3. 在途时间数据格式

预测的在途时间以字典形式传递：

```python
travel_times = {
    'home': 120.5,  # 从HOME点到终点（depot）的累计时间（秒）
    'ooh': [45.2, 67.8, 89.3, ...]  # 从各OOH点到终点（depot）的累计时间列表（秒），顺序与OOH点列表一致
}
```

**注意**：在途时间是从HOME/OOH点到终点（depot）的时间，不是从车辆当前位置到HOME/OOH点的时间。

### 4. 效用计算

在客户选择模型中，在途时间效用项的计算方式为：

```
效用 = 基础效用 + 距离效用 + travel_time_weight * 在途时间
```

其中：

- `travel_time_weight` 应该是负值（例如 -0.001）
- 在途时间越长，效用越低
- 如果 `travel_time_weight` 为 `None` 或不提供在途时间，则不使用在途时间效用

## 注意事项

1. **权重调整**：`travel_time_weight` 的值需要根据实际情况调整。如果时间单位是秒，建议使用较小的负值（如 -0.001）。
2. **模型训练**：在途时间预测模型已集成到主训练流程中，会自动训练。训练数据从每个episode的最终优化路线中自动收集。
3. **车辆速度**：在途时间的计算需要正确的车辆速度。默认使用`config.truck_speed`，如果未设置则使用1.0。确保速度单位与距离矩阵的单位匹配（例如，如果距离是米，速度应该是m/s）。
4. **向后兼容**：如果不启用 `use_travel_time_prediction` 或不提供 `travel_time_weight`，代码会正常工作，只是不使用在途时间效用。
5. **环境访问**：代码通过 `config.env` 访问环境来设置在途时间。确保在配置中正确设置了环境实例。
6. **训练数据量**：在途时间预测模型需要足够的训练数据才能准确预测。建议至少运行几个episode后再使用预测结果。
7. **在途时间定义**：在途时间定义为从HOME点或OOH点到终点（depot）的累计时间。
   - **预测时**：CNN根据当前fleet状态和历史运行情况，预测如果选择HOME或某个OOH点，该点在最终VRP路线中的服务顺序，然后预测从该点到终点（depot）的累计时间
   - **训练时**：从最终优化路线中找到HOME点在路线中的位置，或模拟插入OOH点找到最佳插入位置，然后计算从该位置到终点（depot）的真实累计时间
   - 这反映了客户选择某个选项后，该选项在整个VRP服务顺序中的位置，以及从该位置到服务结束的时间
8. **数据完整性**：修复后的实现会收集所有选项的数据，这大大增加了训练数据的数量和质量，有助于提高模型预测精度。

## 示例代码

```python
# 在配置中启用功能
config.use_travel_time_prediction = True
config.travel_time_weight = -0.001

# 初始化环境
env = Parcelpoint_py(
    ...,
    travel_time_weight=config.travel_time_weight
)

# 初始化agent（会自动创建travel_time_predictor）
agent = DSPO_Revenue(config)

# 在决策时，agent会自动：
# 1. 预测在途时间
# 2. 存储到环境中
# 3. 客户选择模型会自动使用在途时间计算效用
action = agent.get_action(state, training=True)
```

## 已完成的改进

1. ✅ **训练集成**：在途时间预测模型的训练已集成到主训练流程中
2. ✅ **数据收集**：在episode结束时自动收集真实的在途时间数据用于训练
3. ✅ **模型保存/加载**：支持保存和加载训练好的在途时间预测模型（通过MemoryBuffer的save/load方法）
4. ✅ **完整数据收集**：修复了训练数据收集问题，现在会收集所有选项（HOME和所有OOH点）的数据，而不仅仅是实际选择的选项
5. ✅ **特征一致性**：修复了特征构建问题，确保训练和预测时使用相同的特征构建逻辑
6. ✅ **在途时间计算**：修复了在途时间计算逻辑，正确计算从HOME点或OOH点到终点（depot）的累计时间

## 训练细节

### 数据收集

- 在每个episode结束时，`get_per_customer_travel_times`方法会从最终优化路线中计算真实的在途时间
- **重要改进**：现在会为每个客户收集**所有选项**（HOME和所有OOH点）的在途时间数据，而不仅仅是实际选择的选项
- **HOME点时间计算**：从最终优化路线中找到HOME点的位置，计算从HOME点到终点（depot）的累计时间
- **OOH点时间计算**：模拟插入OOH点到客户到达时刻t的fleet状态，找到最佳插入位置，计算从该插入位置到终点（depot）的累计时间
- 这确保了模型能够学习到所有备选选项的在途时间，而不仅仅是实际选择的选项
- 数据自动存储到`travel_time_memory` MemoryBuffer中

### 特征构建

- 训练和预测时使用**相同的特征构建逻辑**
- 对于HOME点：特征包含客户HOME位置
- 对于OOH点：特征包含对应OOH点的位置（而不是客户位置）
- 这确保了训练和预测时的特征一致性

### 训练方法

- **初始阶段**：在`initial_phase_training`中，如果`travel_time_memory`有足够数据，会同时训练在途时间预测模型
- **在线训练**：在`optimize`方法中，每次更新成本预测模型时，也会更新在途时间预测模型
- **训练损失**：使用Huber Loss，与成本预测模型一致

### 模型保存

- 在途时间预测模型会与主模型一起保存
- MemoryBuffer数据会保存到`initial_travel_time_`前缀的文件中
- 可以通过`travel_time_memory.save()`和`travel_time_memory.load()`方法保存/加载训练数据

