不是。**原文不是把 4 张图都放进 CNN 训练。**

更准确地说：

> **一天结束后会解 4 个 CVRP / HGS 路线问题，但 CNN 训练样本只有 3 条，分别对应 A、B、C 三个顾客。**

---

# 1. 假设一天最终有 A、B、C 三个被服务点

一天结束后，完整服务集合是：

```text
A, B, C
```

为了计算真实插入成本，原文会求：

```text
完整路线：A + B + C
去掉 A：B + C
去掉 B：A + C
去掉 C：A + B
```

所以确实会有 **4 次路线求解**：

```text
CVRP(ABC)
CVRP(BC)
CVRP(AC)
CVRP(AB)
```

原文 Figure 5 的意思也是：如果一天有 3 个顾客，就求 1 个完整 CVRP 和 3 个 leave-one-out CVRP，由此得到 3 个 insertion costs。

---

# 2. 但这 4 个路线结果不是 4 张 CNN 训练图

这点最重要。

这 4 个 CVRP 的作用是：

```text
用来计算 label
```

不是：

```text
全部作为 CNN 输入图
```

真正进入 CNN 训练的是 3 条样本：

```text
样本 1：A 对应的状态图 → A 的插入成本 label
样本 2：B 对应的状态图 → B 的插入成本 label
样本 3：C 对应的状态图 → C 的插入成本 label
```

也就是说：

```text
3 个顾客
→ 3 个 CNN training samples
```

而不是：

```text
4 个 CVRP
→ 4 张 CNN training images
```

---

# 3. 三个 label 是怎样算出来的？

完整路线成本记为：

```text
Cost(ABC)
```

那么：

```text
A 的 label = Cost(ABC) - Cost(BC)
B 的 label = Cost(ABC) - Cost(AC)
C 的 label = Cost(ABC) - Cost(AB)
```

这三个差值才是 CNN 要学习的插入成本。

所以训练数据是：

```text
A 的状态图 → Cost(ABC) - Cost(BC)
B 的状态图 → Cost(ABC) - Cost(AC)
C 的状态图 → Cost(ABC) - Cost(AB)
```

---

# 4. 状态图是什么时候存的？

状态图不是一天结束后才统一生成的，而是在**顾客到达并选择后**，在 booking horizon 中存下来。

假设顾客顺序是：

```text
A → B → C
```

那么大概是：

## A 到达并选择后

存一张和 A 有关的状态图：

```text
[A]
```

之后一天结束后，给它配 label：

```text
Cost(ABC) - Cost(BC)
```

---

## B 到达并选择后

此时 A 已经在系统里，B 也选择了一个服务位置。

存一张状态图：

```text
[A, B]
```

之后配 label：

```text
Cost(ABC) - Cost(AC)
```

---

## C 到达并选择后

此时 A、B 已经在系统里，C 也选择了一个服务位置。

存一张状态图：

```text
[A, B, C]
```

之后配 label：

```text
Cost(ABC) - Cost(AB)
```

所以 CNN 训练样本更像是：

```text
图 1：[A]       → A 的真实插入成本
图 2：[A, B]    → B 的真实插入成本
图 3：[A, B, C] → C 的真实插入成本
```

而不是：

```text
图 1：ABC
图 2：BC
图 3：AC
图 4：AB
```

---

# 5. 最容易混淆的地方

你现在混淆的是这两个东西：

| 东西                 | 作用                       |
| ------------------ | ------------------------ |
| CNN 输入图            | 表示顾客到达时的状态，用来训练 CNN      |
| leave-one-out CVRP | 只用来计算 label，不直接作为 CNN 输入 |

所以：

```text
ABC、BC、AC、AB
```

这些路线求解结果主要是为了算：

```text
A、B、C 各自的真实插入成本 label
```

不是为了全部喂给 CNN。

---

# 6. 一句话总结

> **如果一天最终服务了 A、B、C 三个点，原文会解 4 个 CVRP：ABC、BC、AC、AB；但 CNN 训练样本只有 3 条，分别是 A、B、C 在到达时存下来的状态图，label 分别是 Cost(ABC)-Cost(BC)、Cost(ABC)-Cost(AC)、Cost(ABC)-Cost(AB)。**

所以你 PPT 里可以写：

```text
Three accepted customers generate three CNN training samples.
Four CVRP solves are used only to compute the three insertion-cost labels.
```
对，**这句话基本正确**。你可以这样理解：

> **CNN 训练 = 状态图作为输入，日终计算出的插入成本作为 label。**

更完整地说：

```text
一天仿真过程中：
    每个顾客到达并选择后，系统存下对应的状态图

一天仿真结束后：
    用 HGS/CVRP 求完整路线
    再分别移除每个顾客，重新求路线
    用“完整路线成本 − 移除该顾客后的路线成本”
    得到该顾客的插入成本 label

CNN 训练时：
    输入 = 当时存下的状态图
    输出目标 = 该顾客的插入成本 label
```

---

## 用 A、B、C 举例

假设一天内最终有三个顾客被接受：

```text
A, B, C
```

### 1. 一天中间存状态图

假设到达顺序是：

```text
A → B → C
```

那么系统会存：

```text
图 1：A 选择后形成的状态图
图 2：A、B 选择后形成的状态图
图 3：A、B、C 选择后形成的状态图
```

这些是 CNN 的输入数据。

---

### 2. 一天结束后算 label

一天结束后，先求完整路线：

```text
Cost(A+B+C)
```

然后分别移除每个顾客：

```text
Cost(B+C)   # 移除 A
Cost(A+C)   # 移除 B
Cost(A+B)   # 移除 C
```

于是得到三个 label：

```text
A 的 label = Cost(A+B+C) - Cost(B+C)

B 的 label = Cost(A+B+C) - Cost(A+C)

C 的 label = Cost(A+B+C) - Cost(A+B)
```

Akkerman 原文也明确说，训练数据的真实成本是在 cutoff time 之后获得的：移除顾客选择的 delivery location，重新求解 CVRP，再用完整路线和移除后的路线成本差作为 insertion cost。

---

## 所以最终 CNN 训练样本是

```text
状态图 1 → A 的插入成本 label
状态图 2 → B 的插入成本 label
状态图 3 → C 的插入成本 label
```

也就是：

```text
input  = 状态图
target = 插入成本
```

---

## 最准确的一句话

你可以在 PPT 里这样写：

```text
The CNN is trained using state-encoding images as inputs and HGS-derived insertion costs as training labels.
```

中文：

```text
CNN 以状态编码图作为输入，以日终 HGS 计算得到的插入成本作为训练标签。
```

再稍微完整一点：

```text
During the simulated booking horizon, the state encodings are stored. After the cutoff time, insertion-cost labels are obtained by comparing the full HGS route with leave-one-out HGS routes. These state–cost pairs are then used to train the CNN.
```

中文：

```text
在仿真预订期内，系统存储状态编码图。预订期结束后，通过比较完整 HGS 路线与逐一移除顾客后的 HGS 路线，得到插入成本标签。随后，这些“状态图–成本标签”样本用于训练 CNN。
```

---

## 注意一个小细节

不要说：

```text
一天结束后生成很多张图，再放进 CNN。
```

更准确是：

```text
状态图在顾客到达和选择过程中被存下来；
一天结束后主要是计算这些状态图对应的 label。
```

所以最终逻辑是：

```text
仿真过程中存 input
仿真结束后算 label
然后训练 CNN
```
