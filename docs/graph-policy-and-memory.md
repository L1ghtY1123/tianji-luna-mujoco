# Graph、Subgraph、Node、Edge：当前项目的控制层级与记忆

**版本说明：以下为之前 steps 实现的历史讲解。当前代码已经迁移至显式 graph/subgraph/node/edge，且 Decisions 可以生成新的恢复路径。请阅读 [当前实现与两张层级图](native-graph-implementation.md)；以下图片不代表当前生产 schema。**

本项目借鉴 Graph-as-Policy 的可执行图思想，不使用 GaP 的多智能体生成流程，也没有移植其原生运行器。当前运行入口只允许一次初始 Responses 请求；执行期间使用 Decisions 判断与选择已编写的分支。图生成失败不会自动增加 Responses 请求。

官方参考：[GaP 概述](https://graph-robots.github.io/graph-as-policy/docs/index.html)、[运行器规范](https://github.com/graph-robots/graph-as-policy/blob/main/docs/runtime.md)。官方图格式显式包含 nodes、edges、conditional_edges、subgraphs；本项目目前用 steps、if、recover 等结构表达这些控制关系，不能声称是相同 JSON 格式。

| 术语 | 本项目含义 | 当前代码表达 |
|---|---|---|
| Graph：总编排图 | 整个任务可执行的控制策略 | PolicyGraph |
| Subgraph：物体编排子图 | 某个输入物体及其产物的独立编排 | object_task.body |
| Node：节点 | 一次技能调用或一次控制判断 | skill、if、recover 等节点 |
| Edge：边 | 下一步控制转移，可带条件或形成循环 | 步骤顺序、if 分支、recover 的选择和返回路径 |
| 任务要求 | 物体的来源、目的地、处理要求与完成条件，不是动作节点 | TaskSpec |
| 物体清单 | 稳定身份、当前证据、历史观测、操作记录与完成账本 | inventory / task_progress |

使用中英文完整对应名称，不再把“策略图”作为额外概念。一个物体子图不是一个 sub-agent；当前各子图共用同一个运行器和机器人，按总图调度。GaP 官方的技能子图与这里按物体组织的子图也不是同一粒度：这里按物体组织是本项目为清单需求做的适配。

## 图一：控制层级

这张图只说明职责，不把 SAM3 的像素处理细节塞进控制层。虚线为证据读写，实线为控制或调用。

![控制层级](figures/gap-1.png)

<details><summary>展开控制关系的 Mermaid 概念源</summary>

```mermaid
flowchart TB
    subgraph COMPILE[层一：只运行一次的任务编排]
        U[任务、初始RGB\n配置与技能契约] --> R[Responses]
        R --> G[Graph 总编排图\nSubgraph、Node、Edge]
        R --> I[物体清单与每件物体的任务要求]
    end
    subgraph EXEC[层二：执行图并维护状态]
        X[运行器\n执行 Node，沿 Edge 推进]
        X -->|需要语义判断| J[Decisions\n判断／选择恢复边]
        J -->|结果及分支选择| X
        M[(清单记忆\n最后观测、已验证关系、操作记录)]
    end
    subgraph CAP[层三：节点调用的技能实现]
        S[执行被选中的技能\n观测、建图、规划、动作、核验]
    end
    G --> X
    I --> M
    X -->|技能节点调用| S
    S -->|结果／unknown／具体失败| X
    S -.记录证据.-> M
    M -.当前证据与历史记忆.-> J
    classDef model fill:#e9e2ff,stroke:#7555b5,color:#232139
    classDef graphNode fill:#e1efff,stroke:#377cb5,color:#16324c
    classDef memory fill:#fff3d7,stroke:#b78b2e,color:#48391b
    class R,J model
    class G,X graphNode
    class I,M memory
```

</details>

图是策略：同一个节点返回成功、未知或失败时，可以沿不同的边继续，而不是照固定步骤表执行。数值碰撞与关节检查不交给模型投票；所有换位姿动作仍须图中显式节点及合法运动前提。当前初始原始图像是启动输入，之后的任务技能由图控制。

## 图二：多物体独立编排

下面是结构示意，不是替 Responses 写死的任务模板。为了看清层级，只展开每件物体的目标绑定、恢复分流和结果验收；中间的建图、抓姿、抓取、放置节点明确折叠，不能将折叠框误认为单次技能调用。

![多物体独立编排](figures/gap-2.png)

<details><summary>展开多物体图的 Mermaid 概念源</summary>

```mermaid
flowchart TB
    subgraph GRAPH[Graph：整项任务的总编排]
        E([进入总图]) --> SH[共享观测与清单确认节点组]
        subgraph PINK[Subgraph 甲：粉色物体 → 预设输入区域]
            direction TB
            A[Node：绑定甲] -->|成功 Edge| AWORK[[甲的其他抓放节点\n展开略]]
            A -->|unknown Edge| AR{Decisions 选择恢复边}
            AR -->|改变视角| AV[[建图→换位姿→观测\n恢复节点组]]
            AV -->|重新绑定 Edge| A
            AR -->|无可用恢复| AP[保留证据，等待处理]
            AWORK --> AC[Node：核验甲的放置结果]
        end
        subgraph GREEN[Subgraph 乙：绿色物体 → 桌面]
            direction TB
            B[Node：绑定乙] -->|成功 Edge| BWORK[[乙的其他抓放节点\n展开略]]
            B -->|unknown Edge| BR{Decisions 选择恢复边}
            BR -->|改变视角| BV[[建图→换位姿→观测\n恢复节点组]]
            BV -->|重新绑定 Edge| B
            BR -->|无可用恢复| BP[保留证据，等待处理]
            BWORK --> BC[Node：核验乙的放置结果]
        end
        SH --> A
        AC -->|甲完成，调度乙| B
        BC -->|各物体完成条件满足| F[最终回位与整体核验节点组]
        F --> END([任务完成])
    end
    classDef pink fill:#ffe7f0,stroke:#bd5d88,color:#482735
    classDef green fill:#e4f6eb,stroke:#448c62,color:#213b2c
    classDef decision fill:#eee5ff,stroke:#8060ad,color:#302144
    class A,AWORK,AC pink
    class B,BWORK,BC green
    class AR,BR decision
```

</details>

两件物体可以有不同目的地、处理条件、抓取参数和恢复分支。总图选择先后顺序；甲完成不意味着乙完成；拆包产物通过输入—产物关联继承对应输入的任务要求。没有单独启动的物体 agent，也没有每件物体再调用一次 Responses。

图中的恢复边只有被 Responses 写进当前图后才能执行；示意图不代表每次模型都完整生成，也不代表该路径已经实测通过。真实图和实际执行分支保存在实验归档中。

## 图三：观测节点内部的数据处理

这是技能内部数据流，不是另一套任务控制器。

```mermaid
flowchart LR
    RGB[当前每路RGB] --> SAM[SAM3]
    WORD[清单外观短语\n主描述与互补描述] --> SAM
    SAM --> MASK[像素掩码和分割分数]
    MASK --> BOX[程序计算\nSAM3当前候选框]
    BOX --> ID[Decisions 身份关联]
    REF[Responses初始参考框\n参考RGB与清单描述] --> ID
    RGB --> ID
    MASK --> DEPTH[掩码＋标定深度\n候选三维位置与表面]
    CAL[深度与相机内外参] --> DEPTH
    DEPTH -->|辅助定位| ID
    ID --> BIND[永久ID对应当前掩码]
    BIND --> GEO[按已确认ID汇总几何]
    DEPTH --> GEO
```

SAM3 收到图像和文本短语，返回掩码；程序计算候选框。Responses 初始参考框是历史定位提示，不是 SAM3 的分割边界。Decisions 负责对应身份，不绘制像素掩码。只有之后总图调用建图／抓姿节点，才把对应实测几何送入 TSDF/ESDF／GraspGen。

## 绿色物体：保留记忆与获得新证据是两件事

此前清单已经保留绿色物体初始来源证明，但支撑判断只收到了当前像素和当前几何，未收到那份已验证的历史关系。当前改为同时提供：

1. 原始 Decisions 来源关系证明，保留原 observation_id 和 request_hash。
2. 最后观测位置、最后观测时刻、当前是否可见。
3. 该物体后续的抓取尝试、转移记录和当前状态。
4. 当前 RGB 与当前身份绑定的实测几何。

模型可据此判断关系是否连续，而不是把暂时出画解释为之前的信息失效。初始参考框和未经核验的描述仍只是提案。历史信息不会改写时间戳冒充新观测；操作或矛盾证据可能使旧位置失效。当前抓姿和运动仍需要对应的几何、地图与数值检查。

identify_target 是无物理动作、可重试的绑定能力，现在可作为 recover 的 attempt。未知结果可以由 Decisions 选择图中已有换位姿分支。程序不在该技能内部偷偷移动，也不自动为模型补一个恢复图。

完整日志和真实验证结果见 [本轮验证](graph-memory-validation-20261029.md)。
