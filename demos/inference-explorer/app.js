const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  modelKey: "resnet",
  stepIndex: 0,
  timer: null,
  nodes: {},
  edges: {}
};

const ui = {
  contentGrid: document.getElementById("contentGrid"),
  graph: document.getElementById("graph"),
  graphTitle: document.getElementById("graphTitle"),
  stepBadge: document.getElementById("stepBadge"),
  stepTitle: document.getElementById("stepTitle"),
  stepDesc: document.getElementById("stepDesc"),
  stepIO: document.getElementById("stepIO"),
  stepOp: document.getElementById("stepOp"),
  stepHint: document.getElementById("stepHint"),
  attentionGrid: document.getElementById("attentionGrid"),
  prevStep: document.getElementById("prevStep"),
  nextStep: document.getElementById("nextStep"),
  playPause: document.getElementById("playPause"),
  modelButtons: Array.from(document.querySelectorAll(".model-btn")),
  bertExamplePanel: document.getElementById("bertExamplePanel"),
  flashExperience: document.getElementById("flashExperience"),
  flowCardTokenize: document.getElementById("flowCardTokenize"),
  flowCardTensorIn: document.getElementById("flowCardTensorIn"),
  flowCardBertTensor: document.getElementById("flowCardBertTensor"),
  flowCardClsHead: document.getElementById("flowCardClsHead"),
  exRawText: document.getElementById("exRawText"),
  exTokens: document.getElementById("exTokens"),
  exInputIds: document.getElementById("exInputIds"),
  exAttentionMask: document.getElementById("exAttentionMask"),
  exTokenTypeIds: document.getElementById("exTokenTypeIds"),
  exHiddenShape: document.getElementById("exHiddenShape"),
  exPoolerShape: document.getElementById("exPoolerShape"),
  exHiddenSample: document.getElementById("exHiddenSample"),
  exClsCalc: document.getElementById("exClsCalc"),
  exLogits: document.getElementById("exLogits"),
  exProbs: document.getElementById("exProbs"),
  flashSceneTitle: document.getElementById("flashSceneTitle"),
  flashSceneDesc: document.getElementById("flashSceneDesc"),
  flashTimeline: document.getElementById("flashTimeline"),
  flashHBMNote: document.getElementById("flashHBMNote"),
  flashSRAMNote: document.getElementById("flashSRAMNote"),
  flashHBMQ: document.getElementById("flashHBMQ"),
  flashHBMK: document.getElementById("flashHBMK"),
  flashHBMV: document.getElementById("flashHBMV"),
  flashHBMO: document.getElementById("flashHBMO"),
  flashHBMState: document.getElementById("flashHBMState"),
  flashHBMS: document.getElementById("flashHBMS"),
  flashHBMP: document.getElementById("flashHBMP"),
  flashSNote: document.getElementById("flashSNote"),
  flashPNote: document.getElementById("flashPNote"),
  flashTransferChips: document.getElementById("flashTransferChips"),
  flashTransferList: document.getElementById("flashTransferList"),
  flashSRAMQ: document.getElementById("flashSRAMQ"),
  flashSRAMK: document.getElementById("flashSRAMK"),
  flashSRAMV: document.getElementById("flashSRAMV"),
  flashSRAMS: document.getElementById("flashSRAMS"),
  flashRegisters: document.getElementById("flashRegisters"),
  flashRegisterNarrative: document.getElementById("flashRegisterNarrative"),
  flashNarrative: document.getElementById("flashNarrative"),
  flashMath: document.getElementById("flashMath"),
  flashInsight: document.getElementById("flashInsight"),
  flashStandardComplexity: document.getElementById("flashStandardComplexity"),
  flashFlashComplexity: document.getElementById("flashFlashComplexity"),
  flashHBMComplexity: document.getElementById("flashHBMComplexity")
};

const BERT_EXAMPLE = {
  batchSize: 3,
  texts: [
    "这家餐厅上菜有点慢，但味道很不错。",
    "物流速度很快，包装也很完整。",
    "剧情一般，不过演员表演还可以。"
  ],
  tokens: [
    ["[CLS]", "这家", "餐厅", "上菜", "有点", "慢", "，", "但", "味道", "很", "不错", "。", "[SEP]"],
    ["[CLS]", "物流", "速度", "很", "快", "，", "包装", "也", "很", "完整", "。", "[SEP]", "[PAD]"],
    ["[CLS]", "剧情", "一般", "，", "不过", "演员", "表演", "还", "可以", "。", "[SEP]", "[PAD]", "[PAD]"]
  ],
  inputIds: [
    [101, 6821, 2157, 762, 3300, 2714, 8024, 852, 1456, 2523, 7231, 511, 102],
    [101, 3300, 686, 2523, 2571, 8024, 5392, 738, 2523, 5310, 511, 102, 0],
    [101, 3819, 3125, 8024, 679, 782, 4801, 6820, 1377, 511, 102, 0, 0]
  ],
  attentionMask: [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  ],
  tokenTypeIds: [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ],
  hiddenShape: "last_hidden_state shape: [3, 13, 768]",
  poolerShape: "pooler_output shape: [3, 768]",
  hiddenSample: [
    "sample[0] last_hidden_state[0,0,0:8] = [-0.18, 0.42, 0.11, -0.57, 0.03, 0.75, -0.26, 0.08]",
    "sample[1] last_hidden_state[1,0,0:8] = [-0.05, 0.38, -0.12, -0.09, 0.44, 0.52, -0.31, 0.19]",
    "sample[2] last_hidden_state[2,0,0:8] = [0.14, -0.08, 0.27, -0.22, 0.07, 0.16, -0.03, 0.41]"
  ],
  clsCalc: "H_cls = last_hidden_state[:, 0, :]\nlogits = H_cls @ W + b",
  logits: [
    [-0.73, 1.91],
    [-1.22, 2.56],
    [0.48, 0.44]
  ],
  probs: [
    [0.066, 0.934],
    [0.022, 0.978],
    [0.510, 0.490]
  ],
  predLabels: ["正向", "正向", "中性"]
};

const STANDARD_MODELS = {
  resnet: {
    title: "ResNet-50 推理流程",
    viewBox: "0 0 1360 560",
    nodes: [
      { id: "input", label: "输入\n224 x 224 x 3", x: 32, y: 232, w: 120, h: 60, type: "main" },
      { id: "stem", label: "Stem\nConv7x7 + Pool", x: 188, y: 232, w: 135, h: 60, type: "main" },
      { id: "conv2", label: "Conv2_x\n3 Bottleneck", x: 364, y: 232, w: 135, h: 60, type: "residual" },
      { id: "conv3", label: "Conv3_x\n4 Bottleneck", x: 540, y: 232, w: 135, h: 60, type: "residual" },
      { id: "conv4", label: "Conv4_x\n6 Bottleneck", x: 716, y: 232, w: 135, h: 60, type: "residual" },
      { id: "conv5", label: "Conv5_x\n3 Bottleneck", x: 892, y: 232, w: 135, h: 60, type: "residual" },
      { id: "gap", label: "全局平均池化\nGlobal AvgPool", x: 1068, y: 232, w: 120, h: 60, type: "main" },
      { id: "fc", label: "FC\n1000 类别", x: 1218, y: 232, w: 100, h: 60, type: "main" }
    ],
    edges: [
      { id: "e1", from: "input", to: "stem" },
      { id: "e2", from: "stem", to: "conv2" },
      { id: "e3", from: "conv2", to: "conv3" },
      { id: "e4", from: "conv3", to: "conv4" },
      { id: "e5", from: "conv4", to: "conv5" },
      { id: "e6", from: "conv5", to: "gap" },
      { id: "e7", from: "gap", to: "fc" },
      { id: "s1", from: "stem", to: "conv3", kind: "skip", curve: -140 },
      { id: "s2", from: "conv2", to: "conv4", kind: "skip", curve: -140 },
      { id: "s3", from: "conv3", to: "conv5", kind: "skip", curve: -140 }
    ],
    steps: [
      {
        title: "1) 输入进入 Stem",
        desc: "图像先经过大卷积核和池化层，快速提取低层纹理并降低分辨率。",
        io: "224x224x3 -> 56x56x64",
        op: "Conv, BN, ReLU, MaxPool",
        hint: "先观察输入如何被压缩成更适合深层网络处理的特征图。",
        nodes: ["input", "stem"],
        edges: ["e1"]
      },
      {
        title: "2) Conv2_x 残差学习",
        desc: "浅层 Bottleneck 学习基础语义，shortcut 让信息与梯度更容易流过整个块。",
        io: "56x56x64 -> 56x56x256",
        op: "1x1, 3x3, 1x1 + Identity Add",
        hint: "虚线代表 shortcut 支路，主分支与支路最终做残差相加。",
        nodes: ["conv2"],
        edges: ["e2", "s1"]
      },
      {
        title: "3) Conv3_x 语义抽象",
        desc: "特征图尺寸继续缩小，但每个位置看到的语义范围更大。",
        io: "56x56x256 -> 28x28x512",
        op: "Stride Downsample + Residual Blocks",
        hint: "这一层开始从纹理走向更稳定的部件级语义。",
        nodes: ["conv3"],
        edges: ["e3", "s2"]
      },
      {
        title: "4) Conv4_x 深层特征",
        desc: "计算最密集的一段，用来组合更高阶的结构与语义概念。",
        io: "28x28x512 -> 14x14x1024",
        op: "Bottleneck x6",
        hint: "主干路径继续加深，skip connection 仍在帮助信息稳定传播。",
        nodes: ["conv4"],
        edges: ["e4", "s3"]
      },
      {
        title: "5) Conv5_x 到池化",
        desc: "最后一组卷积输出高层语义，再通过全局平均池化压缩成分类向量。",
        io: "14x14x1024 -> 1x1x2048",
        op: "Residual Blocks + Global Average Pooling",
        hint: "观察空间维度如何被压成语义向量。",
        nodes: ["conv5", "gap"],
        edges: ["e5", "e6"]
      },
      {
        title: "6) 全连接输出类别",
        desc: "池化后的向量进入全连接层，输出 logits，再经过 softmax 得到分类概率。",
        io: "2048 -> 1000",
        op: "Linear + Softmax",
        hint: "这一步把抽象语义映射到最终类别空间。",
        nodes: ["fc"],
        edges: ["e7"]
      }
    ]
  },
  bert: {
    title: "BERT 编码器推理流程",
    viewBox: "0 0 1360 560",
    nodes: [
      { id: "tokens", label: "Token IDs", x: 42, y: 250, w: 120, h: 60, type: "main" },
      { id: "embed", label: "词向量 + 位置\nEmbedding", x: 202, y: 250, w: 142, h: 60, type: "main" },
      { id: "attn1", label: "Layer 1\nMulti-Head Attn", x: 402, y: 156, w: 150, h: 62, type: "transformer" },
      { id: "ffn1", label: "Layer 1\nFFN", x: 606, y: 156, w: 116, h: 62, type: "transformer" },
      { id: "norm1", label: "Layer 1\nAdd + Norm", x: 776, y: 156, w: 132, h: 62, type: "transformer" },
      { id: "attn2", label: "Layer 2\nMulti-Head Attn", x: 402, y: 344, w: 150, h: 62, type: "transformer" },
      { id: "ffn2", label: "Layer 2\nFFN", x: 606, y: 344, w: 116, h: 62, type: "transformer" },
      { id: "norm2", label: "Layer 2\nAdd + Norm", x: 776, y: 344, w: 132, h: 62, type: "transformer" },
      { id: "head", label: "任务头\n取 [CLS]", x: 964, y: 250, w: 128, h: 60, type: "main" },
      { id: "output", label: "输出\nLogits", x: 1142, y: 250, w: 120, h: 60, type: "main" }
    ],
    edges: [
      { id: "e1", from: "tokens", to: "embed" },
      { id: "e2", from: "embed", to: "attn1", fromSide: "right", toSide: "left" },
      { id: "e3", from: "attn1", to: "ffn1" },
      { id: "e4", from: "ffn1", to: "norm1" },
      { id: "e5", from: "norm1", to: "attn2", fromSide: "bottom", toSide: "top", curve: 90 },
      { id: "e6", from: "attn2", to: "ffn2" },
      { id: "e7", from: "ffn2", to: "norm2" },
      { id: "e8", from: "norm2", to: "head", fromSide: "right", toSide: "left", curve: -80 },
      { id: "e9", from: "head", to: "output" },
      { id: "s1", from: "embed", to: "norm1", kind: "skip", curve: -120 },
      { id: "s2", from: "norm1", to: "norm2", kind: "skip", curve: 120 }
    ],
    steps: [
      {
        title: "1) 文本分词",
        desc: "原始文本被切成 token ids，并按任务需要加入 [CLS]、[SEP] 等特殊标记。",
        io: "文本 -> Token IDs",
        op: "Tokenizer",
        hint: "先把离散文本变成模型可处理的符号序列。",
        nodes: ["tokens"],
        edges: []
      },
      {
        title: "2) 构造 Embedding",
        desc: "词向量、位置向量和 segment 向量相加，得到送入 Transformer 的连续表示。",
        io: "ids -> X0 (seq_len x hidden)",
        op: "Lookup + Add",
        hint: "这里完成从离散符号到连续张量的映射。",
        nodes: ["tokens", "embed"],
        edges: ["e1"]
      },
      {
        title: "3) 第一层自注意力",
        desc: "多头注意力让每个 token 都能看到上下文，建立词与词之间的重要关联。",
        io: "X0 -> X1_attn",
        op: "QK^T / sqrt(d), Softmax, Weighted Sum",
        hint: "右侧热力图越深，说明该 token 更关注对应位置。",
        nodes: ["attn1"],
        edges: ["e2"],
        attention: [
          [0.42, 0.09, 0.08, 0.16, 0.15, 0.10],
          [0.14, 0.33, 0.10, 0.18, 0.14, 0.11],
          [0.10, 0.13, 0.37, 0.12, 0.17, 0.11],
          [0.12, 0.17, 0.13, 0.31, 0.15, 0.12],
          [0.13, 0.14, 0.18, 0.15, 0.29, 0.11],
          [0.16, 0.13, 0.10, 0.11, 0.14, 0.36]
        ]
      },
      {
        title: "4) 第一层 FFN 与归一化",
        desc: "FFN 对每个 token 做非线性映射，再通过残差连接和 LayerNorm 稳定特征。",
        io: "X1_attn -> X1",
        op: "Linear-GELU-Linear + Add&Norm",
        hint: "BERT 的每层都包含 attention 子层和 FFN 子层。",
        nodes: ["ffn1", "norm1"],
        edges: ["e3", "e4", "s1"]
      },
      {
        title: "5) 第二层继续堆叠",
        desc: "第二层再次执行 attention + FFN，让语义表示更深、更稳定。",
        io: "X1 -> X2_attn -> X2",
        op: "MHA + FFN + Add&Norm",
        hint: "跨层堆叠让模型逐步形成句子级语义。",
        nodes: ["attn2", "ffn2", "norm2"],
        edges: ["e5", "e6", "e7", "s2"],
        attention: [
          [0.50, 0.07, 0.07, 0.12, 0.14, 0.10],
          [0.12, 0.41, 0.09, 0.16, 0.12, 0.10],
          [0.08, 0.10, 0.45, 0.11, 0.17, 0.09],
          [0.10, 0.13, 0.11, 0.40, 0.15, 0.11],
          [0.11, 0.12, 0.15, 0.14, 0.39, 0.09],
          [0.14, 0.10, 0.08, 0.10, 0.12, 0.46]
        ]
      },
      {
        title: "6) 任务头输出",
        desc: "最后取 [CLS] 表示送入任务头，输出 logits 作为下游任务结果。",
        io: "X2[CLS] -> logits",
        op: "Pooling + Linear",
        hint: "分类任务通常直接使用 [CLS] 位置的表示。",
        nodes: ["head", "output"],
        edges: ["e8", "e9"]
      }
    ]
  }
};

const FLASH_DATA = {
  q: [
    [1, 0],
    [0, 1],
    [1, 1],
    [-1, 1]
  ],
  k: [
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 0]
  ],
  v: [
    [10, 0],
    [0, 8],
    [6, 6],
    [20, 0]
  ],
  rowLabelsQ: ["q1", "q2", "q3", "q4"],
  rowLabelsK: ["k1", "k2", "k3", "k4"],
  rowLabelsV: ["v1", "v2", "v3", "v4"],
  rowLabelsO: ["o1", "o2", "o3", "o4"],
  colLabels2: ["d1", "d2"],
  qTileRows: [0, 1],
  kvTiles: [
    [0, 1],
    [2, 3]
  ],
  focusRow: 0
};

const FLASH_DERIVED = buildFlashDerived();

const FLASH_SCENES = [
  {
    label: "Standard",
    title: "先看标准 attention 会把什么放进 HBM",
    desc: "标准 attention 会先 materialize 整个 S = QKᵀ，再 materialize 整个 P = softmax(S)。FlashAttention 的重点不是改结果，而是尽量不让这些 N x N 中间矩阵落到 HBM。",
    hbmNote: "global tensors + standard intermediate matrices",
    sramNote: "not using tiled on-chip state yet",
    transferChips: ["Read Q, K", "Write full S", "Read S, Write full P", "Read P, V"],
    transfers: [
      "先从 HBM 读 Q 和 K，计算完整 4 x 4 的 score matrix S。",
      "把整个 S 写回 HBM。",
      "再读回 S，做 stable softmax，得到整个 P，并再次写回 HBM。",
      "最后读回 P 和 V，得到输出 O。"
    ],
    qRows: [],
    kvRows: [],
    hbmOValues: placeholderOutput(FLASH_DATA.rowLabelsO.length, 2),
    hbmStateRows: [
      ["m", "-", "-", "-", "-"],
      ["l", "-", "-", "-", "-"],
      ["O partial", "...", "...", "...", "..."]
    ],
    sram: {
      q: null,
      k: null,
      v: null,
      s: null
    },
    registers: [
      { label: "m", value: "not started" },
      { label: "l", value: "not started" },
      { label: "r", value: "not started" }
    ],
    registerNarrative: "这一页只是建立对比基线：标准 attention 会在 HBM 中显式存 S 和 P。",
    narrative: [
      "目标行先看 q1 = [1, 0]。",
      "它和全部 K 的点积是 [1, 0, 1, 2]。",
      "标准做法会把整块 S 和整块 P 都 materialize 出来。"
    ].join("\n"),
    math: [
      "S = QK^T",
      "P = softmax(S)",
      "O = PV",
      "",
      "q1 对应的一整行 score:",
      "[1, 0, 1, 2]"
    ].join("\n"),
    insight: [
      "标准 attention 的瓶颈不只是算 QK^T 和 PV。",
      "更重的是：S 和 P 两个 N x N 大矩阵会反复写回和读回 HBM。",
      "FlashAttention 接下来要做的，就是把这些中间结果留在片上。"
    ].join("\n"),
    sNote: "标准 attention 会把整个 score matrix S 写回 HBM。",
    pNote: "标准 attention 会把整个 probability matrix P 写回 HBM。"
  },
  {
    label: "Load Tile 1",
    title: "把第一个 Q tile 和第一个 K/V tile 拉进 SRAM",
    desc: "现在改成 FlashAttention 的视角。我们固定一个 Q tile，把第一个 K/V tile 从 HBM 流式搬到 SRAM。注意：Qᵢ 会尽量停留在片上，后面只继续换 Kⱼ / Vⱼ。",
    hbmNote: "source of Q, K, V tiles",
    sramNote: "Q_i and K_j/V_j are now on chip",
    transferChips: ["Read Q[1:2]", "Read K[1:2]", "Read V[1:2]"],
    transfers: [
      "把 q1、q2 对应的两行加载为 Q_i。",
      "把 k1、k2 和 v1、v2 加载为第一个 K/V tile。",
      "此时 HBM 里还没有 S 和 P，因为我们不会先算整块再落盘。"
    ],
    qRows: FLASH_DATA.qTileRows,
    kvRows: FLASH_DATA.kvTiles[0],
    hbmOValues: placeholderOutput(FLASH_DATA.rowLabelsO.length, 2),
    hbmStateRows: [
      ["m", "-inf", "-", "-", "-"],
      ["l", "0", "-", "-", "-"],
      ["O partial", "[0,0]", "-", "-", "-"]
    ],
    sram: {
      q: sliceRows(FLASH_DATA.q, FLASH_DATA.qTileRows),
      k: sliceRows(FLASH_DATA.k, FLASH_DATA.kvTiles[0]),
      v: sliceRows(FLASH_DATA.v, FLASH_DATA.kvTiles[0]),
      s: null
    },
    registers: [
      { label: "m", value: "-inf" },
      { label: "l", value: "0" },
      { label: "r", value: "[0, 0]" }
    ],
    registerNarrative: "初始化当前焦点行 q1 的 online 状态：m = -inf, l = 0, r = [0, 0]。",
    narrative: [
      "当前只看第一个 tile：Q_i = [q1, q2]，K_1 = [k1, k2]，V_1 = [v1, v2]。",
      "这些小块都被搬到 SRAM，所以后续点积和 softmax 准备都在片上完成。",
      "关键变化：没有整块 S / P 的 HBM 落盘动作。"
    ].join("\n"),
    math: [
      "Br = 2, Bc = 2",
      "Q_i = [[1,0],[0,1]]",
      "K_1 = [[1,0],[0,1]]",
      "V_1 = [[10,0],[0,8]]"
    ].join("\n"),
    insight: [
      "FlashAttention 的工作方式像流水线。",
      "Q_i 留在片上，K_j / V_j 一块一块流过。",
      "这样就把大部分访存压力从 HBM 移到了更快的 SRAM。"
    ].join("\n"),
    sNote: "FlashAttention 不会在这一步把整块 S 写入 HBM。",
    pNote: "FlashAttention 也不会提前创建整个 P。"
  },
  {
    label: "Compute S11",
    title: "在 SRAM 里直接算出局部 score tile S₁₁",
    desc: "一旦 Q_i 和 K_1 都在片上，就可以直接算局部 score tile S_11。这里我们重点跟踪 q1 这一行，它在当前 tile 里只看到了 [1, 0]。",
    hbmNote: "HBM only stores source tensors; local score stays on chip",
    sramNote: "Q_i, K_1, V_1 and S_11 all live on chip",
    transferChips: ["Q_i stays on chip", "Compute S_11 in SRAM"],
    transfers: [
      "没有新的 HBM 往返，当前 tile 的 score 直接在 SRAM 中生成。",
      "q1 对 tile 1 的局部分数是 [1, 0]。",
      "整块 S_11 的结果也只会停留在片上。"
    ],
    qRows: FLASH_DATA.qTileRows,
    kvRows: FLASH_DATA.kvTiles[0],
    hbmOValues: placeholderOutput(FLASH_DATA.rowLabelsO.length, 2),
    hbmStateRows: [
      ["m", "-inf", "-", "-", "-"],
      ["l", "0", "-", "-", "-"],
      ["O partial", "[0,0]", "-", "-", "-"]
    ],
    sram: {
      q: sliceRows(FLASH_DATA.q, FLASH_DATA.qTileRows),
      k: sliceRows(FLASH_DATA.k, FLASH_DATA.kvTiles[0]),
      v: sliceRows(FLASH_DATA.v, FLASH_DATA.kvTiles[0]),
      s: FLASH_DERIVED.scoreTiles[0]
    },
    registers: [
      { label: "focus row", value: "q1" },
      { label: "local scores", value: "[1, 0]" },
      { label: "S_11", value: "2 x 2 on chip" }
    ],
    registerNarrative: "当前只是生成局部 score tile，还没有把它合并进最终 softmax 状态。",
    narrative: [
      "局部 score tile S_11 = Q_i K_1^T。",
      "对 q1 而言，这一步只看到了 q1·k1 = 1 和 q1·k2 = 0。",
      "此时它还不知道后面的 k3、k4 会不会带来更大的分数。"
    ].join("\n"),
    math: [
      "S_11 = [[1, 0],",
      "        [0, 1]]",
      "",
      "focus row q1:",
      "[1, 0]"
    ].join("\n"),
    insight: [
      "FlashAttention 并不是先得到完整 [1, 0, 1, 2] 再做 softmax。",
      "它只看到局部块 [1, 0]，所以 softmax 状态也必须支持在线更新。",
      "这就是 online softmax 存在的原因。"
    ].join("\n"),
    sNote: "注意：这里只生成 S_11 这个小 tile，整块 S 从未进入 HBM。",
    pNote: "因为还没做全局 softmax，所以更不可能提前有整块 P。"
  },
  {
    label: "Init mlr",
    title: "用 tile 1 初始化 q1 的 online softmax 状态",
    desc: "对 q1 来说，目前只见过 [1, 0]。所以此时的 m / l / r 就是“到当前为止这个前缀”的精确 stable softmax 状态。",
    hbmNote: "row state exists in HBM and is loaded into registers for this tile",
    sramNote: "current working copy of m / l / O_i is updated on chip",
    transferChips: ["Load m/l/O_i row state", "Update on chip", "Write row state back"],
    transfers: [
      "这里要更准确地看：跨 tile 的行状态 m_i / l_i / O_i 需要在 HBM 中保留。",
      "当前这个 tile 开始时，会把 q1 这一行的状态从 HBM 读到寄存器 / SRAM 工作区。",
      "对 q1，tile 1 的局部最大值是 1。",
      "更新完成后，新的 m / l / O_i 会再次写回 HBM，供下一个 K/V tile 继续读取。"
    ],
    qRows: FLASH_DATA.qTileRows,
    kvRows: FLASH_DATA.kvTiles[0],
    hbmOValues: placeholderOutput(FLASH_DATA.rowLabelsO.length, 2),
    hbmStateRows: [
      ["m", "1", "-", "-", "-"],
      ["l", "1.3679", "-", "-", "-"],
      ["O partial", "[7.3106,2.1512]", "-", "-", "-"]
    ],
    sram: {
      q: sliceRows(FLASH_DATA.q, FLASH_DATA.qTileRows),
      k: sliceRows(FLASH_DATA.k, FLASH_DATA.kvTiles[0]),
      v: sliceRows(FLASH_DATA.v, FLASH_DATA.kvTiles[0]),
      s: [
        ["e^(1-1)=1", "e^(0-1)=0.3679"]
      ]
    },
    registers: [
      { label: "m^(1)", value: "1" },
      { label: "l^(1)", value: "1.3679" },
      { label: "r^(1)", value: "[10.0000, 2.9432]" },
      { label: "O_i^(1)", value: "[7.3106, 2.1512]" }
    ],
    registerNarrative: "q1 after tile 1:\nm = 1\nl = 1 + e^(-1) = 1.3679\nr = 1*v1 + e^(-1)*v2 = [10.0000, 2.9432]\nO_i = r / l = [7.3106, 2.1512]\n这份行状态会写回 HBM，等待下一块 K/V tile。",
    narrative: [
      "对 q1 的局部分数 [1, 0]，局部最大值 m = 1。",
      "在这个坐标系下，l = 1 + e^(-1) = 1.3679。",
      "同时维护分子向量 r = Σ exp(score - m) * v。"
    ].join("\n"),
    math: [
      "m^(1) = 1",
      "l^(1) = e^(1-1) + e^(0-1) = 1 + 0.3679 = 1.3679",
      "r^(1) = 1*[10,0] + 0.3679*[0,8]",
      "       = [10.0000, 2.9432]",
      "O_i^(1) = r^(1) / l^(1) = [7.3106, 2.1512]"
    ].join("\n"),
    insight: [
      "这里最值得记住的是：r 不是最终输出，而是 softmax 分子的累计向量。",
      "实现层面上，跨 tile 的状态并不是永远只在寄存器里常驻；它会在每轮 tile 之间写回 HBM 再读出。",
      "后面如果看到更大的最大值，旧的 l 和 r 会整体乘上一个缩放因子，这样就能把旧状态换到新的坐标系下。"
    ].join("\n"),
    sNote: "tile 1 的 score 只作为片上的临时值参与 m / l / r 更新。",
    pNote: "整块概率矩阵 P 仍然不存在，只有 q1 的行级摘要状态。"
  },
  {
    label: "Stream Tile 2",
    title: "继续流入第二个 K/V tile，发现更大的分数",
    desc: "现在 Q_i 不动，只把第二个 K/V tile 流进来。对 q1 来说，新的局部分数变成 [1, 2]，于是局部最大值从旧的 1 提升到了 2。",
    hbmNote: "Q_i stays; stream K_2 and V_2 from HBM",
    sramNote: "second KV tile replaces the previous one on chip",
    transferChips: ["Keep Q_i on chip", "Read K[3:4]", "Read V[3:4]", "Reload m/l/O_i"],
    transfers: [
      "Q_i 继续留在片上，不需要再次从 HBM 读取。",
      "只流入第二块 K_2 / V_2，也就是 k3、k4 与 v3、v4。",
      "与此同时，要把上一轮写回 HBM 的 q1 行状态 m / l / O_i 再读回片上。",
      "新局部分数 [1, 2] 告诉我们：真正更大的全局最大值出现了。"
    ],
    qRows: FLASH_DATA.qTileRows,
    kvRows: FLASH_DATA.kvTiles[1],
    hbmOValues: placeholderOutput(FLASH_DATA.rowLabelsO.length, 2),
    hbmStateRows: [
      ["m", "1 → reload", "-", "-", "-"],
      ["l", "1.3679 → reload", "-", "-", "-"],
      ["O partial", "[7.3106,2.1512] → reload", "-", "-", "-"]
    ],
    sram: {
      q: sliceRows(FLASH_DATA.q, FLASH_DATA.qTileRows),
      k: sliceRows(FLASH_DATA.k, FLASH_DATA.kvTiles[1]),
      v: sliceRows(FLASH_DATA.v, FLASH_DATA.kvTiles[1]),
      s: FLASH_DERIVED.scoreTiles[1]
    },
    registers: [
      { label: "m_old", value: "1" },
      { label: "m_block", value: "2" },
      { label: "local row", value: "[1, 2]" },
      { label: "O_i old", value: "[7.3106, 2.1512]" }
    ],
    registerNarrative: "tile 2 for q1:\nfirst reload previous row state from HBM:\nm_old = 1, l_old = 1.3679, O_old = [7.3106,2.1512]\nthen compute local scores = [1, 2]\nm_hat = 2\nl_hat = e^(1-2) + e^(2-2) = 1.3679\nr_hat = e^(-1)*v3 + 1*v4 = [22.2074, 2.2074]",
    narrative: [
      "新的局部 score tile 是 S_12。",
      "对 q1 而言，这一步只看当前 tile 会得到 [1, 2]。",
      "最重要的变化是：新的局部最大值 2 比旧状态里的最大值 1 更大。"
    ].join("\n"),
    math: [
      "local scores = [1, 2]",
      "m_hat = 2",
      "l_hat = e^(1-2) + e^(2-2) = 0.3679 + 1 = 1.3679",
      "r_hat = e^(-1)*[6,6] + 1*[20,0]",
      "      = [22.2074, 2.2074]"
    ].join("\n"),
    insight: [
      "这就是 online softmax 最关键的拐点。",
      "旧状态是按“减 1”的坐标系算的，现在新块要求我们改成“减 2”的坐标系。",
      "实现上，这个旧状态正是刚从 HBM 重新载入片上的那一份，所以旧的 l 和 r 必须整体重标定。"
    ].join("\n"),
    sNote: "tile 2 的局部 score S_12 依然只存在于 SRAM 中。",
    pNote: "我们不会先给 tile 2 单独算一个块内 softmax 再拼接，那样是错的。"
  },
  {
    label: "Merge + Write O",
    title: "重标定旧状态，得到最终输出，再把 O 写回 HBM",
    desc: "最后一步把 tile 1 的旧状态整体乘上 e^(m_old - m_new)，再与 tile 2 的局部状态合并。这样就得到和完整 stable softmax 完全一致的结果，然后只把最终输出 O 写回 HBM。",
    hbmNote: "write updated row state and final O row back to HBM",
    sramNote: "merge happens on chip; only compact row state returns to HBM",
    transferChips: ["Rebase old state", "Merge tile 2", "Write m/l/O_i", "Write O[1]"],
    transfers: [
      "旧状态从 m = 1 的坐标系切换到 m = 2 的坐标系。",
      "合并后得到全局 l 和全局 r。",
      "更准确地说：更新后的行状态 m / l / O_i 会写回 HBM，最终输出 O(q1) 也会写回 HBM。",
      "S 和 P 从头到尾都没有在 HBM 中 materialize。"
    ],
    qRows: FLASH_DATA.qTileRows,
    kvRows: FLASH_DATA.kvTiles[1],
    hbmOValues: FLASH_DERIVED.outputDisplay,
    hbmStateRows: [
      ["m", "2", "-", "-", "-"],
      ["l", "1.8711", "-", "-", "-"],
      ["O partial", "[13.8339,1.7589]", "-", "-", "-"]
    ],
    sram: {
      q: sliceRows(FLASH_DATA.q, FLASH_DATA.qTileRows),
      k: sliceRows(FLASH_DATA.k, FLASH_DATA.kvTiles[1]),
      v: sliceRows(FLASH_DATA.v, FLASH_DATA.kvTiles[1]),
      s: [
        ["weights", "[0.1966, 0.0723, 0.1966, 0.5344]"],
        ["O(q1)", "[13.8339, 1.7589]"]
      ]
    },
    registers: [
      { label: "m_final", value: "2" },
      { label: "l_final", value: "1.8711" },
      { label: "r_final", value: "[25.8869, 3.2902]" },
      { label: "O_i final", value: "[13.8339, 1.7589]" }
    ],
    registerNarrative: "merge for q1:\nl_new = e^(1-2)*1.3679 + 1.3679 = 1.8711\nr_new = e^(1-2)*[10,2.9432] + [22.2074,2.2074]\n     = [25.8869, 3.2902]\nO(q1) = r_new / l_new = [13.8339, 1.7589]\nthen write updated m/l/O_i row state back to HBM",
    narrative: [
      "这一步完成 online softmax 的核心合并。",
      "先把旧状态整体乘上 e^(1-2) = e^(-1)，再加上 tile 2 的局部状态。",
      "最后输出 O(q1) 和标准 attention 对整行 [1,0,1,2] 做 stable softmax 完全一致。"
    ].join("\n"),
    math: [
      "m_new = max(1, 2) = 2",
      "l_new = e^(1-2)*1.3679 + 1*1.3679 = 1.8711",
      "r_new = e^(1-2)*[10,2.9432] + [22.2074,2.2074]",
      "      = [25.8869, 3.2902]",
      "O(q1) = r_new / l_new = [13.8339, 1.7589]"
    ].join("\n"),
    insight: [
      "现在可以把 FlashAttention 记成一句话：",
      "Q_i 留在片上，K_j / V_j 流式经过；局部 score 只在 SRAM 短暂停留；online softmax 只维护 m / l / r；最终只把 O 写回 HBM。",
      "更精确地说：它减少的是 S/P 这类 N² 级中间矩阵的 HBM 往返，而不是所有状态完全不碰 HBM。"
    ].join("\n"),
    sNote: "S 从头到尾没有形成完整的 HBM 张量，只存在局部 tile。",
    pNote: "P 也没有形成完整的 HBM 张量，softmax 只被压缩成行级状态 m / l / r。"
  }
];

function init() {
  bindEvents();
  renderBertExample();
  renderFlashTimeline();
  renderCurrentView();
}

function bindEvents() {
  ui.modelButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.model === state.modelKey) {
        return;
      }

      state.modelKey = btn.dataset.model;
      state.stepIndex = 0;
      stopAutoplay();

      ui.modelButtons.forEach((item) => {
        const active = item === btn;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      });

      renderCurrentView();
    });
  });

  ui.prevStep.addEventListener("click", () => shiftStep(-1));
  ui.nextStep.addEventListener("click", () => shiftStep(1));

  ui.playPause.addEventListener("click", () => {
    if (state.timer) {
      stopAutoplay();
    } else {
      startAutoplay();
    }
  });
}

function renderCurrentView() {
  if (state.modelKey === "flash") {
    renderFlashView();
    return;
  }

  renderStandardView();
}

function renderStandardView() {
  const model = STANDARD_MODELS[state.modelKey];
  ui.contentGrid.classList.remove("hidden");
  ui.flashExperience.classList.add("hidden");
  ui.bertExamplePanel.classList.toggle("hidden", state.modelKey !== "bert");

  ui.graphTitle.textContent = model.title;
  ui.graph.setAttribute("viewBox", model.viewBox);

  drawGraph(model);
  renderStandardStep();
}

function renderStandardStep() {
  const model = STANDARD_MODELS[state.modelKey];
  const step = model.steps[state.stepIndex];

  ui.stepBadge.textContent = `步骤 ${state.stepIndex + 1} / ${model.steps.length}`;
  ui.stepTitle.textContent = step.title;
  ui.stepDesc.textContent = step.desc;
  ui.stepIO.textContent = step.io;
  ui.stepOp.textContent = step.op;
  ui.stepHint.textContent = step.hint;

  Object.values(state.nodes).forEach((node) => node.classList.remove("active"));
  Object.values(state.edges).forEach((edge) => edge.classList.remove("active"));

  step.nodes.forEach((id) => {
    if (state.nodes[id]) {
      state.nodes[id].classList.add("active");
    }
  });

  step.edges.forEach((id) => {
    if (state.edges[id]) {
      state.edges[id].classList.add("active");
    }
  });

  renderAttention(step.attention);
  updateBertExampleFocus();
}

function renderFlashView() {
  ui.contentGrid.classList.add("hidden");
  ui.bertExamplePanel.classList.add("hidden");
  ui.flashExperience.classList.remove("hidden");

  renderFlashTimeline();
  renderFlashScene();
}

function renderFlashTimeline() {
  ui.flashTimeline.innerHTML = "";

  FLASH_SCENES.forEach((scene, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "timeline-btn";
    btn.classList.toggle("active", state.modelKey === "flash" && state.stepIndex === index);
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", String(state.modelKey === "flash" && state.stepIndex === index));
    btn.innerHTML = `<span class="timeline-step">step ${index + 1}</span><span class="timeline-label">${scene.label}</span>`;
    btn.addEventListener("click", () => {
      state.modelKey = "flash";
      state.stepIndex = index;
      ui.modelButtons.forEach((item) => {
        const active = item.dataset.model === "flash";
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      });
      stopAutoplay();
      renderFlashView();
    });
    ui.flashTimeline.appendChild(btn);
  });
}

function renderFlashScene() {
  const scene = FLASH_SCENES[state.stepIndex];

  ui.flashSceneTitle.textContent = `${state.stepIndex + 1}. ${scene.title}`;
  ui.flashSceneDesc.textContent = scene.desc;
  ui.flashHBMNote.textContent = scene.hbmNote;
  ui.flashSRAMNote.textContent = scene.sramNote;
  ui.flashSNote.textContent = scene.sNote;
  ui.flashPNote.textContent = scene.pNote;
  ui.flashSNote.classList.toggle("emphasis", state.stepIndex > 0);
  ui.flashPNote.classList.toggle("emphasis", state.stepIndex > 0);

  renderMatrix(
    ui.flashHBMQ,
    FLASH_DATA.q,
    FLASH_DATA.rowLabelsQ,
    FLASH_DATA.colLabels2,
    {
      layout: "q-two",
      activeRows: scene.qRows,
      focusRow: FLASH_DATA.focusRow
    }
  );

  renderMatrix(
    ui.flashHBMK,
    FLASH_DATA.k,
    FLASH_DATA.rowLabelsK,
    FLASH_DATA.colLabels2,
    {
      layout: "k-two",
      activeRows: scene.kvRows
    }
  );

  renderMatrix(
    ui.flashHBMV,
    FLASH_DATA.v,
    FLASH_DATA.rowLabelsV,
    FLASH_DATA.colLabels2,
    {
      layout: "v-two",
      activeRows: scene.kvRows
    }
  );

  renderMatrix(
    ui.flashHBMO,
    scene.hbmOValues,
    FLASH_DATA.rowLabelsO,
    FLASH_DATA.colLabels2,
    {
      layout: "o-two",
      activeRows: state.stepIndex === FLASH_SCENES.length - 1 ? [0] : [],
      outputRows: state.stepIndex === FLASH_SCENES.length - 1 ? [0] : []
    }
  );

  renderMatrix(
    ui.flashHBMState,
    scene.hbmStateRows,
    ["row-state", "row-state", "row-state"],
    ["q1", "q2", "q3", "q4"],
    {
      layout: "s-four",
      activeRows: [0, 1, 2],
      focusRow: 0
    }
  );

  renderMatrix(
    ui.flashHBMS,
    FLASH_DERIVED.scoreDisplay,
    FLASH_DATA.rowLabelsQ,
    ["k1", "k2", "k3", "k4"],
    {
      layout: "s-four",
      ghosted: state.stepIndex > 0,
      focusRow: FLASH_DATA.focusRow,
      activeRows: state.stepIndex === 0 ? [0] : []
    }
  );

  renderMatrix(
    ui.flashHBMP,
    FLASH_DERIVED.probDisplay,
    FLASH_DATA.rowLabelsQ,
    ["k1", "k2", "k3", "k4"],
    {
      layout: "p-four",
      ghosted: state.stepIndex > 0,
      focusRow: FLASH_DATA.focusRow,
      activeRows: state.stepIndex === 0 ? [0] : []
    }
  );

  renderTransferChips(scene.transferChips);
  renderTransferList(scene.transfers);
  renderSRAMMatrices(scene);
  renderRegisters(scene.registers);

  ui.flashRegisterNarrative.textContent = scene.registerNarrative;
  ui.flashNarrative.textContent = scene.narrative;
  ui.flashMath.textContent = scene.math;
  ui.flashInsight.textContent = scene.insight;
  ui.flashStandardComplexity.textContent = [
    "Standard attention",
    "1. S = QK^T",
    "   Q ∈ R^(N×d), K^T ∈ R^(d×N)",
    "   => N^2 个元素，每个元素是长度 d 的点积",
    "   => O(N^2 d)",
    "",
    "2. P = softmax(S)",
    "   每一行长度 N",
    "   rowmax + exp + rowsum + divide",
    "   => O(N^2)",
    "",
    "3. O = PV",
    "   P ∈ R^(N×N), V ∈ R^(N×d)",
    "   => O(N^2 d)",
    "",
    "总计算复杂度：",
    "O(N^2 d) + O(N^2) + O(N^2 d)",
    "= O(N^2 d)",
    "",
    "额外中间显存：",
    "需要显式存 S 和 P",
    "=> O(N^2)"
  ].join("\n");
  ui.flashFlashComplexity.textContent = [
    "FlashAttention",
    "1. 仍然要算全部 q_i · k_j",
    "   => score 计算量仍是 O(N^2 d)",
    "",
    "2. 仍然要完成 softmax 和对 V 的加权求和",
    "   => 总 FLOPs 量级仍是 O(N^2 d)",
    "",
    "关键区别不在算术复杂度，",
    "而在于不再 materialize 完整 S / P。",
    "",
    "额外中间显存：",
    "只保留 O_i、m_i、l_i 以及当前 tile scratch",
    "=> O(Nd) + O(N)",
    "≈ O(Nd)",
    "",
    "所以：",
    "标准 attention: compute O(N^2 d), memory O(N^2)",
    "FlashAttention: compute O(N^2 d), memory O(Nd)"
  ].join("\n");
  ui.flashHBMComplexity.textContent = [
    "HBM access intuition",
    "",
    "Standard attention rough path:",
    "read Q,K -> write S -> read S -> write P -> read P,V -> write O",
    "",
    "按元素数量粗算：",
    "Q/K/V/O 相关 ≈ O(Nd)",
    "S/P 相关 ≈ O(N^2)",
    "总主导项 ≈ O(N^2)",
    "",
    "FlashAttention rough path:",
    "read Q_i, read K_j/V_j tile,",
    "compute local S_ij in SRAM,",
    "reload and update row state (m_i,l_i,O_i),",
    "write compact row state back",
    "",
    "它仍然会访问 HBM：",
    "1. 读取 Q/K/V",
    "2. 读写行状态 m_i / l_i / O_i",
    "3. 写回最终 O",
    "",
    "但避免了最贵的两类 HBM 中间量：",
    "1. 完整 S ∈ R^(N×N)",
    "2. 完整 P ∈ R^(N×N)",
    "",
    "论文常写成：",
    "standard extra memory = O(N^2)",
    "flash extra memory = O(Nd)",
    "",
    "这就是为什么长序列下 FlashAttention 显存和访存优势都很明显。"
  ].join("\n");

  renderFlashTimeline();
}

function renderSRAMMatrices(scene) {
  if (scene.sram.q) {
    renderMatrix(ui.flashSRAMQ, scene.sram.q, ["q1", "q2"], FLASH_DATA.colLabels2, {
      layout: "q-two",
      activeRows: [0, 1],
      focusRow: 0
    });
  } else {
    renderMatrixMessage(ui.flashSRAMQ, "当前还没有把 Q tile 搬入 SRAM。");
  }

  if (scene.sram.k) {
    renderMatrix(ui.flashSRAMK, scene.sram.k, ["kA", "kB"], FLASH_DATA.colLabels2, {
      layout: "k-two",
      activeRows: [0, 1],
      streamRows: [0, 1]
    });
  } else {
    renderMatrixMessage(ui.flashSRAMK, "当前没有活动的 K tile。");
  }

  if (scene.sram.v) {
    renderMatrix(ui.flashSRAMV, scene.sram.v, ["vA", "vB"], FLASH_DATA.colLabels2, {
      layout: "v-two",
      activeRows: [0, 1],
      streamRows: [0, 1]
    });
  } else {
    renderMatrixMessage(ui.flashSRAMV, "当前没有活动的 V tile。");
  }

  if (scene.sram.s) {
    const rowLabels = scene.sram.s.length === 1 ? ["q1"] : ["q1", "q2"];
    const layout = scene.sram.s[0].length === 2 ? "s-two" : "s-four";
    renderMatrix(ui.flashSRAMS, scene.sram.s, rowLabels, ["c1", "c2"], {
      layout,
      activeRows: [0],
      focusRow: 0
    });
  } else {
    renderMatrixMessage(ui.flashSRAMS, "局部 score tile 还没有生成，或者此时只是在看整体思路。");
  }
}

function renderTransferChips(chips) {
  ui.flashTransferChips.innerHTML = "";
  chips.forEach((text) => {
    const chip = document.createElement("div");
    chip.className = "transfer-chip";
    chip.textContent = text;
    ui.flashTransferChips.appendChild(chip);
  });
}

function renderTransferList(items) {
  ui.flashTransferList.innerHTML = "";
  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    ui.flashTransferList.appendChild(li);
  });
}

function renderRegisters(registers) {
  ui.flashRegisters.innerHTML = "";
  registers.forEach((item) => {
    const chip = document.createElement("div");
    chip.className = "register-chip";
    chip.textContent = `${item.label}: ${item.value}`;
    ui.flashRegisters.appendChild(chip);
  });
}

function renderAttention(matrix) {
  ui.attentionGrid.innerHTML = "";

  if (state.modelKey !== "bert") {
    ui.attentionGrid.classList.add("empty");
    ui.attentionGrid.textContent = "切换到 BERT 可查看 attention heatmap。FlashAttention 已经升级为独立的全屏可视化区域。";
    return;
  }

  if (!matrix) {
    ui.attentionGrid.classList.add("empty");
    ui.attentionGrid.textContent = "当前步骤不展示 attention heatmap。";
    return;
  }

  ui.attentionGrid.classList.remove("empty");

  matrix.flat().forEach((value) => {
    const cell = document.createElement("div");
    cell.className = "att-cell";
    const lightness = 45 + value * 35;
    const alpha = 0.22 + value * 0.7;
    cell.style.background = `hsla(24, 97%, ${lightness}%, ${alpha})`;
    cell.title = `attention: ${value.toFixed(2)}`;
    ui.attentionGrid.appendChild(cell);
  });
}

function renderBertExample() {
  ui.exRawText.textContent = `batch_size = ${BERT_EXAMPLE.batchSize}\n` +
    BERT_EXAMPLE.texts.map((text, index) => `[${index}] "${text}"`).join("\n");

  ui.exTokens.textContent = formatBatchTensor("tokens", BERT_EXAMPLE.tokens);
  ui.exInputIds.textContent = formatBatchTensor("input_ids", BERT_EXAMPLE.inputIds);
  ui.exAttentionMask.textContent = formatBatchTensor("attention_mask", BERT_EXAMPLE.attentionMask);
  ui.exTokenTypeIds.textContent = formatBatchTensor("token_type_ids", BERT_EXAMPLE.tokenTypeIds);
  ui.exHiddenShape.textContent = BERT_EXAMPLE.hiddenShape;
  ui.exPoolerShape.textContent = BERT_EXAMPLE.poolerShape;
  ui.exHiddenSample.textContent = BERT_EXAMPLE.hiddenSample.join("\n");
  ui.exClsCalc.textContent = BERT_EXAMPLE.clsCalc;
  ui.exLogits.textContent = formatBatchTensor("logits", BERT_EXAMPLE.logits);
  ui.exProbs.textContent = BERT_EXAMPLE.probs
    .map((row, index) => `[${index}] probs=${JSON.stringify(row)} -> 预测标签: ${BERT_EXAMPLE.predLabels[index]}`)
    .join("\n");
}

function updateBertExampleFocus() {
  const cards = [
    ui.flowCardTokenize,
    ui.flowCardTensorIn,
    ui.flowCardBertTensor,
    ui.flowCardClsHead
  ];

  cards.forEach((card) => card.classList.remove("active"));

  if (state.modelKey !== "bert") {
    return;
  }

  let focusCard = ui.flowCardTokenize;
  if (state.stepIndex === 1) {
    focusCard = ui.flowCardTensorIn;
  } else if (state.stepIndex >= 2 && state.stepIndex <= 4) {
    focusCard = ui.flowCardBertTensor;
  } else if (state.stepIndex >= 5) {
    focusCard = ui.flowCardClsHead;
  }

  focusCard.classList.add("active");
}

function shiftStep(delta) {
  const steps = currentSteps();
  state.stepIndex = (state.stepIndex + delta + steps.length) % steps.length;
  renderCurrentView();
}

function currentSteps() {
  return state.modelKey === "flash" ? FLASH_SCENES : STANDARD_MODELS[state.modelKey].steps;
}

function startAutoplay() {
  stopAutoplay();
  ui.playPause.textContent = "暂停播放";
  state.timer = window.setInterval(() => {
    shiftStep(1);
  }, state.modelKey === "flash" ? 3200 : 2400);
}

function stopAutoplay() {
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
  ui.playPause.textContent = "自动播放";
}

function drawGraph(model) {
  state.nodes = {};
  state.edges = {};
  ui.graph.innerHTML = "";

  const defs = createSvg("defs");
  const marker = createSvg("marker", {
    id: "arrow",
    viewBox: "0 0 10 10",
    refX: "9",
    refY: "5",
    markerWidth: "6",
    markerHeight: "6",
    orient: "auto-start-reverse"
  });
  marker.appendChild(createSvg("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#89a7cc" }));
  defs.appendChild(marker);
  ui.graph.appendChild(defs);

  model.edges.forEach((edge) => {
    const path = createSvg("path", {
      d: buildEdgePath(edge, model.nodes),
      class: `edge ${edge.kind || ""}`.trim(),
      "marker-end": "url(#arrow)"
    });
    ui.graph.appendChild(path);
    state.edges[edge.id] = path;
  });

  model.nodes.forEach((node) => {
    const g = createSvg("g", { class: `node ${node.type || "main"}`, "data-id": node.id });
    const rect = createSvg("rect", {
      x: String(node.x),
      y: String(node.y),
      width: String(node.w),
      height: String(node.h)
    });
    g.appendChild(rect);

    const text = createSvg("text", {
      x: String(node.x + node.w / 2),
      y: String(node.y + node.h / 2)
    });

    const lines = node.label.split("\n");
    const startY = node.y + node.h / 2 - (lines.length - 1) * 8;
    lines.forEach((line, idx) => {
      const tspan = createSvg("tspan", {
        x: String(node.x + node.w / 2),
        y: String(startY + idx * 16)
      });
      tspan.textContent = line;
      text.appendChild(tspan);
    });

    g.appendChild(text);
    ui.graph.appendChild(g);
    state.nodes[node.id] = g;
  });
}

function buildEdgePath(edge, nodes) {
  const fromNode = nodes.find((node) => node.id === edge.from);
  const toNode = nodes.find((node) => node.id === edge.to);

  if (!fromNode || !toNode) {
    return "";
  }

  const start = getAnchor(fromNode, edge.fromSide || "right");
  const end = getAnchor(toNode, edge.toSide || "left");

  const sx = start.x;
  const sy = start.y;
  const ex = end.x;
  const ey = end.y;

  const curve = edge.curve || 0;
  const c1x = sx + (ex - sx) * 0.35;
  const c2x = sx + (ex - sx) * 0.65;
  const c1y = sy + curve;
  const c2y = ey + curve;

  return `M ${sx} ${sy} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${ex} ${ey}`;
}

function getAnchor(node, side) {
  const centerX = node.x + node.w / 2;
  const centerY = node.y + node.h / 2;

  if (side === "left") {
    return { x: node.x, y: centerY };
  }

  if (side === "top") {
    return { x: centerX, y: node.y };
  }

  if (side === "bottom") {
    return { x: centerX, y: node.y + node.h };
  }

  return { x: node.x + node.w, y: centerY };
}

function renderMatrix(container, matrix, rowLabels, colLabels, options = {}) {
  container.innerHTML = "";

  const wrapper = document.createElement("div");
  wrapper.className = "tensor-matrix";

  matrix.forEach((row, rowIndex) => {
    const rowEl = document.createElement("div");
    rowEl.className = `tensor-row ${options.layout || "q-two"}`;

    const label = document.createElement("div");
    label.className = "tensor-row-label";

    const isActive = (options.activeRows || []).includes(rowIndex);
    const isStream = (options.streamRows || []).includes(rowIndex);
    const isOutput = (options.outputRows || []).includes(rowIndex);
    const isFocus = options.focusRow === rowIndex;
    const shouldDim =
      ((options.activeRows || []).length > 0 || (options.outputRows || []).length > 0) &&
      !isActive &&
      !isOutput &&
      !isFocus;

    label.classList.toggle("active-row", isActive);
    label.classList.toggle("active-stream", isStream);
    label.classList.toggle("active-output", isOutput);
    label.classList.toggle("focus-cell", isFocus);
    label.classList.toggle("dimmed", shouldDim);
    if (options.ghosted) {
      label.classList.add("ghosted");
    }
    label.textContent = rowLabels[rowIndex];
    rowEl.appendChild(label);

    row.forEach((value) => {
      const cell = document.createElement("div");
      cell.className = "tensor-cell";
      cell.classList.toggle("active-row", isActive);
      cell.classList.toggle("active-stream", isStream);
      cell.classList.toggle("active-output", isOutput);
      cell.classList.toggle("focus-cell", isFocus);
      cell.classList.toggle("dimmed", shouldDim);
      if (options.ghosted) {
        cell.classList.add("ghosted");
      }
      cell.textContent = value;
      rowEl.appendChild(cell);
    });

    wrapper.appendChild(rowEl);
  });

  container.appendChild(wrapper);
}

function renderMatrixMessage(container, text) {
  container.innerHTML = "";
  const block = document.createElement("p");
  block.className = "matrix-note";
  block.textContent = text;
  container.appendChild(block);
}

function formatBatchTensor(name, rows) {
  const body = rows.map((row, index) => `[${index}] ${JSON.stringify(row)}`).join("\n");
  return `${name}\n${body}`;
}

function sliceRows(matrix, rows) {
  return rows.map((rowIndex) => matrix[rowIndex]);
}

function placeholderOutput(rows, cols) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => "..."));
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function softmaxRow(row) {
  const max = Math.max(...row);
  const exps = row.map((value) => Math.exp(value - max));
  const denom = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / denom);
}

function weightedSum(weights, vectors) {
  const out = new Array(vectors[0].length).fill(0);
  weights.forEach((weight, rowIndex) => {
    vectors[rowIndex].forEach((value, colIndex) => {
      out[colIndex] += weight * value;
    });
  });
  return out;
}

function formatNumber(value, digits = 4) {
  return Number(value).toFixed(digits);
}

function buildFlashDerived() {
  const score = FLASH_DATA.q.map((qRow) => FLASH_DATA.k.map((kRow) => dot(qRow, kRow)));
  const probs = score.map((row) => softmaxRow(row));
  const outputs = probs.map((row) => weightedSum(row, FLASH_DATA.v));

  return {
    score,
    probs,
    outputs,
    scoreDisplay: score.map((row) => row.map((value) => formatNumber(value, 0))),
    probDisplay: probs.map((row) => row.map((value) => formatNumber(value, 4))),
    outputDisplay: outputs.map((row) => row.map((value) => formatNumber(value, 4))),
    scoreTiles: [
      [
        [formatNumber(score[0][0], 0), formatNumber(score[0][1], 0)],
        [formatNumber(score[1][0], 0), formatNumber(score[1][1], 0)]
      ],
      [
        [formatNumber(score[0][2], 0), formatNumber(score[0][3], 0)],
        [formatNumber(score[1][2], 0), formatNumber(score[1][3], 0)]
      ]
    ]
  };
}

function createSvg(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => {
    el.setAttribute(key, value);
  });
  return el;
}

init();
