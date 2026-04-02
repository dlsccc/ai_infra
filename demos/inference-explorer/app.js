const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  modelKey: "resnet",
  stepIndex: 0,
  timer: null,
  nodes: {},
  edges: {}
};

const ui = {
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
  exProbs: document.getElementById("exProbs")
};

const BERT_EXAMPLE = {
  batchSize: 3,
  texts: [
    "这家餐厅上菜很慢，但是味道不错。",
    "物流速度很快，包装也很完整。",
    "剧情一般，演员表演中规中矩。"
  ],
  tokens: [
    ["[CLS]", "这", "家", "餐", "厅", "上", "菜", "很", "慢", "，", "但", "是", "[SEP]"],
    ["[CLS]", "物流", "速度", "很", "快", "，", "包装", "也", "很", "完整", "。", "[SEP]", "[PAD]"],
    ["[CLS]", "剧情", "一般", "，", "演员", "表演", "中规", "中矩", "。", "[SEP]", "[PAD]", "[PAD]", "[PAD]"]
  ],
  inputIds: [
    [101, 6821, 2157, 762, 1324, 677, 5784, 2523, 2714, 8024, 852, 3221, 102],
    [101, 3300, 6843, 686, 2523, 2571, 8024, 5392, 738, 2523, 5310, 511, 102],
    [101, 3819, 3125, 671, 8024, 782, 4801, 704, 5775, 511, 102, 0, 0]
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
  hiddenShape: "last_hidden_state 形状: [3, 13, 768]",
  poolerShape: "pooler_output 形状: [3, 768]",
  hiddenSample: [
    "sample[0] last_hidden_state[0,0,0:8] = [-0.18, 0.42, 0.11, -0.57, 0.03, 0.75, -0.26, 0.08]",
    "sample[1] last_hidden_state[1,0,0:8] = [-0.05, 0.38, -0.12, -0.09, 0.44, 0.52, -0.31, 0.19]",
    "sample[2] last_hidden_state[2,0,0:8] = [0.14, -0.08, 0.27, -0.22, 0.07, 0.16, -0.03, 0.41]"
  ],
  clsCalc: "H_cls = last_hidden_state[:, 0, :]  # 形状 [batch_size, 768]\nlogits = H_cls @ W + b          # 形状 [batch_size, num_labels]",
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

const MODELS = {
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
        hint: "前两个节点高亮，表示特征提取刚开始。",
        nodes: ["input", "stem"],
        edges: ["e1"]
      },
      {
        title: "2) Conv2_x 残差学习",
        desc: "浅层 Bottleneck 负责学习基础语义，shortcut 帮助梯度和信息稳定传递。",
        io: "56x56x64 -> 56x56x256",
        op: "1x1, 3x3, 1x1 + Identity Add",
        hint: "虚线代表 shortcut 分支，与主分支在残差加法处融合。",
        nodes: ["conv2"],
        edges: ["e2", "s1"]
      },
      {
        title: "3) Conv3_x 语义抽象",
        desc: "随着下采样进行，感受野变大，模型由纹理识别逐步过渡到部件级语义。",
        io: "56x56x256 -> 28x28x512",
        op: "Stride Downsample + Residual Blocks",
        hint: "主干向前推进，同时 shortcut 保持信息流顺畅。",
        nodes: ["conv3"],
        edges: ["e3", "s2"]
      },
      {
        title: "4) Conv4_x 深层特征",
        desc: "这一阶段计算量较高，用于聚合更复杂的高层语义组合。",
        io: "28x28x512 -> 14x14x1024",
        op: "Bottleneck x6",
        hint: "高亮路径更长，表示特征持续深度精炼。",
        nodes: ["conv4"],
        edges: ["e4", "s3"]
      },
      {
        title: "5) Conv5_x 到池化",
        desc: "最终卷积块输出高层语义图，再经全局平均池化压缩为分类向量。",
        io: "14x14x1024 -> 1x1x2048",
        op: "Residual Blocks + Global Average Pooling",
        hint: "末端节点高亮，表示进入分类前的收敛阶段。",
        nodes: ["conv5", "gap"],
        edges: ["e5", "e6"]
      },
      {
        title: "6) 全连接输出类别",
        desc: "池化向量送入全连接层，输出 logits 并通过 softmax 得到分类概率。",
        io: "2048 -> 1000",
        op: "Linear + Softmax",
        hint: "最后一跳高亮，表示一次前向推理完成。",
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
      { id: "attn1", label: "第1层\n多头注意力", x: 402, y: 156, w: 142, h: 62, type: "transformer" },
      { id: "ffn1", label: "第1层\nFFN", x: 596, y: 156, w: 116, h: 62, type: "transformer" },
      { id: "norm1", label: "第1层\nAdd+Norm", x: 764, y: 156, w: 132, h: 62, type: "transformer" },
      { id: "attn2", label: "第2层\n多头注意力", x: 402, y: 344, w: 142, h: 62, type: "transformer" },
      { id: "ffn2", label: "第2层\nFFN", x: 596, y: 344, w: 116, h: 62, type: "transformer" },
      { id: "norm2", label: "第2层\nAdd+Norm", x: 764, y: 344, w: 132, h: 62, type: "transformer" },
      { id: "head", label: "任务头\n[CLS]", x: 964, y: 250, w: 128, h: 60, type: "main" },
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
        desc: "原始文本被切分为 token id，并按任务需求加入 [CLS]、[SEP] 等特殊标记。",
        io: "文本 -> Token IDs",
        op: "Tokenizer",
        hint: "起始节点高亮，表示离散符号输入模型。",
        nodes: ["tokens"],
        edges: []
      },
      {
        title: "2) 构造 Embedding",
        desc: "词向量与位置向量相加，形成可进入 Transformer 的连续表示。",
        io: "ids -> X0 (seq_len x hidden)",
        op: "Lookup + Add",
        hint: "从 token 到 embedding 的流向被点亮。",
        nodes: ["tokens", "embed"],
        edges: ["e1"]
      },
      {
        title: "3) 第1层自注意力",
        desc: "多头注意力为每个 token 建立上下文依赖关系，捕获重要词间关联。",
        io: "X0 -> X1_attn",
        op: "QK^T / sqrt(d), Softmax, Weighted Sum",
        hint: "热力图颜色越深，表示注意力权重越高。",
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
        title: "4) 第1层 FFN 与归一化",
        desc: "FFN 对每个 token 做非线性映射，再通过残差连接和 LayerNorm 稳定特征。",
        io: "X1_attn -> X1",
        op: "Linear-GELU-Linear + Add&Norm",
        hint: "虚线边表示残差 shortcut。",
        nodes: ["ffn1", "norm1"],
        edges: ["e3", "e4", "s1"]
      },
      {
        title: "5) 第2层语义深化",
        desc: "第二层重复注意力与 FFN 过程，进一步强化句子级语义表达。",
        io: "X1 -> X2_attn -> X2",
        op: "MHA + FFN + Add&Norm",
        hint: "跨层路径高亮，展示编码器堆叠计算过程。",
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
        desc: "最终取 [CLS] 表示输入任务头，输出 logits 作为下游任务结果。",
        io: "X2[CLS] -> logits",
        op: "Pooling + Linear",
        hint: "终点边高亮，表示一次推理完成。",
        nodes: ["head", "output"],
        edges: ["e8", "e9"]
      }
    ]
  }
};

function init() {
  bindEvents();
  renderBertExample();
  renderModel();
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
      renderModel();
    });
  });

  ui.prevStep.addEventListener("click", () => {
    shiftStep(-1);
  });

  ui.nextStep.addEventListener("click", () => {
    shiftStep(1);
  });

  ui.playPause.addEventListener("click", () => {
    if (state.timer) {
      stopAutoplay();
    } else {
      startAutoplay();
    }
  });
}

function renderModel() {
  const model = MODELS[state.modelKey];
  ui.graphTitle.textContent = model.title;
  ui.graph.setAttribute("viewBox", model.viewBox);

  drawGraph(model);
  renderStep();
  toggleBertExample();
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

function renderStep() {
  const model = MODELS[state.modelKey];
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

function renderAttention(matrix) {
  ui.attentionGrid.innerHTML = "";

  if (state.modelKey !== "bert") {
    ui.attentionGrid.classList.add("empty");
    ui.attentionGrid.textContent = "切换到 BERT 并进入注意力步骤后可查看热力图。";
    return;
  }

  if (!matrix) {
    ui.attentionGrid.classList.add("empty");
    ui.attentionGrid.textContent = "当前步骤不展示注意力矩阵。";
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

function formatBatchTensor(name, rows) {
  const body = rows.map((row, index) => `[${index}] ${JSON.stringify(row)}`).join("\n");
  return `${name}\n${body}`;
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

function toggleBertExample() {
  const isBert = state.modelKey === "bert";
  ui.bertExamplePanel.classList.toggle("hidden", !isBert);
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
  const steps = MODELS[state.modelKey].steps;
  state.stepIndex = (state.stepIndex + delta + steps.length) % steps.length;
  renderStep();
}

function startAutoplay() {
  stopAutoplay();
  ui.playPause.textContent = "暂停播放";
  state.timer = window.setInterval(() => {
    shiftStep(1);
  }, 2200);
}

function stopAutoplay() {
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
  ui.playPause.textContent = "自动播放";
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

function createSvg(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => {
    el.setAttribute(key, value);
  });
  return el;
}

init();
