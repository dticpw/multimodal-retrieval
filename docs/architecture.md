# 多模态 RAG 系统架构与运行逻辑

本文档以一次真实查询 `"a dog playing on grass"` 为例，完整追踪从用户输入到终端返回结果的全部流程。

---

## 一、系统整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                      离线阶段 (Ingest)                        │
│                                                              │
│  Flickr30K CSV ──→ 解析图文对 ──→ CLIP 编码 ──→ 存储          │
│                                      │                       │
│                          ┌───────────┼───────────┐           │
│                          ▼           ▼           ▼           │
│                    image.index  text.index  metadata.db      │
│                     (FAISS)     (FAISS)     (SQLite)         │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                      在线阶段 (Serve)                         │
│                                                              │
│  用户 query ──→ FastAPI ──→ Retriever 编排                    │
│                                │                             │
│                    ┌───────────┴───────────┐                 │
│                    ▼                       ▼                 │
│              CLIP 编码 query          LLM 生成回答            │
│                    │                       ▲                 │
│                    ▼                       │                 │
│             FAISS 近邻搜索 ──→ SQLite 查元数据 ──→ 组装上下文   │
└──────────────────────────────────────────────────────────────┘
```

系统分为两个阶段：
- **离线阶段**：将数据集的图片和文本用 CLIP 编码为向量，存入 FAISS 索引和 SQLite 数据库
- **在线阶段**：接收用户查询，实时编码、检索、生成回答

---

## 二、离线阶段：数据摄入 (`scripts/ingest.py`)

运行命令：`python -m scripts.ingest`（4K 子集）或 `python -m scripts.ingest --full`（全量）

### 步骤详解

```
train4K.csv (800 图, ~4000 条 caption)
       │
       ▼
┌─────────────────────────┐
│ Step 1: 解析 CSV         │
│ parse_flickr_csv()       │
│                         │
│ CSV 格式 (竖线分隔):     │
│ filename | comment_num | caption
│                         │
│ 输出: {filename: [(comment_num, caption), ...]}
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ Step 2: 验证图片存在      │
│                         │
│ 逐个检查 flickr30k-images/ 下文件是否存在
│ 输出: valid_images = [(filename, full_path), ...]
└─────────┬───────────────┘
          │
          ├──────────────────────────────┐
          ▼                              ▼
┌───────────────────────┐   ┌───────────────────────┐
│ Step 3: 编码图片        │   │ Step 4: 编码文本        │
│ CLIPEncoder            │   │ CLIPEncoder            │
│   .encode_images()     │   │   .encode_texts()      │
│                       │   │                       │
│ 每张图片:              │   │ 每条 caption:           │
│ JPG → PIL.Image       │   │ text → tokenize        │
│ → CLIPProcessor       │   │ → CLIPProcessor        │
│   (resize 224×224,    │   │   (padding, truncation, │
│    normalize)         │   │    max_length=77)       │
│ → vision_model        │   │ → text_model            │
│ → visual_projection   │   │ → text_projection       │
│ → 512 维向量           │   │ → 512 维向量             │
│ → L2 归一化            │   │ → L2 归一化              │
│                       │   │                       │
│ 输出: (800, 512)      │   │ 输出: (~4000, 512)     │
│       float32         │   │       float32          │
└─────────┬─────────────┘   └─────────┬─────────────┘
          │                            │
          ▼                            ▼
┌───────────────────────┐   ┌───────────────────────┐
│ Step 5a: 构建图像索引   │   │ Step 5b: 构建文本索引   │
│ FAISSIndexer           │   │ FAISSIndexer           │
│   .build_image_index() │   │   .build_text_index()  │
│                       │   │                       │
│ IndexFlatIP(512)      │   │ IndexFlatIP(512)      │
│ = 内积索引              │   │ = 内积索引              │
│ (L2归一化后 = 余弦相似度)│   │ (L2归一化后 = 余弦相似度)│
│                       │   │                       │
│ 保存: indexes/        │   │ 保存: indexes/         │
│       image.index     │   │       text.index       │
└───────────────────────┘   └───────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Step 6: 存储元数据                │
│ MetadataStore (SQLite)           │
│                                  │
│ images 表:                       │
│ ┌─────┬───────────┬──────────┐   │
│ │ idx │ image_id  │ filename │   │
│ │ 0   │ 100009... │ 100009.. │   │
│ │ 1   │ 100159... │ 100159.. │   │
│ └─────┴───────────┴──────────┘   │
│                                  │
│ captions 表:                     │
│ ┌────────────┬───────────┬─────┐ │
│ │ caption_idx│ image_idx │ ... │ │
│ │ 0          │ 0         │ ... │ │
│ │ 1          │ 0         │ ... │ │
│ └────────────┴───────────┴─────┘ │
│                                  │
│ 关键: idx = FAISS 中的向量位置     │
│ 通过 idx 可以从 FAISS 结果        │
│ 反查到图片文件名和 caption 文本    │
│                                  │
│ 保存: data/metadata.db           │
└──────────────────────────────────┘
```

### 核心设计：FAISS idx 与 SQLite idx 的对齐

这是整个系统的关键。摄入时，图片按排序顺序编号：
- 第 0 张图 → FAISS image_index 的第 0 个向量 → SQLite images 表 idx=0
- 第 1 张图 → FAISS image_index 的第 1 个向量 → SQLite images 表 idx=1

caption 同理。这样 FAISS 搜索返回的 index 可以直接在 SQLite 中查到对应元数据。

---

## 三、在线阶段：RAG 查询全链路

### 以 `"a dog playing on grass"` 为例

```
用户终端
│
│  POST http://localhost:8000/api/v1/rag/query
│  Body: {"query": "a dog playing on grass", "top_k": 5}
│
▼
```

### Step 1: FastAPI 路由接收请求

**文件**: `app/api/routes.py:37-43`

```python
@router.post("/rag/query", response_model=RAGResponse)
def rag_query(
    query: RAGQuery,                                    # Pydantic 自动解析 JSON
    retriever: Retriever = Depends(get_retriever),      # 依赖注入 (单例)
    generator: LLMGenerator = Depends(get_generator),   # 依赖注入 (单例)
):
    return retriever.rag_query(query.query, query.top_k, generator)
```

**发生了什么**：
1. FastAPI 将 JSON body 解析为 `RAGQuery(query="a dog playing on grass", top_k=5)`
2. 通过 `Depends()` 注入单例的 `Retriever` 和 `LLMGenerator`
3. 首次请求时会触发 CLIP 模型加载和 FAISS 索引加载（后续请求复用）



> **FastAPI**
>
> * 是一个 Python Web 框架，作用是把你的 Python 函数变成可以通过 HTTP 请求调用的 API 服务。
> * 没有 FastAPI 时，你的检索代码只能在 Python 脚本里直接调用。有了 FastAPI，外部任何程序（浏览器、curl、前端页面、手机 App）都能通过 HTTP 调用你的代码。
>
> 在本项目中，FastAPI 只做三件事：
>
> 1. 接收请求并解析参数：用户发来的 {"query": "a dog playing on grass", "top_k": 5} 被自动变成 RAGQuery(query="a dog playing on grass", top_k=5)。
> 2. 依赖注入（管理单例服务）
> 3. 把 Python 对象序列化为 JSON 返回 `return RAGResponse(answer="...", sources=[...], ...)`
>
> 





### Step 2: Retriever 编排 RAG 流程

**文件**: `app/services/retriever.py:145-164`

```python
def rag_query(self, query: str, top_k: int, generator: LLMGenerator) -> RAGResponse:
    # 阶段 A: 检索
    retrieval_resp = self.text_to_image(query, top_k)   # ← 见 Step 3-5 # 调 encoder → indexer → metadata
    sources = retrieval_resp.results					# 拿到 5 个检索结果

    # 阶段 B: 生成
    answer = generator.generate(query, sources)          # ← 见 Step 6-7 # 调 LLM API

    return RAGResponse(answer=answer, sources=sources, ...)
```

**两个阶段**：先 Retrieval（检索），后 Generation（生成）。这就是 RAG 的 R 和 G。

* R (Retrieval)：调 text_to_image()，内部串联 CLIP 编码 → FAISS 搜索 → SQLite 查元数据，拿到 5 个最相关的图片及其 captions
* G (Generation)：把检索结果 + 用户问题交给 LLM 生成回答







### Step 3: CLIP 编码用户 query

> 这是检索的第一步——把用户的文本 "a dog playing on grass" 变成一个 512 维向量。

**文件**: `app/services/encoder.py:42-70`

```
"a dog playing on grass"
        │
        ▼
┌─────────────────────────────────────────────────┐
│ CLIPEncoder.encode_texts()                       │
│                                                  │
│ 1. CLIPProcessor tokenize:                       │
│    "a dog playing on grass"                      │
│    → input_ids: [49406, 320, 1929, 1823, ...]    │
│    → attention_mask: [1, 1, 1, 1, ...]           │
│    → padding 到 max_length=77                     │
│                                                  │
│ 2. text_model (CLIP Transformer Encoder):        │
│    input_ids → 12 层 Transformer                  │
│    → pooler_output: (1, 512)                     │
│    这是 [EOS] token 的隐藏状态                     │
│                                                  │
│ 3. text_projection (线性层 512→512):              │
│    将 Transformer 输出投影到                       │
│    图文共享的 embedding 空间                       │
│    → embeddings: (1, 512)                        │
│                                                  │
│ 4. L2 归一化:                                     │
│    embeddings / ||embeddings||₂                   │
│    归一化后内积 = 余弦相似度                        │
│    → query_emb: (1, 512), float32, L2-normalized │
└─────────────────────────────────────────────────┘
```

**关键点**：CLIP 的设计使得文本向量和图像向量处于同一个 512 维空间。语义相近的图文对，其向量的余弦相似度更高。







### Step 4: FAISS 近邻搜索

> 现在用 query 向量去 FAISS 索引里找最相似的图片向量。

**文件**: `app/services/indexer.py:43-51`

```
query_emb (1, 512)
        │
        ▼
┌─────────────────────────────────────────────────┐
│ FAISSIndexer.search_images(query_emb, top_k=5)   │
│                                                  │
│ image_index 中有 800 个图像向量 (4K 子集)          │
│                                                  │
│ IndexFlatIP.search():                            │
│   计算 query_emb 与所有 800 个向量的内积            │
│   (向量已 L2 归一化，内积 = 余弦相似度)              │
│                                                  │
│   query · v₀ = 0.312                             │
│   query · v₁ = 0.187                             │
│   query · v₂ = 0.429  ← 高相似度                  │
│   ...                                            │
│   query · v₇₉₉ = 0.201                           │
│                                                  │
│   排序取 top_k=5:                                 │
│   scores:  [0.429, 0.415, 0.398, 0.391, 0.385]  │
│   indices: [2,     156,   423,   87,    601]     │
│                                                  │
│ 输出:                                             │
│   scores: shape (1, 5) — 相似度分数                │
│   indices: shape (1, 5) — FAISS 中的向量位置       │
└─────────────────────────────────────────────────┘
```

**注意**：这里返回的 indices 是 FAISS 索引中的位置编号，不是图片 ID。需要通过 SQLite 反查。

> 当前配置下FAISS几乎无法起到加速作用。用的是 IndexFlatIP——这是 FAISS 最基础的索引，本质就是暴力遍历。
>
> FAISS 真正的价值在于规模扩大之后
>
> | 数据规模   | IndexFlatIP（暴力） | FAISS近似索引 | 加速比 |
> | ---------- | ------------------- | ------------- | ------ |
> | 800 (当前) | ~1ms                | 不需要        | 1x     |
> | 10 万      | ~50ms               | ~1ms (IVF)    | 50x    |
> | 100 万     | ~500ms              | ~2ms (IVF+PQ) | 250x   |
> | 1 亿       | ~50s                | ~5ms (HNSW)   | 10000x |
>
> 
>
> 实际上向量数据库的底层就是FAISS和HNSW等东西。





### Step 5: SQLite 元数据查询

> FAISS 只返回了位置编号，我们还不知道这些是哪些图片。需要通过 SQLite 反查。

**文件**: `app/services/metadata.py:114-146` + `app/services/retriever.py:32-69`

```
indices: [2, 156, 423, 87, 601]
        │
        ▼
┌────────────────────────────────────────────────────┐
│ MetadataStore.get_images_by_indices([2,156,423,...])│
│                                                    │
│ SQL 查询:                                          │
│ SELECT idx, image_id, filename, filepath           │
│ FROM images WHERE idx IN (2, 156, 423, 87, 601)   │
│                                                    │
│ 结果:                                              │
│ ┌─────┬────────────┬──────────────────┐            │
│ │ idx │ image_id   │ filename         │            │
│ │ 2   │ 1019077836 │ 1019077836.jpg   │            │
│ │ 156 │ 1026685415 │ 1026685415.jpg   │            │
│ │ 423 │ 1164131282 │ 1164131282.jpg   │            │
│ │ 87  │ 1072153132 │ 1072153132.jpg   │            │
│ │ 601 │ 1119015538 │ 1119015538.jpg   │            │
│ └─────┴────────────┴──────────────────┘            │
│                                                    │
│ 对每个图片再查 captions:                             │
│ SELECT caption FROM captions                       │
│ WHERE image_idx = ? ORDER BY caption_number        │
│                                                    │
│ 例如 idx=2 的 captions:                             │
│ - "A brown dog plays with a water hose on grass"   │
│ - "A big brown dog is playing in the grass..."     │
│ - ...                                              │
└────────────────────────────────────────────────────┘
        │
        ▼
组装为 5 个 RetrievalResult 对象:
[
  RetrievalResult(image_id="1019077836", filename="1019077836.jpg",
                  filepath="E:\\PG\\dataset\\...", score=0.429,
                  captions=["A brown dog plays with...", ...]),
  ...
]
```

> 到这里，检索阶段 (R) 结束。拿到了 5 张最相关图片的文件名、路径、相似度分数和文字描述。





### Step 6: 组装 LLM Prompt

**文件**: `app/services/generator.py:22-31`

```
5 个 RetrievalResult
        │
        ▼
┌──────────────────────────────────────────────────┐
│ _format_sources(sources) → context string         │
│                                                  │
│ 拼装为:                                           │
│ ┌──────────────────────────────────────────────┐ │
│ │ [来源 1] 文件: 1019077836.jpg (相似度: 0.429)  │ │
│ │ 描述: A brown dog plays with a water hose    │ │
│ │       on grass; A big brown dog is playing   │ │
│ │       in the grass...                        │ │
│ │                                              │ │
│ │ [来源 2] 文件: 1026685415.jpg (相似度: 0.415)  │ │
│ │ 描述: A black dog walks with a green toy...  │ │
│ │                                              │ │
│ │ ... (共 5 条来源)                              │ │
│ └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│ 最终发给 LLM 的 messages:                         │
│                                                  │
│ system: "你是一个知识库问答助手。根据检索到的       │
│          图文内容回答用户的问题。回答要求：          │
│          1. 仅根据提供的检索结果回答，不要编造信息  │
│          2. 如果检索结果不足以回答问题，明确告知    │
│          3. 引用具体的图片文件名作为依据            │
│          4. 使用中文回答"                          │
│                                                  │
│ user:   "检索到的相关内容：                        │
│          [来源 1] 文件: 1019077836.jpg ...         │
│          [来源 2] ...                             │
│          ...                                     │
│          用户问题：a dog playing on grass"         │
└──────────────────────────────────────────────────┘
```



### Step 7: LLM 生成回答

> 调用 LLM API来根据message回答问题

**文件**: `app/services/generator.py:51-68`

```
拼装好的 messages
        │
        ▼
┌──────────────────────────────────────────────────┐
│ OpenAI SDK → HTTP POST                            │
│                                                  │
│ 请求:                                             │
│   URL: {LLM_BASE_URL}/chat/completions           │
│   model: claude-sonnet-4-20250514 (或你配置的模型)    │
│   messages: [system + user]                      │
│   max_tokens: 1024                               │
│   temperature: 0.7                               │
│                                                  │
│ LLM 根据检索结果生成结构化回答:                     │
│   - 描述每张检索到的图片中狗在草地上玩耍的场景      │
│   - 引用具体文件名 (1019077836.jpg 等)             │
│   - 使用中文                                      │
│                                                  │
│ 返回: answer 字符串                                │
└──────────────────────────────────────────────────┘
```



### Step 8: 组装响应返回

**文件**: `app/services/retriever.py:159-164`

```python
RAGResponse(
    answer="根据检索到的内容，我可以为您提供多个狗在草地上...",  # LLM 生成的回答
    sources=[RetrievalResult(...), ...],   # 5 个检索结果
    retrieval_ms=42.35,                    # 检索耗时
    generation_ms=3521.17,                 # LLM 生成耗时
)
```

FastAPI 自动序列化为 JSON 返回给客户端。

> FastAPI 自动把这个 Pydantic 对象序列化为 JSON，返回 HTTP 200 给你的终端。

---

## 四、核心模型与组件

### 4.1 CLIP (Contrastive Language-Image Pre-training)

```
                    CLIP 模型结构
┌─────────────────────────────────────────────┐
│                                             │
│  文本侧                    图像侧            │
│  ┌─────────┐              ┌─────────┐       │
│  │Tokenizer│              │Resize   │       │
│  │ (77 tok)│              │ 224×224 │       │
│  └────┬────┘              └────┬────┘       │
│       ▼                        ▼            │
│  ┌─────────┐              ┌─────────┐       │
│  │  Text   │              │ Vision  │       │
│  │Transformer              │Transformer     │
│  │ (12层)  │              │ (12层)  │       │
│  └────┬────┘              └────┬────┘       │
│       ▼                        ▼            │
│  pooler_output            pooler_output     │
│  (1, 512)                 (1, 768)          │
│       ▼                        ▼            │
│  ┌──────────┐            ┌───────────┐      │
│  │  text_   │            │ visual_   │      │
│  │projection│            │projection │      │
│  │ (512→512)│            │ (768→512) │      │
│  └────┬─────┘            └─────┬─────┘      │
│       ▼                        ▼            │
│   text_emb (512)         image_emb (512)    │
│       │                        │            │
│       └───────── 共享空间 ──────┘            │
│          余弦相似度可直接计算                  │
└─────────────────────────────────────────────┘
```

**核心思想**：CLIP 在 4 亿图文对上做对比学习，使匹配的图文对向量靠近、不匹配的远离。训练完成后，文本和图像共享同一个 512 维语义空间，可以直接用余弦相似度计算跨模态相关性。

**本项目使用 ViT-B/32**：
- Vision Transformer，patch size 32，输入 224×224
- 文本端最大 77 tokens
- 输出 512 维向量

### 4.2 FAISS (Facebook AI Similarity Search)

```
IndexFlatIP (Inner Product) — 精确搜索
┌──────────────────────────────────┐
│ 存储 N 个 512 维向量               │
│                                  │
│ 搜索时: 暴力计算 query 与每个      │
│ 向量的内积，返回 top_k 最大值      │
│                                  │
│ 复杂度: O(N × d)                  │
│ N=800 (4K子集), d=512             │
│ 搜索延迟: ~1ms                    │
│                                  │
│ 因为向量已 L2 归一化:              │
│ inner_product(a, b) = cos(a, b)  │
│ 所以 IndexFlatIP = 余弦相似度搜索  │
└──────────────────────────────────┘
```

**双索引设计**：
- `image.index`: 每张图一个向量 (800 个)，用于 text→image 和 image→image 检索
- `text.index`: 每条 caption 一个向量 (~4000 个)，用于 image→text 检索

### 4.3 依赖注入 (Singleton)

```
┌─────────────────────────────────────┐
│ dependencies.py                      │
│                                     │
│ @lru_cache(maxsize=1)               │
│ 确保每个服务只实例化一次:              │
│                                     │
│ get_encoder()  → CLIPEncoder (单例)  │
│       首次调用时加载 CLIP 模型到 GPU   │
│                                     │
│ get_indexer()  → FAISSIndexer (单例)  │
│       首次调用时从磁盘加载 FAISS 索引  │
│                                     │
│ get_metadata() → MetadataStore (单例)│
│       首次调用时连接 SQLite            │
│                                     │
│ get_retriever() → Retriever (单例)   │
│       组合 encoder + indexer + metadata│
│                                     │
│ get_generator() → LLMGenerator (单例)│
│       初始化 OpenAI client            │
└─────────────────────────────────────┘
```

---

## 五、数据流完整时序图

```
用户                FastAPI            Retriever         CLIPEncoder        FAISSIndexer      MetadataStore     LLMGenerator        外部 LLM API
 │                    │                   │                  │                  │                 │                 │                    │
 │  POST /rag/query   │                   │                  │                  │                 │                 │                    │
 │ {"query":"a dog    │                   │                  │                  │                 │                 │                    │
 │  playing on grass",│                   │                  │                  │                 │                 │                    │
 │  "top_k": 5}       │                   │                  │                  │                 │                 │                    │
 │───────────────────>│                   │                  │                  │                 │                 │                    │
 │                    │  rag_query()      │                  │                  │                 │                 │                    │
 │                    │──────────────────>│                  │                  │                 │                 │                    │
 │                    │                   │                  │                  │                 │                 │                    │
 │                    │                   │  encode_texts()  │                  │                 │                 │                    │
 │                    │                   │ ["a dog playing  │                  │                 │                 │                    │
 │                    │                   │  on grass"]      │                  │                 │                 │                    │
 │                    │                   │─────────────────>│                  │                 │                 │                    │
 │                    │                   │                  │ tokenize         │                 │                 │                    │
 │                    │                   │                  │ → text_model     │                 │                 │                    │
 │                    │                   │                  │ → text_projection│                 │                 │                    │
 │                    │                   │                  │ → L2 normalize   │                 │                 │                    │
 │                    │                   │  query_emb       │                  │                 │                 │                    │
 │                    │                   │  (1, 512)        │                  │                 │                 │                    │
 │                    │                   │<─────────────────│                  │                 │                 │                    │
 │                    │                   │                  │                  │                 │                 │                    │
 │                    │                   │  search_images(query_emb, 5)       │                 │                 │                    │
 │                    │                   │────────────────────────────────────>│                 │                 │                    │
 │                    │                   │                  │   暴力内积搜索    │                 │                 │                    │
 │                    │                   │                  │   query·v_i      │                 │                 │                    │
 │                    │                   │  scores, indices │   取 top 5       │                 │                 │                    │
 │                    │                   │<────────────────────────────────────│                 │                 │                    │
 │                    │                   │                  │                  │                 │                 │                    │
 │                    │                   │  get_images_by_indices([2,156,...]) │                 │                 │                    │
 │                    │                   │───────────────────────────────────────────────────>│                 │                    │
 │                    │                   │                  │                  │  SQL 查图片信息  │                 │                    │
 │                    │                   │                  │                  │  SQL 查 captions │                 │                    │
 │                    │                   │  5个 RetrievalResult               │                 │                 │                    │
 │                    │                   │<───────────────────────────────────────────────────│                 │                    │
 │                    │                   │                  │                  │                 │                 │                    │
 │                    │                   │  generate(query, sources)          │                 │                 │                    │
 │                    │                   │───────────────────────────────────────────────────────────────────>│                    │
 │                    │                   │                  │                  │                 │ format_sources()│                    │
 │                    │                   │                  │                  │                 │ 拼装 prompt     │                    │
 │                    │                   │                  │                  │                 │                 │  POST /v1/chat/    │
 │                    │                   │                  │                  │                 │                 │  completions       │
 │                    │                   │                  │                  │                 │                 │───────────────────>│
 │                    │                   │                  │                  │                 │                 │                    │
 │                    │                   │                  │                  │                 │                 │  生成中文回答       │
 │                    │                   │                  │                  │                 │                 │<───────────────────│
 │                    │                   │  answer string   │                  │                 │                 │                    │
 │                    │                   │<───────────────────────────────────────────────────────────────────│                    │
 │                    │                   │                  │                  │                 │                 │                    │
 │                    │  RAGResponse      │                  │                  │                 │                 │                    │
 │                    │  {answer, sources, │                  │                  │                 │                 │                    │
 │                    │   retrieval_ms,   │                  │                  │                 │                 │                    │
 │                    │   generation_ms}  │                  │                  │                 │                 │                    │
 │                    │<──────────────────│                  │                  │                 │                 │                    │
 │                    │                   │                  │                  │                 │                 │                    │
 │  200 OK            │                   │                  │                  │                 │                 │                    │
 │  JSON response     │                   │                  │                  │                 │                 │                    │
 │<───────────────────│                   │                  │                  │                 │                 │                    │
```

---

## 六、性能特征

| 阶段 | 耗时量级 | 瓶颈 |
|------|---------|------|
| CLIP 编码 query | ~5-10ms | GPU 推理 (首次会触发模型加载 ~3s) |
| FAISS 搜索 800 个向量 | ~1ms | CPU 内积计算 (暴力搜索) |
| SQLite 查询 5 条记录 | ~1ms | 磁盘 I/O (已有索引) |
| LLM 生成回答 | ~2000-5000ms | 网络延迟 + LLM 推理 (主要瓶颈) |
| **端到端** | **~2-5s** | **LLM 生成占 95%+ 时间** |

---

## 七、文件与职责对照

| 文件 | 职责 | 在 RAG 查询中的角色 |
|------|------|-------------------|
| `app/main.py` | FastAPI 应用入口 | 启动 uvicorn 服务器 |
| `app/api/routes.py` | 路由定义 | 接收请求，注入依赖，调用 retriever |
| `app/api/dependencies.py` | 依赖注入 | lru_cache 单例管理所有服务 |
| `app/core/config.py` | 配置管理 | Pydantic Settings，读取 .env |
| `app/models/schemas.py` | 数据模型 | frozen Pydantic 模型，请求/响应格式 |
| `app/services/encoder.py` | CLIP 编码 | 将 query 文本编码为 512 维向量 |
| `app/services/indexer.py` | FAISS 索引 | 向量近邻搜索，返回 top_k indices |
| `app/services/metadata.py` | SQLite 存储 | 根据 indices 查图片信息和 captions |
| `app/services/retriever.py` | 编排服务 | 串联 encoder→indexer→metadata→generator |
| `app/services/generator.py` | LLM 生成 | 拼装 prompt，调用外部 LLM API |
| `scripts/ingest.py` | 数据摄入 | 离线构建 FAISS 索引和 SQLite 数据库 |
| `app/demo.py` | Gradio Demo | 浏览器可视化界面，直接调用服务层 |

---

## 八、Gradio Demo 界面 (`app/demo.py`)

> P3 新增。提供浏览器可视化界面，面试时 `python -m app.demo` 一键演示。

### 架构差异：Gradio vs FastAPI

```
FastAPI 模式 (API 服务):
  HTTP 请求 → FastAPI 路由 → Depends() 注入 → 服务层 → JSON 响应

Gradio 模式 (Demo 界面):
  浏览器操作 → Gradio 回调 → 直接调用服务层单例 → 浏览器渲染
```

Gradio Demo **不经过 FastAPI HTTP 层**，直接复用 `dependencies.py` 的 `@lru_cache` 单例：

```python
from app.api.dependencies import get_retriever, get_indexer, get_metadata, get_generator

retriever = get_retriever()   # 复用同一个单例工厂
indexer = get_indexer()
metadata = get_metadata()
generator = get_generator()
```

好处：无网络开销、无序列化开销、共享模型实例不重复加载。

### 四个功能 Tab

```
┌─────────────────────────────────────────────────────────────┐
│  Gradio Blocks (http://localhost:7860)                       │
│                                                             │
│  ┌──────────┬──────────┬──────────┬──────────┐              │
│  │ 文本搜图  │ 以图搜图  │ RAG 问答  │ 系统状态  │              │
│  └──────────┴──────────┴──────────┴──────────┘              │
│                                                             │
│  Tab 1 - 文本搜图:                                           │
│    输入: 文本框 + Top K 滑块                                  │
│    处理: retriever.text_to_image(query, top_k)              │
│    输出: Gallery 图片墙 + Markdown 详情                       │
│                                                             │
│  Tab 2 - 以图搜图:                                           │
│    输入: 图片上传 + Top K 滑块                                │
│    处理: retriever.image_to_image(path, top_k)              │
│    输出: Gallery 图片墙 + Markdown 详情                       │
│                                                             │
│  Tab 3 - RAG 问答:                                           │
│    输入: 文本框 + 检索数量滑块                                 │
│    处理: retriever.rag_query(query, top_k, generator)       │
│    输出: LLM 回答文本 + 来源图片 Gallery + 耗时统计            │
│                                                             │
│  Tab 4 - 系统状态:                                           │
│    处理: metadata.count_*() + indexer.*_index_size           │
│    输出: 索引大小、图片数、caption 数、设备信息等               │
└─────────────────────────────────────────────────────────────┘
```

### 关键配置

Gradio 6 要求在 `launch()` 中传 `allowed_paths` 以授权访问本地图片目录：

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft(),
    allowed_paths=[settings.flickr30k_image_dir],  # 必须，否则图片无法显示
)
```

### 启动方式

```bash
conda activate th123
cd E:/PG/multimodal-retrieval
python -m app.demo
# 浏览器打开 http://localhost:7860
```

---

## 九、项目展示页 (`showcase/`)

纯静态 HTML + CSS 单页面，部署到 Cloudflare Pages (`retrieval.koa-ol.com`)。

```
showcase/
├── index.html      # 项目介绍、架构图、技术栈、benchmark 表格
├── style.css       # 深色主题响应式样式
└── images/         # Demo 截图 (手动添加)
```

不含任何后端逻辑，仅展示项目信息和 Demo 截图，用于简历链接。
