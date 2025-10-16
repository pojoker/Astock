# 在 `Factor24_Pretrained_ROBERT.ipynb` 中使用自定义数据

本文档说明如何将新的语料接入 `code/Factor24_Pretrained_ROBERT.ipynb` Notebook，并总结数据文件必须满足的格式要求。

## 1. 文件组织位置
- Notebook 位于 `code/` 目录下，默认按 `../data/pre/` 相对路径读取 `train.csv`、`val.csv`、`test.csv` 与 `ood.csv` 四个文件。若要替换为自定义数据，建议：
  1. 在 `data/` 目录下新建子目录（例如 `data/custom/`）。
  2. 将预处理好的 `train.csv`、`val.csv`、`test.csv` 与 `ood.csv` 复制到该目录中。
  3. 在 Notebook 顶部新增或修改一个路径变量：
     ```python
     from pathlib import Path

     DATA_DIR = Path('../data/custom')  # 指向新的数据目录
     df_train = pd.read_csv(DATA_DIR / 'train.csv', sep='\t')
     df_val = pd.read_csv(DATA_DIR / 'val.csv', sep='\t')
     df_test = pd.read_csv(DATA_DIR / 'test.csv', sep='\t')
     df_ood = pd.read_csv(DATA_DIR / 'ood.csv', sep='\t')
     ```
  4. Notebook 其余与 `df_train`、`df_val`、`df_test`、`df_ood` 相关的单元均可复用，无需额外调整。

## 2. 必须包含的字段
`Factor24_Pretrained_ROBERT.ipynb` 在预处理和建模阶段会直接访问下列列名，请确保四个数据文件均提供这些字段：

| 列名 | 内容说明 | 备注 |
| --- | --- | --- |
| `text_a` | 已分好句的新闻正文（字符串）。 | 作为 BERT 文本输入。 |
| `label` | 情感类别标签，整数 `0`/`1`/`2` 分别代表 `negative`、`neutral`、`positive`。 | 训练、验证、测试与 OOD 文件均需提供。 |
| `verb`、`A0`、`A1` | 语义角色标注的跨度信息，格式为字符串化的列表，例如 `'[(4, 2), (31, 2)]'`。 | Notebook 会调用 `string_to_tuples_list` 将其恢复为 `List[Tuple[int, int]]`。 |
| `verbA0A1` | 将动词与对应 A0/A1 角色对齐后的嵌套列表，格式需能被 `ast.literal_eval` 正确解析。 | `mask()` 函数依赖该字段生成掩码。 |
| `stock_factors` | 299 维的量化因子向量，使用字符串化列表存储，例如 `'[0.12, -0.08, ...]'`。 | 读取后会被转换为 `float` 数组。 |
| `DATE` | 时间戳，支持 `YYYY-MM-DD` 或 `YYYY-MM-DD HH:MM:SS`。 | Notebook 中的 OOD 评估会根据该列筛选时间窗口。 |

除上述核心字段外，其余列（如 `CODE`、`TITLE` 等）可保留以便后续分析，但 Notebook 不会直接使用。

### SRL 角色含义
- `verb`：谓词触发词或中心动词，用于描述一条新闻中的关键动作或事件。模型会依据这些片段来聚合上下文。
- `A0`：PropBank 标注体系下的施事（Agent）角色，通常对应执行动作的主体，例如发布公告的公司、做出决策的机构等。
- `A1`：PropBank 中的受事（Patient/Theme）角色，常指动作的客体、受到影响的对象，或动词描述的结果与内容。

在 `verbA0A1` 列中，三类角色会按照 `[verb_spans, A0_spans, A1_spans]` 的嵌套顺序存储，Notebook 会利用这些掩码把不同角色的文字片段提取出来，进一步与股票量化因子拼接用于情感分类。

## 3. 预处理要求
- **缺失处理**：Notebook 会丢弃 `verbA0A1` 为空 (`NaN`) 或字面值为 `'[]'` 的样本；请在准备数据时确认语义角色标注产出完整。
- **类型转换**：
  - `verb`/`A0`/`A1` 字段的字符串必须能够解析为二维列表，元素为 `(起始索引, 长度)`。
  - `verbA0A1` 字段需满足三层嵌套结构 `[[verb_spans], [A0_spans], [A1_spans]]`，并与 `mask()` 函数中对 `row['verbA0A1'][j][k]` 的索引方式保持一致。
  - `stock_factors` 应包含 299 个可转换为浮点数的值，以便在 `mask()` 中调用 `map(float, ...)`。
- **时间字段**：若要在 Notebook 中继续使用 OOD 切片分析，请确保 `DATE` 列包含可比较的时间戳；否则需要相应调整筛选条件。

## 4. 常见调试提示
- 如果执行 `mask(df)` 时出现索引错误，通常意味着 `verbA0A1` 的嵌套层级或跨度值缺失，请检查该列的格式是否与示例一致。
- 当 `create_data_loader()` 报错时，请确认 `df.text_a`、`df.label`、`df.stock_factors` 等列存在，并且未被提前删除或重命名。
- 若仅替换推理阶段的数据，可在训练完成后单独读取新的 CSV，套用与 Notebook 中 `df_ood` 相同的预处理、掩码与 `create_data_loader()` 流程，然后调用 `get_predictions()`。

按照以上要求准备并引用新的数据文件，即可在不修改模型结构的前提下复用 Notebook 的完整训练与评估流程。
