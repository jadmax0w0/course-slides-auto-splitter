from abc import ABC

from ollama import chat, ChatResponse
from paddleocr import PaddleOCR


class PageSimilarityBase(ABC):
    def check_sim(self, theme_desc: str, text1: str, images1: list[str], text2: str, images2: list[str], **kwargs) -> bool:
        raise NotImplementedError()


class LLMPageSimilarity(PageSimilarityBase):
    def __init__(self):
        super().__init__()

        self.ocr_model = None

        self.sys_prompt = """# Role
你是一名资深的教育内容结构分析专家。你的任务是分析给定 PDF 文档中的两页 Slide（幻灯片）的文本内容，判断它们是否属于同一个**微观子主题（Sub-topic）**或**具体的知识点单元**。

# Context
你将获得以下信息：
1. **Document Main Topic (文档大主题)**：整个 PDF 文档的核心议题。
2. **Slide A Text**：前一页的文本内容。
3. **Slide B Text**：后一页的文本内容。

# Analysis Criteria (判断标准)
请基于以下逻辑进行分析，**分辨粒度必须精细**：

**判定为属于同一小主题 (Conclusion: 1) 的情况：**
* **内容延续**：Slide B 是 Slide A 未完成句子的延续，或 Slide B 继续列举 Slide A 中未讲完的列表（List）。
* **概念深入**：Slide B 对 Slide A 中提出的具体概念进行定义解释、举例说明或深入推导，且没有引入新的核心概念。
* **视觉/逻辑连贯**：两页共享完全相同的二级标题或层级结构，且内容紧密相关。

**判定为不属于同一小主题 (Conclusion: 0) 的情况：**
* **新概念开启**：Slide B 引入了一个新的定义、新的算法步骤或新的章节，与 Slide A 的具体焦点不同。
* **逻辑转折**：Slide A 是关于“优点”，Slide B 转向“缺点”；或 Slide A 是“背景”，Slide B 转向“方法论”（除非它们被明确包裹在同一个极小的逻辑块中）。
* **独立性**：Slide B 的内容可以独立于 Slide A 存在，且讲述了该大主题下的另一个侧面。

# Negative Constraints (重要限制)
* **严禁**仅仅因为两页内容都属于“文档大主题”就判定它们相似。绝大多数 Slide 都会符合大主题，这**不是**判断依据。你需要寻找的是**微观层面**的语义连贯性。
* 忽略页眉、页脚的页码或通用版权声明造成的文本差异。

# Output Format
严格遵守以下输出格式，不要包含任何 Markdown 代码块标记（如 ```）：

Analysis: [在此处简要说明分析过程。指出两页内容的具体差异或联系，明确说明为何判定为相同或不同。]
Conclusion: [0 或 1]
"""

        self.user_prompt = """请分析以下两页 Slide 的文本内容：

**Document Main Topic (文档大主题)**:
{{main_topic}}

**Slide A Text (前一页)**:
\"\"\"
{{slide_text_a}}
\"\"\"

**Slide B Text (后一页)**:
\"\"\"
{{slide_text_b}}
\"\"\"

根据 System Prompt 中的标准进行分析并给出结论。
"""

        self.sys_prompt_ocr = """# Role
你是一名资深的教育内容结构分析专家。你的任务是分析给定 PDF 文档中两页 Slide（幻灯片）的内容，结合**页面常规文本**和**图片 OCR 文本**，判断它们是否属于同一个**微观子主题（Sub-topic）**或**具体的知识点单元**。

# Context
你将获得以下信息：
1. **Document Main Topic (文档大主题)**：整个 PDF 文档的核心议题。
2. **Slide A**：包含常规文本 (Main Text) 和图片 OCR 内容 (OCR Text)。
3. **Slide B**：包含常规文本 (Main Text) 和图片 OCR 内容 (OCR Text)。

# Analysis Criteria (判断标准)
请综合常规文本和 OCR 信息，基于以下逻辑进行精细化判断：

**判定为属于同一小主题 (Conclusion: 1) 的情况：**
* **图文互补**：Slide A 的常规文本在解释某个概念，而 Slide B 的图片 OCR 内容正是该概念的图表、代码截图或公式推导（反之亦然）。
* **视觉流延续**：Slide A 和 Slide B 的图片 OCR 内容显示了同一个流程图的不同部分，或同一个算法步骤的连续演示。
* **内容/列表延续**：Slide B 是 Slide A 未完成句子的延续，或继续列举 Slide A 中未讲完的列表。
* **强相关性**：即便两页标题不同，但 Slide B 的 OCR 内容明显是对 Slide A 提出的问题的解答或数据的可视化展示。

**判定为不属于同一小主题 (Conclusion: 0) 的情况：**
* **新概念/新章节**：Slide B 引入了新的定义、新的实验结果或进入了新的章节标题，且与 Slide A 的具体焦点不再重合。
* **图表含义断裂**：Slide A 的 OCR 内容展示的是“模型 A 的架构”，而 Slide B 的 OCR 内容展示的是“模型 B 的架构”或“实验结果对比”，且两页之间没有明显的过渡语句。
* **独立性**：两页内容虽然都属于文档大主题，但在微观逻辑上是并列关系（如：分别介绍两种不同的算法），而非承接关系。

# Negative Constraints (重要限制)
* **OCR 噪音处理**：OCR 文本可能包含乱码、碎片化字符或错误的标点。请关注其中的**关键词、核心术语和数字**，忽略格式噪音。
* **严禁**仅仅因为两页内容都属于“文档大主题”就判定它们相似。
* 如果某页没有 OCR 内容（标记为 None 或空），则仅基于常规文本进行判断。

# Output Format
严格遵守以下输出格式，不要包含 Markdown 代码块标记：

Analysis: [在此处简要说明分析过程。请明确指出是依据文本连续性还是 OCR 内容的关联性做出的判断。]
Conclusion: [0 或 1]
"""

        self.user_prompt_ocr = """请分析以下两页 Slide 的内容（包含文本和图片 OCR 信息）：

**Document Main Topic (文档大主题)**:
{{main_topic}}

---
**Slide A (前一页)**:
[Main Text]:
\"\"\"
{{slide_text_a}}
\"\"\"
[Image OCR Text]:
\"\"\"
{{slide_ocr_a}}
\"\"\"

---
**Slide B (后一页)**:
[Main Text]:
\"\"\"
{{slide_text_b}}
\"\"\"
[Image OCR Text]:
\"\"\"
{{slide_ocr_b}}
\"\"\"

---
根据 System Prompt 中的标准（特别是图文互补和逻辑延续性）进行分析并给出结论。
"""

    def ocr_image(self, img_path: str):
        try:
            itexts = []
            if self.ocr_model is None:
                self.ocr_model = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
            res = self.ocr_model.predict(img_path, return_word_box=False)
            for r in res:
                itexts.extend(r['rec_texts'])
            return "\n".join(itexts)
        
        except Exception as e:
            print(f"Exception when ocr: \n{e}")
            return None

    def check_sim(self, theme_desc, text1, images1, text2, images2, use_ocr = True) -> str:
        if use_ocr:
            images1_text = ""
            images2_text = ""
            images1_texts = []
            images2_texts = []

            for img_path in images1:
                img_text = self.ocr_image(img_path)
                if isinstance(img_text, str):
                    images1_texts.append(img_text)
            for img_path in images2:
                img_text = self.ocr_image(img_path)
                if isinstance(img_text, str):
                    images2_texts.append(img_text)
            
            if images1_texts:
                images1_text = "\n\n".join(images1_texts)
            if images2_texts:
                images2_text = "\n\n".join(images2_texts)
            
            if images1_text == "" and images2_text == "":
                use_ocr = False
        elif images1 is not None or images2 is not None:
            print("Warning: Using images but not enabling OCR, omitting images")
        
        user_prompt = self.user_prompt_ocr if use_ocr else self.user_prompt
        user_prompt = user_prompt.replace("{{main_topic}}", theme_desc).replace("{{slide_text_a}}", text1).replace("{{slide_text_b}}", text2)
        if use_ocr:
            user_prompt = user_prompt.replace("{{slide_ocr_a}}", images1_text).replace("{{slide_ocr_a}}", images2_text)
        return chat(
            model="qwen3:4b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt_ocr if use_ocr else self.sys_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={"seed": 1234},
        ).message.content
    
    def extract_label(self, llm_output: str):
        import re
        # Conclusion:  -> 匹配固定的文字
        # \s* -> 匹配任意数量的空白字符（空格、换行、Tab等）
        # ([01])       -> 捕获组，只匹配数字 0 或 1
        pattern = r"Conclusion:\s*([01])"
        
        match = re.search(pattern, llm_output, re.IGNORECASE)
        
        if match:
            return int(match.group(1))
        else:
            return None
