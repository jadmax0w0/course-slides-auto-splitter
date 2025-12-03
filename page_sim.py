from abc import ABC

from ollama import chat, ChatResponse
from paddleocr import PaddleOCR


class PageSimilarityBase(ABC):
    def check_sim(self, theme_desc: str, text1: str, images1: list[str], text2: str, images2: list[str], **kwargs) -> bool:
        raise NotImplementedError()


class LLMPageSimilarity(PageSimilarityBase):
    def __init__(self):
        super().__init__()
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

    def ocr_image(self, img_path: str):
        try:
            itexts = []
            ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
            res = ocr.predict(img_path, return_word_box=False)
            for r in res:
                itexts.extend(r['rec_texts'])
            return "\n".join(itexts)
        
        except Exception as e:
            print(f"Exception when ocr: \n{e}")
            return None

    def check_sim(self, theme_desc, text1, images1, text2, images2, use_ocr = True) -> str:
        if use_ocr:
            images1_text = []
            images2_text = []

            for img_path in images1:
                img_text = self.ocr_image(img_path)
                if isinstance(img_text, str):
                    images1_text.append(img_text)
            for img_path in images2:
                img_text = self.ocr_image(img_path)
                if isinstance(img_text, str):
                    images2_text.append(img_text)
            
            if images1_text:
                text1 += "\n\n" + "\n\n".join(images1_text)
            if images2_text:
                text2 += "\n\n" + "\n\n".join(images2_text)
        elif images1 is not None or images2 is not None:
            print("Warning: Using images but not enabling OCR, omitting images")
        
        user_prompt = self.user_prompt
        user_prompt = user_prompt.replace("{{main_topic}}", theme_desc).replace("{{slide_text_a}}", text1).replace("{{slide_text_b}}", text2)
        return chat(
            model="qwen3:4b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt,
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
