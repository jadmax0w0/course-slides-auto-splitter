import sys; sys.path.append(".")
from pdf_proc import PdfProcessor
from page_sim import LLMPageSimilarity


"""
流程：
1. 导出所有页面
2. 针对每个页面，导出其文本 & 图像
3. 调用相似度判断函数，判断相邻两页是否属于同一个主题；是则归入一类，不是则创建新类别
4. 遍历每个主题，查看里面的页面数是否超过阈值，如果超过，那么继续针对这个类别细分
"""


def main(pdf_path: str):
    pdf_processor = PdfProcessor(pdf_path)
    pdf_processor.pre_split(max_pages_per_split=1)

    page_texts = []
    page_images = []

    for pp in range(pdf_processor.pdf_page_count):
        text, image_paths = pdf_processor.get_page_info(pp)
        page_texts.append(text)
        page_images.append(image_paths)
    
    page_theme_categorized = []

    for 
    
    pdf_processor.finalize()