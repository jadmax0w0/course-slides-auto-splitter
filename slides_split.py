from tqdm import tqdm

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


def main(pdf_path: str, theme_desc: str):
    pdf_processor = PdfProcessor(pdf_path)
    pdf_processor.pre_split(max_pages_per_split=1)

    page_texts = []
    page_images = []

    for pp in range(pdf_processor.pdf_page_count):
        text, image_paths = pdf_processor.get_page_info(pp)
        page_texts.append(text)
        page_images.append(image_paths)
    
    page_sim = LLMPageSimilarity()

    page_theme_categorized = [[0]]

    for pp in tqdm(range(1, len(page_texts)), desc="Checking page", total=len(page_texts) - 1):
        ptext0, ptext1 = page_texts[pp - 1], page_texts[pp]
        pimgs0, pimgs1 = page_images[pp - 1], page_images[pp]
        judge = page_sim.check_sim(theme_desc, ptext0, pimgs0, ptext1, pimgs1, use_ocr=True)
        result = page_sim.extract_label(judge)
        if result == 1:
            page_theme_categorized[-1].append(pp)
        else:
            page_theme_categorized.append([pp])
    
    import pdb; pdb.set_trace()
    
    pdf_processor.finalize()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-d", "--theme-desc", type=str, required=True)

    args = parser.parse_args()

    main(args.file, args.theme_desc)
