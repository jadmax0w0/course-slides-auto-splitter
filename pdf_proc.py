import os
import pypdf
import spire.pdf as sppdf

import sys; sys.path.append(".")
import utils


PDF_TEMP_DIR = "./pdf_proc_temp/"


class PdfProcessor:
    def __init__(
            self,
            pdf_path: str,
    ):
        self._fpath = pdf_path

        self.pdf_dir, self.pdf_fullname = os.path.split(self._fpath)
        self.pdf_name, self.pdf_ext = os.path.splitext(self.pdf_fullname)

        self.pdf_split_paths = None
        self.pdf_split_ranges = None
    
    @property
    def pdf_page_count(self):
        pdf_read = pypdf.PdfReader(self._fpath)
        cnt = len(pdf_read.pages)
        pdf_read.close()
        return cnt
    
    @property
    def is_pre_split_done(self):
        return utils.has_files_with_prefix(PDF_TEMP_DIR, self.pdf_name)
    
    def pre_split(self, max_pages_per_split: int = 1):
        if self.is_pre_split_done:
            return
        
        assert max_pages_per_split >= 1, f"{max_pages_per_split=}, should be >= 1"

        pdf_read = pypdf.PdfReader(self._fpath)

        split_indices = list(range(0, len(pdf_read.pages), max_pages_per_split)) + [len(pdf_read.pages)]
        split_indices = sorted(list(set(split_indices)))
        self.pdf_split_ranges = list(zip(split_indices[:-1], split_indices[1:]))

        os.makedirs(PDF_TEMP_DIR, exist_ok=True)

        self.pdf_split_paths = []
        for s, t in self.pdf_split_ranges:
            output_path = os.path.join(PDF_TEMP_DIR, f"{self.pdf_name}_pp_{s}_{t}.pdf")
            with open(output_path, mode="wb") as f:
                pdf_write = pypdf.PdfWriter()
                for pp in range(s, t):
                    pdf_write.add_page(pdf_read.pages[pp])
                pdf_write.write(f)
                self.pdf_split_paths.append(output_path)
        
        pdf_read.close()
        
        assert self.is_pre_split_done, "Pre-split check failed"
    
    def get_page_info(self, page_id: int):
        """
        Returns:
            (text, image_paths)
        """
        pdf_fpath = self._fpath
        local_page_id = page_id

        # Map local_page_id to splitted pdf files, if they exist
        if self.is_pre_split_done:
            for split_id, (s, t) in enumerate(self.pdf_split_ranges):
                if s <= page_id < t:
                    pdf_fpath = self.pdf_split_paths[split_id]
                    local_page_id = page_id - s
                    break
        
        # Use the MIGHTY spire pdf to read page elements (pre-split is to bypass the free-version limitations of spire.pdf)
        pdf = sppdf.PdfDocument()
        pdf.LoadFromFile(pdf_fpath)

        page = pdf.Pages.get_Item(local_page_id)

        text_extract = sppdf.PdfTextExtractor(page)
        image_helper = sppdf.PdfImageHelper()

        # Buffer temp text
        text = text_extract.ExtractText(sppdf.PdfTextExtractOptions())

        # Save temp images
        images_info = image_helper.GetImagesInfo(page)
        image_paths = []
        for img_id, img_info in enumerate(images_info):
            outpath = os.path.join(PDF_TEMP_DIR, f"{self.pdf_name}_pp_{page_id}_img_{img_id}.png")
            image_paths.append(outpath)
            img_info.Image.Save(outpath)
        
        pdf.Close()
        
        return text, image_paths
    
    def finalize(self):
        related_temp_files = []
        utils.has_files_with_prefix(PDF_TEMP_DIR, self.pdf_name, related_temp_files)

        for fpath in related_temp_files:
            os.remove(fpath)