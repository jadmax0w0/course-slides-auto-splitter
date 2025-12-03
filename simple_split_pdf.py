def split_pdf(pdf_path: str, start_page: int, end_page: int, out_dir: str = "./pdf_split/"):
    import os
    import pypdf

    pdf_name = os.path.splitext(os.path.split(pdf_path)[1])[0]

    pdf_read = pypdf.PdfReader(pdf_path)

    os.makedirs(out_dir, exist_ok=True)

    if end_page <= 0:
        end_page = "last"
    output_path = os.path.join(out_dir, f"{pdf_name}_pp_{start_page}_{end_page}.pdf")
    with open(output_path, mode="wb") as f:
        pdf_write = pypdf.PdfWriter()
        if end_page == "last":
            end_page = len(pdf_read.pages)
        for pp in range(start_page - 1, end_page):
            pdf_write.add_page(pdf_read.pages[pp])
        pdf_write.write(f)
    
    pdf_read.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-s", "--start-page", type=int, required=True, help="Start from 1")
    parser.add_argument("-e", "--end-page", type=int, default=-1)

    args = parser.parse_args()

    split_pdf(args.file, args.start_page, args.end_page)
