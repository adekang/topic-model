### 第一步：读取pdf_dir路径下所有.pdf为后缀的文件，打开CSV文件以写入文件名和标题
### 第二步：手动对CSV文件内错误标题进行修改


# 读取路径下所有.pdf为后缀的文件
pdf_dir = f'D:\\DeskTop\\work\\北理工\\文献数据\\1001-2011-ycy\\entry'
# 合并后的PDF名字
output_pdf_path = "./老师的论文集.pdf"
# 用于中间 存放文件名与标题的CSV文件
TitlesCSV = './老师的论文集.csv'

import csv
import html
import os
import re  # 导入正则表达式模块

import fitz  # PyMuPDF
from PyPDF2 import PdfReader


def find_non_text_chars(sentence):
    # 用于检测提取的文本中是否出现非文本类型的，若有则通过类似title = title.replace("ﬁ", "fi")替换
    import regex as re
    # 定义正则表达式，匹配非文本字符（除了字母、数字、空格和标点符号之外的字符）
    non_text_pattern = re.compile(r'[^a-zA-Z0-9\s\p{P}]', re.UNICODE)
    # 使用正则表达式搜索句子中的非文本字符
    non_text_chars = non_text_pattern.findall(sentence)
    # 打印出非文本字符及其类型
    for char in non_text_chars:
        print(title)
        print(f"非文本字符 '{char}' 的类型是 '{type(char)}\n\n'")
    return None


def get_pdf_title_1(pdf_path):
    """读取PDF文件的标题，并进行处理。"""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        doc_info = pdf_reader.metadata
        # 尝试从文档信息中获取标题
        paper_title = doc_info.get('/Title', 'untitled') if doc_info else 'untitled'

        # 如果标题有效，则进行进一步处理
        if paper_title != 'untitled' and paper_title != 'Untitled' and not paper_title.endswith('.pdf'):
            # 解码HTML实体
            paper_title = html.unescape(paper_title)
            # 替换不适合作为文件名的字符
            paper_title = re.sub(r'[:/\\*?"\'<>|]', ' ', paper_title)
        else:
            # 无效的标题，返回默认值
            paper_title = 'untitled'
        return paper_title


def get_pdf_title_2(pdf_path):
    # 检查文件名是否符合特定模式
    filename = os.path.basename(pdf_path)
    if filename == "xxx.pdf":
        return "Estynamical systems"

    doc = fitz.open(pdf_path)
    first_page = doc[0]  # 只查看第一页

    # 获取页面上所有文本块，每个块包含文字、字体大小和位置
    blocks = first_page.get_text("dict")["blocks"]
    # 只考虑页面上半部分的文本块
    mid_y = first_page.rect.height / 2
    top_blocks = [b for b in blocks if b['type'] == 0 and b['bbox'][3] < mid_y]

    # 提取每个文本块的字体大小和文本内容
    text_blocks_with_size = []
    for block in top_blocks:
        if 'lines' in block:  # 确保文本块包含行
            for line in block['lines']:
                if 'spans' in line:  # 确保行包含span
                    for span in line['spans']:
                        if 'size' in span and len(span['text'].strip()) >= 2:  # 检查span中是否有size信息且文本长度符合要求
                            text_blocks_with_size.append((span['text'], span['size'], span['bbox']))

    # 排除特定关键词
    excluded_keywords = ["Research Article", "Physica A", "Neurocomputing",
                         "Sustainable Energy Technologies and Assessments"]
    filtered_blocks = [block for block in text_blocks_with_size if
                       not any(keyword in block[0] for keyword in excluded_keywords)]

    # 在过滤后的文本块中基于字体大小和垂直位置来识别可能的标题
    if filtered_blocks:
        max_font_size = max([size for _, size, _ in filtered_blocks], default=0)
        possible_title_blocks = [block for block in filtered_blocks if block[1] == max_font_size]

        # 合并具有相同最大字体大小的连续文本块
        title_texts = [block[0] for block in possible_title_blocks]
        title = " ".join(title_texts) if title_texts else "untitled"
    else:
        title = "untitled"

    doc.close()
    title = title.replace("ﬁ", "fi")
    title = title.replace("ﬀ", "ff")
    # 查找句子中的非文本字符
    find_non_text_chars(title)
    return title


def get_pdf_title(pdf_path):
    # 先使用get_pdf_title_1获取标题，若获取失败则使用get_pdf_title_2获取
    paper_title = get_pdf_title_1(pdf_path)  # 假设这是从PDF提取标题的函数
    # 编写一个正则表达式来匹配以连续4个数字和.pdf为后缀的字符串
    # 匹配以连续三个数字和.pdf结尾的字符串，或者包含空格和点的字符串，以及不包含空格但包含点的字符串
    regex_pattern = r'\d{3}\.pdf$|^[A-Z]+-\w+\s\d+\.\.\d+$|\w+\.\d+\s\d+\.\.\d+$|^[a-zA-Z]+_\d+\w*$'

    # 判断条件：标题不是'untitled'且不匹配正则表达式（即不是以连续4个数字和.pdf结尾）
    if paper_title != 'untitled' and not re.search(regex_pattern, paper_title):
        return paper_title
    else:
        paper_title = get_pdf_title_2(pdf_path)
        return paper_title


def get_titles_from_directory(directory_path, specific_file):
    titles = []
    specific_pdf_path = None  # 用于存储特定文件的路径
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                if file == specific_file:  # 如果当前文件是特定文件
                    specific_pdf_path = pdf_path
                else:
                    try:
                        title = get_pdf_title(pdf_path)
                        titles.append((file, title))
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

    # 处理特定文件
    if specific_pdf_path:
        try:
            title = get_pdf_title(specific_pdf_path)
            titles.insert(0, (specific_file, title))  # 将特定文件的标题插入到列表的最前面
        except Exception as e:
            print(f"Error processing {specific_file}: {e}")

    return titles


if __name__ == '__main__':
    specific_file = "lic health.pdf"
    # 替换为你的PDF文件所在的目录路径
    directory_path = pdf_dir
    titles = get_titles_from_directory(directory_path, specific_file)
    with open(TitlesCSV, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['Files', 'Title'])  # 写入头部信息
        for file, title in titles:
            # 写入文件名和标题
            csv_writer.writerow([file, title])
