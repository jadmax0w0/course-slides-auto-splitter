import os
from typing import Optional

def has_files_with_prefix(directory, prefix, out_list: Optional[list] = None):
    """
    检查目录中是否存在至少一个文件，其文件名以指定前缀开始。
    
    Args:
        directory (str): 要检查的目录路径。
        prefix (str): 文件名前缀。
        
    Returns:
        bool: 如果存在匹配的文件，返回True；否则返回False。
    """
    if isinstance(out_list, list):
        out_list.clear()

    # 检查目录是否存在且是一个目录
    if not os.path.isdir(directory):
        return False
    
    # 遍历目录中的所有条目
    result = False
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        # 检查是否为文件且文件名以前缀开始
        if os.path.isfile(full_path) and filename.startswith(prefix):
            if isinstance(out_list, list):
                out_list.append(full_path)
            result = True
    return result