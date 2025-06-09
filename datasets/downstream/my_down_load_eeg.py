'''import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse

# 认证信息
username = "nedc-tuh-eeg"
password = "L5!Np#8Z$7xJmRpK"

# 目标起始目录（只下载 `edf/` 及以下内容）
base_url = "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_events/v2.0.1/edf/"
save_dir = "./downloads/tuh_eeg_events_v2.0.1"

# 记录已访问的目录，避免无限递归
visited_dirs = set()

# 创建会话，带上身份认证
session = requests.Session()
session.auth = (username, password)


def get_file_links(url):
    """ 获取网页中所有文件和子目录链接 """
    response = session.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        full_url = urljoin(url, href)
        # **过滤掉无效链接**
        if href and not href.startswith("?") and not href.startswith("Parent Directory") and not href.startswith("../"):
            if full_url.startswith(base_url):  # 确保不跳出 `edf/` 目录
                links.append(href)

    return links


def sanitize_filename(url):
    """ 处理非法文件名 """
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    return filename.replace("?", "_").replace(";", "_").replace("=", "_")


def download_file(file_url, local_path):
    """ 下载单个文件 """
    if os.path.exists(local_path):
        print(f"✅ 已存在，跳过: {local_path}")
        return

    print(f"⬇️ 正在下载: {file_url}")
    response = session.get(file_url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), desc=os.path.basename(local_path)):
            f.write(chunk)


def recursive_download(url, local_dir):
    """ 递归下载整个 `edf/` 及以下的 `train/` `eval/` 目录和 `.edf` `.lab` 文件 """
    if url in visited_dirs:
        print(f"⚠️ 已访问，跳过: {url}")
        return  # 避免死循环

    visited_dirs.add(url)  # 记录当前访问的目录
    os.makedirs(local_dir, exist_ok=True)  # **保持原始目录结构**

    links = get_file_links(url)

    for link in links:
        full_url = urljoin(url, link)
        local_path = os.path.join(local_dir, link)  # **保持原始目录结构**

        if link.endswith("/") and full_url.startswith(base_url):  # 目录，递归下载
            print(f"📂 进入子目录: {full_url}")
            recursive_download(full_url, local_path)
        elif (link.endswith(".edf") or link.endswith(".lab") or link.endswith(".rec") or link.endswith(".htk")) and full_url.startswith(base_url):  # 下载 `.edf` 和 `.lab`
            download_file(full_url, local_path)


# 开始递归下载（仅 `edf/` 及其子目录）
recursive_download(base_url, save_dir)
print("🎉 全部下载完成！")
'''

import os
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse

# 认证信息
username = "nedc-tuh-eeg"
password = "L5!Np#8Z$7xJmRpK"

# 目标起始目录（只下载 `edf/` 及以下内容）
base_url = "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_events/v2.0.1/edf/"
save_dir = "./downloads/tuh_eeg_events_v2.0.1"

# 记录已访问的目录，避免无限递归
visited_dirs = set()

# 创建会话，带上身份认证
session = requests.Session()
session.auth = (username, password)

# **全局请求参数**
MAX_RETRIES = 5  # 失败时最大重试次数
TIMEOUT = 15  # 超时时间
DELAY_BETWEEN_REQUESTS = 2  # 访问间隔，防止请求过快被封锁


def get_file_links(url):
    """ 获取网页中所有文件和子目录链接 """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            links = []
            for link in soup.find_all("a"):
                href = link.get("href")
                full_url = urljoin(url, href)
                # **过滤掉无效链接**
                if href and not href.startswith("?") and not href.startswith("Parent Directory") and not href.startswith("../"):
                    if full_url.startswith(base_url):  # 确保不跳出 `edf/` 目录
                        links.append(href)

            time.sleep(DELAY_BETWEEN_REQUESTS)  # **避免请求过快**
            return links

        except requests.exceptions.RequestException as e:
            print(f"⚠️ 请求失败 ({retries + 1}/{MAX_RETRIES})，错误: {e}")
            time.sleep(5)  # 失败后等待 5 秒再重试
            retries += 1

    print(f"❌ 无法获取 {url}，跳过...")
    return []  # 最终请求失败则返回空列表


def sanitize_filename(url):
    """ 处理非法文件名 """
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    return filename.replace("?", "_").replace(";", "_").replace("=", "_")


def download_file(file_url, local_path):
    """ 下载单个文件，支持自动重试 """
    if os.path.exists(local_path):
        print(f"✅ 已存在，跳过: {local_path}")
        return

    retries = 0
    while retries < MAX_RETRIES:
        try:
            print(f"⬇️ 正在下载: {file_url}")
            response = session.get(file_url, stream=True, timeout=TIMEOUT)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc=os.path.basename(local_path)):
                    f.write(chunk)

            time.sleep(DELAY_BETWEEN_REQUESTS)  # **避免请求过快**
            return

        except requests.exceptions.RequestException as e:
            print(f"⚠️ 下载失败 ({retries + 1}/{MAX_RETRIES})，错误: {e}")
            time.sleep(5)  # 失败后等待 5 秒再重试
            retries += 1

    print(f"❌ 无法下载 {file_url}，跳过...")


def recursive_download(url, local_dir):
    """ 递归下载整个 `edf/` 及以下的 `train/` `eval/` 目录和 `.edf` `.lab` `.rec` `.htk` 文件 """
    if url in visited_dirs:
        print(f"⚠️ 已访问，跳过: {url}")
        return  # 避免死循环

    visited_dirs.add(url)  # 记录当前访问的目录
    os.makedirs(local_dir, exist_ok=True)  # **保持原始目录结构**

    links = get_file_links(url)

    for link in links:
        full_url = urljoin(url, link)
        local_path = os.path.join(local_dir, link)  # **保持原始目录结构**

        if link.endswith("/") and full_url.startswith(base_url):  # 目录，递归下载
            print(f"📂 进入子目录: {full_url}")
            recursive_download(full_url, local_path)
        elif (link.endswith(".edf") or link.endswith(".lab") or link.endswith(".rec") or link.endswith(".htk")) and full_url.startswith(base_url):  # 下载 `.edf` `.lab` `.rec` `.htk`
            download_file(full_url, local_path)


# 开始递归下载（仅 `edf/` 及其子目录）
recursive_download(base_url, save_dir)
print("🎉 全部下载完成！")
