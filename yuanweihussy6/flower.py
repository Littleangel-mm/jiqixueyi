import os
import requests
from bs4 import BeautifulSoup

# 示例：抓取图片
def download_images(url, save_folder='images'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 从 URL 获取网页内容
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 找到所有 img 标签
    images = soup.find_all('img')
    img_urls = [img['src'] for img in images if 'src' in img.attrs]

    # 下载图片
    for idx, img_url in enumerate(img_urls):
        img_data = requests.get(img_url).content
        img_name = os.path.join(save_folder, f'image_{idx}.jpg')
        with open(img_name, 'wb') as f:
            f.write(img_data)

# 示例 URL
url = 'https://unsplash.com/s/photos/flowers'
download_images(url)
