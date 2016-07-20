

from bs4 import BeautifulSoup
import requests
from urllib import urlretrieve
import os

def get_img_urls_page(site):
    html = requests.get(site).content
    soup = BeautifulSoup(html, 'html.parser')
    imgs = soup.find_all('img')
    ls = set()
    for img in imgs:
        link = img.get('src')
        ls.add(link)
    return ls

def get_img_urls_many_pages(site, n):
    ls = []
    for x in range(1, n+1):
        url = site
        if x > 1:
            url = url + '{}'.format(x)
        ls2 = get_img_urls_page(url)
        ls = ls + list(ls2)
    return set(ls)

def download_photos(ls, outfolder = 'images/'):

    for url in ls:
        filename = url.split('/')[-1]
        outpath = os.path.join(outfolder, filename)

        urlretrieve(url, outpath)

if __name__ == '__main__':
    site = 'http://foter.com/car/'

    img_urls = get_img_urls_many_pages(site, 18)

    download_photos(img_urls)
