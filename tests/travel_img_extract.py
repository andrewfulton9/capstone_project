# Use to scrape all the stock vacation photos from
# http://photoeverywhere.co.uk/


from bs4 import BeautifulSoup
import requests
from urllib import urlretrieve
import os


def get_col_urls_list(soup_obj):
    ls = set()
    links = soup_obj.find_all('a')
    for link in links:
        url = link.get('href')
        cats = ['east', 'west']
        param = 'index' not in url
        if url[:4] in cats and len(url) > 5 and param:
            if url[-1] == '/':
                 url = url[:-1]
            ls.add(url)
    return ls

def col_url_list(urllist, root_site = 'http://photoeverywhere.co.uk/'):
    ls = set()
    for url in urllist:
        full_url = root_site + url
        ls.add(full_url)
    return ls

def combine_image_urls(d):
    ls = set()
    for key in d:
        for value in d[key]:
            url = key + '/' + value
            ls.add(url)
    return ls

def parse_image_url(url):
    html1 = requests.get(url).content
    soup1 = BeautifulSoup(html1, 'html.parser')
    imgs = soup1.find_all('a')
    img_ls = []
    for img in imgs:
        img_url = img.get('href')

        param1 = img_url[:6] == 'slides'
        param2 = 'index' not in img_url
        if param1 and param2:
            splat = img_url.split('/')
            splat = splat[1].split('.')
            img_url = splat[0] + '.jpg'
            img_ls.append(img_url)
    return img_ls

def get_ind_img_urls(full_url_list):
    d = {}
    for url in full_url_list:
        img_ls = parse_image_url(url)
        x = 2
        while x < 6:
            try:
                new_url = url + '/index{}.htm'.format(x)
                ls2 = parse_image_url(new_url)
                img_ls = img_ls + ls2
            except:
                continue
            x += 1
        d[url] = img_ls
    ls = combine_image_urls(d)
    return ls

def download_photos(ls, outfolder = 'images/'):

    for url in ls:
        filename = url.split('/')[-1]
        outpath = os.path.join(outfolder, filename)

        urlretrieve(url, outpath)

if __name__ == '__main__':
    site = 'http://photoeverywhere.co.uk/'
    html = requests.get(site).content
    soup = BeautifulSoup(html, 'html.parser')

    ls = get_col_urls_list(soup)
    col_urls = col_url_list(ls, root_site = site)
    d = get_ind_img_urls(col_urls)

    outfolder = '~/Dropbox/Galvanize/capstone_project/tests/images'
    download_photos(d)
