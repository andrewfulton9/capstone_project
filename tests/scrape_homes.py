from bs4 import BeautifulSoup
import requests
from urllib import urlretrieve
import os
import time

class ScrapeHomes(object):

    def __init__(self, site, n_photos, outfolder):
        self.site = site
        self.n_photos = n_photos
        self.outfolder = outfolder
        self.pg1_urls = self.get_img_urls_page()
        self.all_urls = self.get_img_urls_many_pages()

    def get_img_urls_page(self, url=None):
        if url:
            html = requests.get(url).content
        else:
            html = requests.get(self.site).content

        soup = BeautifulSoup(html, 'html.parser')
        imgs = soup.find_all('img')
        ls = set()
        for img in imgs:
            link = img.get('src')
            ls.add(link)
        return ls

    def get_img_urls_many_pages(self):
        ls = []
        for x in range(0, self.n_photos+1, 75):
            url = self.site
            if x > 74:
                url = url + '#start:{}'.format(x)
            ls2 = self.get_img_urls_page(url)
            ls = ls + list(ls2)
        #ls = [url for url in ls if 'canstockphoto' in url]
        self.all_urls_list = ls
        return set(ls)

    def download_photos(self, ls = None):
        if ls:
            urls = ls
        else:
            urls = self.all_urls
        for url in urls:
            # time.sleep(1)
            filename = url.split('/')[-1]
            outpath = os.path.join(self.outfolder, filename)

            if filename in os.listdir(self.outfolder):
                break
            try:
                urlretrieve(url, outpath)
            except:
                continue

if __name__ == '__main__':

    outpath = '../../../../Desktop/CapstoneImages/Home'
    #outpath = 'images'

    site = 'https://www.dreamstime.com/search.php?srh_field=house&s_ph=y&s_il=y&s_rf=y&s_ed=y&s_clc=y&s_clm=y&s_orp=y&s_ors=y&s_orl=y&s_orw=y&s_st=new&s_sm=all&s_rsf=0&s_rst=7&s_mrg=1&s_sl0=y&s_sl1=y&s_sl2=y&s_sl3=y&s_sl4=y&s_sl5=y&s_mrc1=y&s_mrc2=y&s_mrc3=y&s_mrc4=y&s_mrc5=y&s_exc=&items=1000&pg='

    for x in range(1, 101):
        pg = site + str(x)
        imgs = ScrapeHomes(pg, 50, outpath+'/dreamstime')

        for x in range(100, len(imgs.all_urls) + 1, 100):
            lower = x - 100
            to_scrape = list(imgs.all_urls)[lower:x]
            print 'scraping {} - {}'.format(lower, x)
            imgs.download_photos(ls = to_scrape)
            print 'sleeping...'
            time.sleep(30)

    site2 = 'http://www.canstockphoto.com/images-photos/home.html'

    imgs2 = ScrapeHomes(site2, 50, outpath + '/canstock')

    imgs2.download_photos()

    print 'dreamstime urls', len(imgs.all_urls)
    print 'dreamstime images: ', len(os.listdir(imgs.outfolder))
    print
    print 'canstock urls', len(imgs2.all_urls)
    print 'canstock images: ', len(os.listdir(imgs2.outfolder))
#
# http://www.canstockphoto.com/images-photos/home.html
# http://www.canstockphoto.com/images-photos/home.html#start:75

# https://www.dreamstime.com/search.php?srh_field=house&s_ph=y&s_il=y&s_rf=y&s_ed=y&s_clc=y&s_clm=y&s_orp=y&s_ors=y&s_orl=y&s_orw=y&s_st=new&s_sm=all&s_rsf=0&s_rst=7&s_mrg=1&s_sl0=y&s_sl1=y&s_sl2=y&s_sl3=y&s_sl4=y&s_sl5=y&s_mrc1=y&s_mrc2=y&s_mrc3=y&s_mrc4=y&s_mrc5=y&s_exc=&items=10000&pg=1
