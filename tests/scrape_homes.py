from bs4 import BeautifulSoup
import requests
from urllib import urlretrieve
import os
import time
import boto


class Scraper(object):

    def __init__(self, search, n_photos,
                 bucket=None, aws_un = None, aws_pw = None,
                 outfolder='images'):

        self.search = search
        self.n_photos = n_photos
        self.pages2scrape = (n_photos / 1000) + 1
        self.page = 1
        self.site = self.make_site()
        self.outfolder = outfolder
        self.bucket = bucket
        self.aws_un = aws_un
        self.aws_pw = aws_pw
        self.all_urls = self.get_img_urls_many_pages()

    def make_site(self, page = None):
        '''
        INPUT: self
        OUTPUT: the url to scrape w/ the search term and pg # included
        '''
        if page == None:
            page = self.page
        return( 'https://www.dreamstime.com/search.php?srh_field={}&s_ph=y&s_il=y&s_rf=y&s_ed=y&s_clc=y&s_clm=y&s_orp=y&s_ors=y&s_orl=y&s_orw=y&s_st=new&s_sm=all&s_rsf=0&s_rst=7&s_mrg=1&s_sl0=y&s_sl1=y&s_sl2=y&s_sl3=y&s_sl4=y&s_sl5=y&s_mrc1=y&s_mrc2=y&s_mrc3=y&s_mrc4=y&s_mrc5=y&s_exc=&items=1000&pg={}'.format(self.search, page)
               )

    def get_img_urls_per_page(self, url=None):
        '''
        Input: a url of the site to scrape
        Output: list of urls for the images displayed on page

        gets the html parses the html to find the image links on a webpage
        '''

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
        '''
        Input: None
        Output: A list of all urls to retrieve images
        '''
        ls = []
        for pg in range(1, self.pages2scrape+1):
            self.page = pg
            url = self.make_site(page = pg)
            ls2 = self.get_img_urls_per_page(url)
            ls = ls + list(ls2)
        return set(ls)

    def download_subset(self, ls = None):
        '''
        INPUT: none or list
        OUTPUT: none

        takes a list of urls to be downloaded and downloads them
        to the outfolder.
        '''
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

    def transfer_2_s3(self):
        already_in = [f.name for f in self.bucket.list()]
        for x in os.listdir(self.outfolder):
            if x in already_in:
                continue
            filename = self.outfolder + '/' + x
            k = self.bucket.new_key(x)
            k.set_contents_from_filename(filename)
            already_in += x


    def empty_image_dir(self):
        for f in os.listdir(self.outfolder):
            file_del = self.outfolder + '/' + f
            os.remove(file_del)

    def download_move(self):
        '''
        INPUT: None
        OUTPUT: None

        meat of the class. This downloads the photos
        '''
        for x in range(100, len(self.all_urls)+1, 100):
            lower = x - 100
            to_scrape = list(self.all_urls)[lower:x]
            print 'scraping {} - {}'.format(lower, x)
            self.download_subset(ls = to_scrape)
            print 'sleeping'
            self.transfer_2_s3()
            time.sleep(15)
            self.empty_image_dir()



if __name__ == '__main__':

    access_key = os.environ['AWS_ACCESS_KEY_ID1']
    access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY1']

    conn = boto.connect_s3(access_key, access_secret_key)

    bucket_name = 'ajfcapstonetravel'



    travel_terms = ['vacation', 'spain vacation',
                    'new zealand vacation',
                    'mexico vacation', 'Chicago vacation',
                    'Las Vegas Vacation', 'road trip',
                    'new orleans vacation', 'ski trip']

    home_terms = ['home', 'apartment', 'home repairs',
                  'home renovation', 'new home', 'moving',
                  'home maintanence', 'tools', 'tool kit']

    cars = ['car', 'truck', 'automobile', 'road',
            'convertible', 'sedan', 'classic cars',
            'Toyota', 'Chevrolet','chevy', 'Honda',
            'Nissan', 'Ford', 'motorcycle', 'moped']

    special_events = ['Wedding', 'Funeral', 'Graduation',
                      'College' 'Christmas', 'birth',
                      'baby', 'newborn', 'engagement',
                      'birthday', 'engagement ring',
                      'presents', 'gifts']

    savings = ['retirement', 'savings', 'finance', 'debt',
               'emergency', 'money', 'piggy bank',
               'retire', 'bank', 'stock market']

    lists = [travel_terms, home_terms, cars,
             special_events, savings]
    buckets = ['ajfcapstonetravel',
               'ajfcapstonehome',
               'ajfcapstonecars',
               'ajfcapstonspecevents',
               'ajfcapstonesavings']


    for cat, bucket in zip(lists, buckets):
        b = conn.get_bucket(bucket)
        for term in cat:
            scrape = Scraper(term, 15000, bucket = b,
                             aws_un = access_key,
                             aws_pw = access_secret_key)
            scrape.download_move()
