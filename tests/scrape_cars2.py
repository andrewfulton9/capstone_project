from scrape_homes import ScrapeHomes
import time
import boto


outpath = '../../../../Desktop/CapstoneImages/Cars'
#outpath = 'images'

access_key = os.environ['AWS_ACCESS_KEY_ID1']
access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY1']

site = 'https://www.dreamstime.com/search.php?srh_field=car&s_ph=y&s_il=y&s_rf=y&s_ed=y&s_clc=y&s_clm=y&s_orp=y&s_ors=y&s_orl=y&s_orw=y&s_st=new&s_sm=all&s_rsf=0&s_rst=7&s_mrg=1&s_sl0=y&s_sl1=y&s_sl2=y&s_sl3=y&s_sl4=y&s_sl5=y&s_mrc1=y&s_mrc2=y&s_mrc3=y&s_mrc4=y&s_mrc5=y&s_exc=&items=1000&pg='

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
