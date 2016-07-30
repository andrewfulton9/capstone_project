from image_processing import *

bucket_ls = ['ajfcapstonecars', 'ajfcapstonehome', 'ajfcapstonesavings',
                 'ajfcapstonespecevents', 'ajfcapstonetravel']

process_imgs(bucket_ls, img_size = 50, sample_size='half', bin_size=100000)
