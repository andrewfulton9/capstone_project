from image_processing import *

bucket_ls = ['ajfcapstonecars', 'ajfcapstonehome', 'ajfcapstonesavings',
                 'ajfcapstonespecevents', 'ajfcapstonetravel']

bin_save_arrs(bucket_ls, img_size = 100, bin_size=25000, sample_size='half')
