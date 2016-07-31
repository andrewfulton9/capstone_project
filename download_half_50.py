from image_processing import *

bucket_ls = ['ajfcapstonecars', 'ajfcapstonehome', 'ajfcapstonesavings',
                 'ajfcapstonespecevents', 'ajfcapstonetravel']

bin_save_arrs(bucket_ls, img_size = 50, sample_size='half', bin_size=25000)
