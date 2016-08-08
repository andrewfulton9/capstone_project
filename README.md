# Dobot Image Classifier

A project in collaboration with Dobot, Inc. with the purpose of helping users determine appropriate savings goals based on images users upload to represent their savings goals

### Goal

The goal of this project is to classify images that users upload to represent their savings goals in order to better understand what users savings goals are.

### Motivations

My interest in this completing project largely arised from two things. First American's savings habits have become pretty abysmal as this point, with people spending more and saving less. For example, according to a survey conducted by GOBankingRates.com, 62% of Americans have less than $1000 in savings and 21% don't even have a savings account.

Second due to my background in Neuroscience, I think the idea of method of analysis that is based on how neurons in the central nervous system communicate is really interesting and so I wanted to broaden my understanding of neural nets and practice implementing them

### Tools

##### Computing
  - AWS EC2

##### Storage
  - AWS S3

##### Scraping
  - BeautifulSoup

##### Processing
  - Numpy
  - Pandas
  - SciKit-Image

##### Neural Nets
  - Keras
  - Theano

### Scraping

The first step of my project was getting a training set on which to train my model on. I began by deciding which categories to use to train my model. Based on the data that I got from Dobot I settled on the following five:
  - Car expenses
  - Home expenses
  - Special events
  - General savings
  - Travel

Next I began looking at places to scrape from. My goal was to get around 100,000 images per category since I expected a fair amount of leakage between groups and because neural nets are notoriously expensive to train. I found a stock photo site that I could get ~10,000 images per keyword so I next came up with 10-15 keywords per category to scrape from.

### Processing

Once I had all my images scraped, I needed to process them. To do this I down sampled each to photo to 50x50 and 100x100 pixels to standardize the shape of each photo and to see how much of an effect the size of each photo would have on my models. I also took each of these 50x50 and 100x100 arrays for each photo and broke them into another dimension so that each image was represented by a 3x50x50 and 3x100x100 arrays with the top dimension representing the red, blue and green pixel values for each pixel. To do this processing I used python's SciKit-Image library. I again used an EC2 AWS machine to run the program and again saved the images that I scraped in AWS S3 buckets.

### Convolutional Neural Net

Once all my images were processed I used then to train the neural nets. I wanted to try a couple of different models for my task. After some research I decided on the 16 and 19 layered VGG convolutional Neural Nets. Each of which have 5 convolutional layers with a max pooling step between each one and 3 dense fully connected layers. I batch trained each network to account for the massive size of the training set which ended up being around 400,000 images. I ran the nets on GPU optimized AWS EC2 instances in order to reduce the runtime on training the models. I used the Keras python librarty with theanos backend to build the models
