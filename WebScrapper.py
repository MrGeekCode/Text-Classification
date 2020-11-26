import os
import requests
from bs4 import BeautifulSoup
import random
import time
time_list = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,15,14]
file_name = []
# Getting file names from different folders and storing them in file_name list
os.chdir(r'C:\Users\mr.geek\Desktop\Mphil thesis')
path = os.getcwd()
for (dirpath, dirnames, filenames) in os.walk(path):
    file_name.extend(filenames)
# Creating Relevant Folders in scraped data root folder on desktop
os.chdir(r'c:\users\mr.geek\Desktop\scraped data')
path = os.getcwd()
os.mkdir(path + "\Philonthropists")
os.mkdir(path + "\Politcians")
os.mkdir(path + "\Showbiz")
os.mkdir(path + "\sportsmen")
os.mkdir(path + "\Writers")
count = 0
for name in file_name:
    if (count > 200):
        os.chdir(r'c:\users\mr.geek\Desktop\scraped data\Writers')
        status = 'Writers Data is being scraped.'
    elif (count > 150):
        os.chdir(r'c:\users\mr.geek\Desktop\scraped data\sportsmen')
        status = 'sportsmen Data is being scraped.'
    elif (count > 100):
        os.chdir(r'c:\users\mr.geek\Desktop\scraped data\Showbiz')
        status = 'Showbiz Data is being scraped.'
    elif (count > 50):
        os.chdir(r'c:\users\mr.geek\Desktop\scraped data\Politcians')
        status = 'Philontrhopists Data is being scraped.'
    else:
        os.chdir(r'c:\users\mr.geek\Desktop\scraped data\Philonthropists')
        status = 'Philontrhopists Data is being scraped.'
    file = open(name, 'w', encoding='ascii')
    filename_wo_ext = os.path.splitext(name)[0]
    url_name = filename_wo_ext.replace(" ", "_")
    url = 'https://en.wikipedia.org/wiki/' + url_name
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.content, 'html.parser')
    body = soup.find(id='bodyContent')
    content = body.find(id='mw-content-text')
    paragraphs = content.find_all('p')
    count = count + 1
    # print(paragraphs[1].get_text())
    for p in paragraphs:
        text = p.get_text()
        ftext = text.encode("ascii", "ignore")
        fftext = ftext.decode("ascii")
        # print(fftext)
        file.write(fftext)
    print(status +" " + name + " File has been scraped and saved. Total of %d files have been scraped so far \n" % count)
    wait = random.choice(time_list)
    time.sleep(wait)
