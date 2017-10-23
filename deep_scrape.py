# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:12:34 2017

@author: Poul Gunnar
"""

from lxml import html
import requests


#Collects hyperlinks from the front page of www.denstoredanske.dk, from the most active currently, daily, and weekly. 
def starter():
    link = "http://denstoredanske.dk/"
    page = requests.get(link)
    tree = html.fromstring(page.content)
    k = tree.xpath('//div[@class="icon badge-10 cols-3"]//@href')
    return(k)


# scrapes the main text body for hyperlinks on the given hyperlink. Must be an article on www.denstoredanske.dk
def getlinks(x):
    page = requests.get(x)
    tree = html.fromstring(page.content)
    links = tree.xpath('//div[@class="body-content"]/div[@id]/p//@href|//div[@class="body-content"]/p//@href')
    return(links)
  

# scrapes text from the main body from all the hyperlinks in the given list. Must be articles on www.denstoredanske.dk
def gettext(x):
    result = []
    for i in x:
        page = requests.get(i)
        tree = html.fromstring(page.content)
        texts = tree.xpath('//div[@class="body-content"]/div[@id]/p//text()|//div[@class="body-content"]/p//text()')
        result.extend(texts)
    return(result)
    

# Collects hyperlinks recursively from the front page 'most active' links, and the links in the main body of the underlying, for 'x' recursions.
# checks for duplicates through set.    
def vraps(x):
    checked = []
    #get starter
    links = set(starter())
    #itterate over desired recursions
    for k in range(x):
        #clean list of checked links
        listtoget = [z for z in links if z not in checked]
        #get links from unchecked sites and extend the checked list
        for i in listtoget:
            n = getlinks(i)
            links.update(n)
            checked.append(i)
    return(links)
    

def forkortelser():
    x = "https://syntaksis.dk/forkortelser/"
    page = requests.get(x)
    tree = html.fromstring(page.content)
    shorts_1 = tree.xpath('//td[@class="column-1"]/text()')
    shorts_1_1 = [x.split("(") for x in shorts_1]
    result = [val.replace(")", "") for sublist in shorts_1_1 for val in sublist]
    
    x = "http://www.grafisk-litteratur.dk/?id=16"
    page = requests.get(x)
    tree = html.fromstring(page.content)
    shorts_2 = tree.xpath('//div[@class="forkortelse"]/text()')
    #for i in shorts_2
    
    shorts = result
    return(shorts)
    
# This is where we combine everything
def getter(x):
    #collect links
    links = vraps(x)
    text = []
    #itterate through links to get text and collect it in one big list
    for i in links:
        res = gettext(i)
        text.extend(res)
    #join all text in list into one string
    text = ''.join(text)    
    
    # here we need to split it based on periods, but ignoring 'forkortelser'
    return(text)


# write it all to a .csv file. REMEMBER, IMPORT WITH UTF-8 ENCODING!!!
def importer(num):
    res = getter(num)
    with open("output.csv", 'wb', encoding = "utf-8") as resultFile:
        wr = csv.writer(resultFile, dialect = "excel")
        wr.writerow(res)    
