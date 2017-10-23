#%%
from lxml import html
import requests
import re


#%%
#Collects hyperlinks from the front page of www.denstoredanske.dk, from the most active currently, daily, and weekly. 
def starter():
    link = "http://denstoredanske.dk/"
    page = requests.get(link)
    tree = html.fromstring(page.content)
    k = tree.xpath('//div[@class="icon badge-10 cols-3"]//@href')
    return(k)

#%%
# scrapes the main text body for hyperlinks on the given hyperlink. Must be an article on www.denstoredanske.dk
def getlinks(x):
    page = requests.get(x)
    tree = html.fromstring(page.content)
    links = tree.xpath('//div[@class="body-content"]/div[@id]/p//@href|//div[@class="body-content"]/p//@href')
    result = [str(i) for i in links]
    return(result)
  
#%%
# scrapes text from the main body from all the hyperlinks in the given list. Must be articles on www.denstoredanske.dk
def gettext(x):
    page = requests.get(x)
    tree = html.fromstring(page.content)
    texts = tree.xpath('//div[@class="body-content"]/div[@id]/p//text()|//div[@class="body-content"]/p//text()')
    result = [str(i) for i in texts]
    return(result)
    
#%%
# Collects hyperlinks recursively from the front page 'most active' links, and the links in the main body of the underlying, for 'x' recursions.
# checks for duplicates through set.    
def vraps(num):
    checked = []
    regex = re.compile(r"^http://denstoredanske.dk")
    #get starter
    links = set(starter())
    #itterate over desired recursions
    for k in range(num):
        #clean list of checked links
        listtoget = [z for z in links if z not in checked]
        #get links from unchecked sites and extend the checked list
        for i in listtoget:
            n = getlinks(i)
            n = filter(regex.match, getlinks(i))
            links.update(n)
            checked.append(i)
    return(links)
    
#%%
def forkortelser():
    
    # get raw words from "syntaksis"
    x = "https://syntaksis.dk/forkortelser/"
    page = requests.get(x)
    tree = html.fromstring(page.content)
    shorts_1 = tree.xpath('//td[@class="column-1"]/text()')
    
    #split off and trim by ()
    shorts_1_1 = [x.split("(") for x in shorts_1]
    result_1 = [val.replace(")", "") for sublist in shorts_1_1 for val in sublist]
    
    
    #Get raw words from "grafisk-litteratur"
    x = "http://www.grafisk-litteratur.dk/?id=16"
    page = requests.get(x)
    tree = html.fromstring(page.content)
    shorts_2 = tree.xpath('//div[@class="forkortelse"]/text()')
    
    # split by , and "eller", and flatten the resulting list
    n_1 = [x.split(",") for x in shorts_2]
    n_2 = [x.split("eller") for sublist in n_1 for x in sublist]
    n_3 = [x for sublist in n_2 for x in sublist]
    
    #split the few words with a / without meaning in it, by /
    #THIS SHOULD BE RE-CHECKED WHEN RUN FOR FINAL IMPORT
    result_2 = []
    for index, item in enumerate(n_3):
        if index in {67, 84, 85, 86, 172, 174}:
           x = item.split("/")
           for i in x:
               result_2.append(i)
        else:
            result_2.append(item)
    
    
    #combine data from "syntaksis" and "grafsisk-litteratur"
    result_1.extend(result_2)
    # strip begining and ending whitespace from words
    result = [x.strip(" ") for x in result_1]
    list(set(result))
    return(result)
    

#%%

def treatment(text):
    #joins the text with a space, to avoid sentances without a space after a period.
    a = ' '.join(text)
    # remove space doublicates.
    a.replace("  ", " ")
    #split the text into sentances.
    b = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', a)
    return(b)
    
    
#%%    
# This is where we combine everything
def getter(num):
    #collect links
    links = list(vraps(num))
    text = []
    #itterate through links to get text and collect it in one big list
    for i in links:
        res = gettext(i)
        text.extend(res)
    #apply the treatment function
    result = treatment(text)    
    
    # here we need to split it based on periods, but ignoring 'forkortelser'
    return(result)

#%%
# write it all to a .csv file. REMEMBER, IMPORT WITH UTF-8 ENCODING!!!
def importer(num):
    res = getter(num)
    with open("output.csv", 'wb', encoding = "utf-8") as resultFile:
        wr = csv.writer(resultFile, dialect = "excel")
        wr.writerow(res)
