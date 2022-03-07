import requests
from bs4 import BeautifulSoup
import re
import pandas as pd


def scrapeSDN(url, fileName):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    numPages = int(soup.find_all(
        "li", attrs={"class": "pageNav-page"})[4].find("a").text)
    baseLink = "https://forums.studentdoctor.net"
    database = []

    for i in range(0, numPages):
        pageLink = url + "page-" + str(i+1)
        page = requests.get(pageLink)
        soup = BeautifulSoup(page.content, "html.parser")
        postTitles = soup.find_all("div", {"class": "structItem-title"})
        threads = []
        for j in range(0, len(postTitles)):
            link = baseLink + postTitles[j].find("a", href=True)['href']
            threads.append(scrapeThread(link))
        database.append(pd.concat(threads))
    final = pd.concat(database)
    final.to_csv(fileName)
    return


def scrapeThread(link):
    thread = requests.get(link)
    soup = BeautifulSoup(thread.content, "html.parser")
    if soup.find_all("li", attrs={"class": "pageNav-page"}):
        numPages = int(soup.find_all(
            "li", attrs={"class": "pageNav-page"})[-1].find("a").text)
    else:
        numPages = 1
    # Scrape all pages
    allPosts = []
    for i in range(numPages):
        pageLink = link + "page-" + str(i+1)
        pageGet = requests.get(pageLink)
        pageSoup = BeautifulSoup(pageGet.content, "html.parser")
        dates = pageSoup.find_all("li", {"class": "u-concealed"})
        users = pageSoup.find_all("h4", {"class": "message-name"})
        postNumbers = list(range(i*50+1, i*50+1+len(users)))
        posts = pageSoup.find_all("div", {"class": "bbWrapper"})
        if (pageSoup.find_all("div", {"id": "js-solutionHighlightBlock"})):
            posts.pop(1)
        threadName = pageSoup.find("h1", {"class": "p-title-value"})

        postList = []
        for j in range(len(posts)):
            entry = []
            entry.append(threadName.text)
            entry.append(postNumbers[j])
            entry.append(users[j].find("a")['data-user-id'])
            entry.append(dates[j].find('time')['data-date-string'])
            entry.append(dates[j].find('time')['data-time-string'])
            if posts[j].find("blockquote"):
                entry.append(posts[j].text.replace(
                    posts[j].find("blockquote").text, ""))
            else:
                entry.append(posts[j].text)
            postList.append(entry)
        allPosts.append(pd.DataFrame(data=postList, columns=[
                        'threadName', 'postNumbers', 'user', 'date', 'time', 'post']))
    return pd.concat(allPosts)


# Scrape Why School Threads
scrapeSDN("https://forums.studentdoctor.net/forums/help-me-decide-x-vs-y-medical-school-2021-2022.1106/", "whySchool.csv")

# Scrape WAMC Threads
scrapeSDN("https://forums.studentdoctor.net/forums/what-are-my-chances-wamc-medical.418/", "wamc.csv")

# Scrape Premed DO Posts
scrapeSDN("https://forums.studentdoctor.net/forums/pre-medical-do.13/",
          "allPostsDO.csv")
