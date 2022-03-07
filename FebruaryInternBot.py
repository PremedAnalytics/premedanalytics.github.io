from time import sleep
import praw
import re

copypasta = "What the fuck did you just fucking say about me, you little ICU attending bitch? I’ll have you know I graduated top of my class in medical school 8 months ago, and I’m a February intern who's been involved in numerous clinical rotations and critical care subIs. I am trained in writing 8 notes a day and I’m the top documentationist in the entire residency class. You are nothing to me but just another overbearing attending. I will work independently and at my own pace with precision the likes of which has never been seen before on this Earth, mark my fucking words. You think you can get away with saying that shit to me in the ICU team room? Think again, fucker. As we speak I am contacting my peers in the r-residency subreddit so you better prepare for the storm, maggot. The storm that wipes out the pathetic little thing you call your recently graduated attending physician authority. You’re fucking redundant, kid. I can preround anywhere, anytime, and I can place a central line by myself in over seven hundred ways, and that’s just with my bare hands. Not only am I extensively trained in presenting a systems based assessment on rounds, but I have access to the entire arsenal of Netflíx aníme shows and I will use it to its full extent to ignore your miserable ass off the face of the continent, you little shit. If only you could have known what unholy retribution your little “clever” passive-aggressive comment was about to bring down upon you, maybe you would have held your fucking tongue. But you couldn’t, you didn’t, and now you’re paying the price, you goddamn idiot. I will shit fury all over you and you will drown in it. You’re fucking with a February intern, kiddo."
copypasta += '\n\n beep boop, I am a bot created by the nerds over at r/premeddata'
f = open('IDs-responded-to.txt', 'r+')
reddit = praw.Reddit('FebIntern')

subreddits_list = "premed+medicalschool+residency"
subreddits = reddit.subreddit(subreddits_list)

f = open('IDs-responded-to.txt', 'r+')
IDs = f.readlines()

for submission in subreddits.hot(limit=10):
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        if str(comment.id) + '\n' not in IDs:  # Only reply to posts you haven't replied to
            if comment.author.name != 'february_intern_bot':  # Don't respond to self
                if re.search("february intern", comment.body, re.IGNORECASE):
                    comment.reply(copypasta)
                    f.write(comment.id + '\n')
f.close()
