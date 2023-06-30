from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import TimeoutException

import pandas as pd
import clipboard

PATH_JERE = "F:\\KULIAH\\Project_Tolie\\chromedriver.exe"
PATH_TOLIE = "C:\\Users\\DevOps\\Documents\\Project Cuan\\Reza Tanjung Thesis\\Scrapper\\chromedriver.exe"
driver = webdriver.Chrome(PATH_TOLIE)

driver.set_window_size(1024, 600)
driver.maximize_window()

driver.get("https://twitter.com/login")
time.sleep(10)

username = driver.find_element(By.XPATH, "//input[@name='text']")
username.send_keys("jigongfiraun")
next_button = driver.find_element(By.XPATH, "//span[contains(text(),'Next')]")
next_button.click()

time.sleep(5)
password = driver.find_element(By.XPATH, "//input[@name='password']")
password.send_keys("Cl100299")
log_in = driver.find_element(By.XPATH, "//span[contains(text(),'Log in')]")
log_in.click()

time.sleep(15)

search = "(to:bankBCA)"
clipboard.copy(search)
search_box = driver.find_element(
    By.XPATH, "//input[@data-testid='SearchBox_Search_Input']"
)
search_box.send_keys(Keys.CONTROL + "v")
search_box.send_keys(Keys.ENTER)

time.sleep(3)

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

wait = WebDriverWait(driver, 10)

tweets = []
emojis = []
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    articles = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
    for article in articles:
        try:
            tweet_element = WebDriverWait(article, 5).until(
                EC.presence_of_element_located(
                    (By.XPATH, ".//div[@data-testid='tweetText']")
                )
            )
            tweet = tweet_element.text
            if tweet not in tweets:
                tweets.append(tweet)
                search_emoji = driver.find_element(
                    By.CSS_SELECTOR,
                    ".r-4qtqp9.r-dflpy8.r-sjv1od.r-zw8f10.r-10akycc.r-h9hxbl",
                )
                emoji = search_emoji.get_attribute("alt")
                emojis.append(emoji)
                print(tweet)
        except TimeoutException:
            continue
        except StaleElementReferenceException:
            continue

    # Scroll to the bottom of the page
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait for the page to load new tweets
    time.sleep(5)  # Perpanjang waktu tunggu menjadi 5 detik (atau sesuai kebutuhan)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(10)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        # If the page height remains the same, there are no more tweets to load
        break
    else:
        last_height = new_height


df = pd.DataFrame({"Tweets": tweets, "Emojis": emojis})

df.to_excel(
    "C:\\Users\\DevOps\\Documents\\Project Cuan\\Reza Tanjung Thesis\\flask-dashboard\\static\\assets\\datatweet.xlsx",
    index=False,
)

df.head()
