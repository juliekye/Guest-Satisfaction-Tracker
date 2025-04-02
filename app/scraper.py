import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from datetime import datetime
import time

class ChoiceReviewsScraper:
    def __init__(self, url: str):
        self.url = url
        self.driver = None

    def scroll_reviews_modal(self, driver, scroll_container):
        last_count = -1
        stable_loops = 0

        while True:
            review_blocks = driver.find_elements(By.CLASS_NAME, "individual-review")
            current_count = len(review_blocks)

            if current_count == last_count:
                stable_loops += 1
            else:
                stable_loops = 0
                last_count = current_count

            if stable_loops >= 3:
                print("✅ Done scrolling — no more new reviews")
                break

            driver.execute_script("arguments[0].scrollTop += 500;", scroll_container)
            time.sleep(0.3)
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", scroll_container)
            time.sleep(1.5)
            
    def get_all_review_blocks(self, url: str):
        # Set up undetected-chromedriver
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")

        driver = uc.Chrome(options=options)

        driver.get(url)
        time.sleep(3)  # Let page JS load

        wait = WebDriverWait(driver, 10)

        # Reject cookies
        try:
            reject_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-track-id='RejectAllCookies-Banner-BTN']")))
            reject_button.click()
            time.sleep(2)
            print("✅ Rejected cookies")
        except:
            pass
            print("⚠️ No cookie banner found")

        # Scroll to make the reviews tab visible
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)

        # Click Reviews tab
        reviews_link = wait.until(EC.presence_of_element_located((By.ID, "NavBarReviews")))
        driver.execute_script("arguments[0].scrollIntoView(true);", reviews_link)
        time.sleep(1)
        reviews_link = driver.find_element(By.ID, "NavBarReviews")
        reviews_link.click()
        print("✅ Clicked Reviews tab")

        # Click 'Show all reviews'
        show_all_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.reviews-button")))
        try:
            show_all_btn.click()
            print("✅ Clicked 'Show all reviews' button")
        except Exception as e:
            print(f"⚠️ Click intercepted, using JavaScript click: {e}")
            driver.execute_script("arguments[0].click();", show_all_btn)
        print("✅ Clicked 'Show all reviews' button")
        time.sleep(2)

        # Wait for **modal window** to load
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "modal-window")))
        time.sleep(2)

        # Find the **scrollable modal container** (inside the modal window)
        modal_window = driver.find_element(By.CLASS_NAME, "modal-window")  # The full modal
        scroll_container = modal_window.find_element(By.CLASS_NAME, "modal-window-inner")  # The scrolling div
        self.scroll_reviews_modal(driver, scroll_container)

        html = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html, "html.parser")
        review_blocks = soup.find_all("div", class_="individual-review")

        print(f"✅ Total reviews parsed: {len(review_blocks)}")
        for i in range(20):
            print(f"\nReview {i}:\n{review_blocks[i].text.strip()}")
        return review_blocks, driver


class TripAdvisorScraper:
    def __init__(self, url: str):
        self.url = url
        self.driver = None

    def scrape_property(self, property_code):
        reviews = []
        offset = 0

        while True:
            url = self._build_url(property_code, offset)
            page = self._get_page(url)
            blocks = self._extract_review_blocks(page)

            if not blocks:
                break

            for block in blocks:
                review = self.parser.parse_review_block(block, property_code)
                reviews.append(review)

            offset += len(blocks)

        return reviews
