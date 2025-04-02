from datetime import datetime
from db.tables import Review

def normalize_date(raw_date: str) -> str:
    """Converts 'March 2025' to '2025-03-01'. Returns None if parsing fails."""
    try:
        return datetime.strptime(raw_date.strip(), "%B %Y").strftime("%Y-%m-%d")
    except:
        return None

class ChoiceReviewsParser:
    def __init__(self):
        pass
    def parse_choice_review_block(self,block, property_code: str) -> Review:
        """
        Parses a BeautifulSoup <div class="individual-review"> block.
        """
        # Stars: from the first line (still uses .text because star isn't in a clean tag)
        try:
            stars_text = block.get_text(strip=True)
            stars = float(stars_text.split()[0])
        except:
            stars = None

        # Title
        h5 = block.find("h5")
        title = h5.get_text(strip=True) if h5 else ""

        # Date
        date_div = block.find("div", class_="date")
        date = normalize_date(date_div.get_text(strip=True)) if date_div else None

        # Body
        body_div = block.find("div", class_="review-text")
        body = body_div.get_text(" ", strip=True) if body_div else ""

        age = None
        try:
            age_container = block.find("strong", string="Age:")
            if age_container and age_container.next_sibling:
                age = age_container.next_sibling.get_text(strip=True)
        except:
            age = None

        return Review(
            property_code=property_code,
            stars=stars,
            title=title,
            date=date,
            body=body,
            age=age
        )
