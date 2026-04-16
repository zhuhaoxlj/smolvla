from bs4 import BeautifulSoup
import sys

with open("Cannot reproduce SmolVLA results on LIBERO benchmark · Issue #2354 · huggingface_lerobot.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

# Find all comments/posts
comments = soup.find_all("div", class_="edit-comment-hide")
for i, c in enumerate(comments):
    text = c.get_text(separator="\n", strip=True)
    if "object" in text.lower():
        print(f"--- Comment {i} ---")
        lines = text.split("\n")
        # Print lines around "object"
        for j, line in enumerate(lines):
            if "object" in line.lower():
                start = max(0, j - 3)
                end = min(len(lines), j + 3)
                print("\n".join(lines[start:end]))
                print("-" * 20)
