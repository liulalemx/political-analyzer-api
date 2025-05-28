# mastodon_client.py
import os
from mastodon import Mastodon
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# Setup Mastodon client
mastodon = Mastodon(
    client_id = os.getenv("MASTODON_CLIENT_ID"),
    client_secret =os.getenv("MASTODON_CLIENT_SECRET"),
    access_token = os.getenv("MASTODON_ACCESS_TOKEN"),
    api_base_url ="https://mastodon.social"
)

def get_user_info_and_posts(username: str, limit=50):
    """Fetch profile + posts of a Mastodon user"""
    try:
        # Search for the user
        acct = mastodon.account_search(username, limit=1)
        if not acct:
            return {"error": "User not found"}

        acct_info = acct[0]
        acct_id = acct_info['id']

        # Fetch recent posts
        statuses = mastodon.account_statuses(acct_id, limit=limit)

        def clean_html(content):
            return BeautifulSoup(content, 'html.parser').get_text()

        posts = [
            {
                "id": s['id'],
                "content": clean_html(s['content']),
                "created_at": s['created_at'],
                "favourites_count": s['favourites_count'],
                "reblogs_count": s['reblogs_count'],
                "replies_count": s['replies_count'],
            }
            for s in statuses
        ]

        # Build user profile info
        user_info = {
            "id": acct_info['id'],
            "username": acct_info['username'],  # e.g. "liul"
            "acct": acct_info['acct'],          # e.g. "liul@mastodon.social"
            "display_name": acct_info['display_name'],
            "avatar": acct_info['avatar_static'],
            "bio": BeautifulSoup(acct_info['note'], 'html.parser').get_text(),
            "created_at": acct_info['created_at'].isoformat() if acct_info['created_at'] else None,
            "followers_count": acct_info['followers_count'],
            "following_count": acct_info['following_count'],
            "statuses_count": acct_info['statuses_count'],
            "posts": posts
        }

        return user_info

    except Exception as e:
        print(f"Error fetching Mastodon data: {e}")
        return {"error": str(e)}
