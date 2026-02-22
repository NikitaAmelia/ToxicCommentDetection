from src.preprocessing import clean_text

def test_clean_text():
    text = "RT @user Hello!!! Visit http://example.com #AI"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert "#" not in cleaned
    assert "@" not in cleaned
    assert "rt" not in cleaned
