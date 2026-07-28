from text_utils import clean_text


def test_clean_text_normalizes_social_content():
    result = clean_text(
        "¡¡¡BITCOIN!!! @usuario https://example.com #Bullish $BTC 1234"
    )

    assert result == "bitcoin!! bullish"


def test_clean_text_preserves_spanish_characters():
    assert clean_text("¡Adopción en España!") == "adopción en españa!"
