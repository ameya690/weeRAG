from wee import Tokenizer

def test_roundtrip():
    tok = Tokenizer()
    tok.train(["hello world", "hello there"], vocab_size=200)
    s = "hello world hello"
    ids = tok.encode(s, add_special=True)
    assert tok.decode(ids) == s
