from llm_from_scratch.tokenizer import CARD_END, CARD_START, BPETokenizer, CharTokenizer


def test_char_tokenizer_roundtrip(tmp_path):
    dataset_path = tmp_path / "dataset.txt"
    mapping_path = tmp_path / "tokenizer.json"
    text = f"{CARD_START}\nname: test card\n{CARD_END}\n"

    dataset_path.write_text(text)

    tokenizer = CharTokenizer()
    tokenizer.build_mapping(
        input_dataset_path=dataset_path,
        mapping_save_path=mapping_path,
    )

    tokenizer = CharTokenizer(mapping_path=mapping_path)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_tokenizer_roundtrip(tmp_path):
    dataset_path = tmp_path / "dataset.txt"
    mapping_path = tmp_path / "tokenizer.json"
    text = f"{CARD_START}\nname: test card\ndescription: café ● effect\n{CARD_END}\n"

    dataset_path.write_text(text)

    tokenizer = BPETokenizer()
    tokenizer.build_mapping(
        input_dataset_path=dataset_path,
        mapping_save_path=mapping_path,
        vocab_size=512,
    )

    tokenizer = BPETokenizer(mapping_path=mapping_path)

    assert tokenizer.decode(tokenizer.encode(text)) == text
