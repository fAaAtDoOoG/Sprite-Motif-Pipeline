import pytest

from sprite_motif_pipeline.config import parse_size


def test_parse_square_size():
    assert parse_size("64", (1, 1)) == (64, 64)


def test_parse_rect_size():
    assert parse_size("1024x768", (1, 1)) == (1024, 768)


def test_parse_invalid_size():
    with pytest.raises(ValueError):
        parse_size("wide", (1, 1))
