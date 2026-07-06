from sprite_motif_pipeline.runner import seeds_for_batch


def test_seeds_for_batch_are_sequential_with_base_seed():
    assert seeds_for_batch(10, 3) == [10, 11, 12]


def test_seeds_for_batch_random_count():
    assert len(seeds_for_batch(None, 4)) == 4
