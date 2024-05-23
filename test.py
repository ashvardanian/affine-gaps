from random import choice
from typing import Iterable, Tuple

import pytest
import numpy as np


def test_against_levenshtein():

    from affine_gaps import needleman_wunsch_gotoh_alignment
    from Bio import Align

    # The Levenshtein distance is the same as the Needleman-Wunsch algorithm with gap scores -1, -1
    a = "kitten"
    b = "sitting"
    levenshtein_distance = needleman_wunsch_goto


def biopy_alignment_score(
    a: str,
    b: str,
    open_gap_score: int,
    extend_gap_score: int,
) -> Tuple[int, str, Iterable[Iterable[int]]]:
    from Bio import Align
    from Bio.Align import substitution_matrices

    # The aligner picks a different algorithm based on settings.
    # The Needleman-Wunsch algorithm is used if `open_gap_score` and `extend_gap_score` are equal.
    # If they are different, the Gotoh algorithm is used.
    # https://github.com/biopython/biopython/blob/abf5a3b077d2b4af08aed390cbe0af48bdb75f97/Bio/Align/_pairwisealigner.c#L3743C1-L3754C37
    aligner = Align.PairwiseAligner(mode="global")
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    score = int(aligner.score(a, b))

    # Remove the stop codon from the alphabet before using it
    alphabet = str(aligner.substitution_matrix.alphabet).replace("*", "")
    matrix = np.array(aligner.substitution_matrix).astype(np.int8)
    matrix = matrix[: len(alphabet), : len(alphabet)]

    return score, alphabet, matrix


def affinegaps_alignment_score(
    a: str,
    b: str,
    open_gap_score: int,
    extend_gap_score: int,
) -> Tuple[int, str, Iterable[Iterable[int]]]:
    from affine_gaps import needleman_wunsch_gotoh_score
    from affine_gaps import default_substitution_alphabet, default_substitution_matrix

    score = needleman_wunsch_gotoh_score(
        a,
        b,
        gap_opening=open_gap_score,
        gap_extension=extend_gap_score,
    )

    return score, default_substitution_alphabet, default_substitution_matrix


typical_alignment_gap_scores = [
    # (-2, -2),
    (-2, -1),
    (-2, 0),
    (2, 3),
    (12, 13),
    (-10, -1),
]


@pytest.mark.repeat(30)
@pytest.mark.parametrize("first_length", [20, 100])
@pytest.mark.parametrize("second_length", [20, 100])
@pytest.mark.parametrize("gap_scores", typical_alignment_gap_scores)
def test_against_biopython(
    first_length: int,
    second_length: int,
    gap_scores: Tuple[int, int],
):
    open_gap_score, extend_gap_score = gap_scores

    # Make sure we generate different strings each time
    common_aminoacids = "ARNDCQEGHILKMFPSTWYVBZX"
    a = "".join(choice(common_aminoacids) for _ in range(first_length))
    b = "".join(choice(common_aminoacids) for _ in range(second_length))

    biopy_score, _, _ = biopy_alignment_score(
        a,
        b,
        open_gap_score=open_gap_score,
        extend_gap_score=extend_gap_score,
    )
    affinegaps_score, _, _ = affinegaps_alignment_score(
        a,
        b,
        open_gap_score=open_gap_score,
        extend_gap_score=extend_gap_score,
    )

    assert (
        affinegaps_score >= biopy_score
    ), "Affine Gaps alignments should be at least as good as BioPython"
