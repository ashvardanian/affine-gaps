from random import choice, getrandbits, randint
from typing import Iterable, Tuple
import string
import re

import pytest
import numpy as np
import stringzilla as sz
from Bio import Align
from Bio.Align import substitution_matrices

from affine_gaps import needleman_wunsch_gotoh_alignment, needleman_wunsch_gotoh_score


def replace_single_dashes(text, replacement):
    within_line = r"[^-](-)[^-]"
    before_line = r"^(-)[^-]"
    after_line = r"[^-](-)$"
    result = re.sub(within_line, replacement, text)
    result = re.sub(before_line, replacement, text)
    result = re.sub(after_line, replacement, text)
    return result


@pytest.mark.parametrize("min_length", [3, 7])
@pytest.mark.parametrize("max_length", [7, 15])
def test_against_levenshtein(min_length, max_length):

    alphabet = string.ascii_lowercase[:3]
    str1 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))
    str2 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))

    distance = sz.edit_distance(str1, str2)
    aligned1, aligned2, aligned_score = needleman_wunsch_gotoh_alignment(
        str1,
        str2,
        substitution_alphabet=alphabet,
        gap_opening=-1,
        gap_extension=-1,
        match=0,
        mismatch=-1,
    )
    only_score = needleman_wunsch_gotoh_score(
        str1,
        str2,
        substitution_alphabet=alphabet,
        gap_opening=-1,
        gap_extension=-1,
        match=0,
        mismatch=-1,
    )
    assert distance == -aligned_score, f"Wrong alignment: {aligned1} & {aligned2}"
    assert distance == -only_score, f"Wrong alignment: {aligned1} & {aligned2}"


@pytest.mark.parametrize("min_length", [3, 7])
@pytest.mark.parametrize("max_length", [7, 15])
@pytest.mark.parametrize("match_score", [1, 2, 3])
@pytest.mark.parametrize("mismatch_score", [-4, -2])
@pytest.mark.parametrize("gap_opening", [-5, -1])
def test_gap_expansions(
    min_length: int,
    max_length: int,
    match_score: int,
    mismatch_score: int,
    gap_opening: int,
):

    alphabet = string.ascii_lowercase[:3]
    str1 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))
    str2 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))

    aligned1, aligned2, score = needleman_wunsch_gotoh_alignment(
        str1,
        str2,
        substitution_alphabet=alphabet,
        gap_opening=gap_opening,
        gap_extension=0,
        match=match_score,
        mismatch=mismatch_score,
    )

    # If there is a gap in any of the strings, we can expand that gap
    # infinitely, and if the `gap_extension` cost is set to zero, no
    # penalty will be incurred.
    present_gaps = aligned1.count("-") + aligned2.count("-")
    if present_gaps == 0:
        return

    # Let's now validate the baseline for our air-gapped strings
    air_gapped1 = replace_single_dashes(aligned1, " ")
    air_gapped2 = replace_single_dashes(aligned2, " ")
    aig_gapped_score = needleman_wunsch_gotoh_alignment(
        air_gapped1,
        air_gapped2,
        substitution_alphabet=alphabet + " ",
        gap_opening=gap_opening,
        gap_extension=0,
        match=match_score,
        mismatch=mismatch_score,
    )

    # Keep growing those gaps and make sure the score remains the same
    for gap_width in range(2, 5):
        wide_gapped1 = replace_single_dashes(aligned1, " " * gap_width)
        wide_gapped2 = replace_single_dashes(aligned2, " " * gap_width)
        wide_gapped_score = needleman_wunsch_gotoh_alignment(
            wide_gapped1,
            wide_gapped2,
            substitution_alphabet=alphabet + " ",
            gap_opening=gap_opening,
            gap_extension=0,
            match=match_score,
            mismatch=mismatch_score,
        )
        assert wide_gapped_score == aig_gapped_score, f"Gap width: {gap_width}"


def biopy_alignment_score(
    a: str,
    b: str,
    open_gap_score: int,
    extend_gap_score: int,
) -> Tuple[int, str, Iterable[Iterable[int]]]:

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
