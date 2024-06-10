from typing import Tuple, Optional, Callable

import numpy as np
import numba as nb
from colorama import Fore, Style
from colorama import init as _colorama_init


_colorama_init(autoreset=True)

# Constants for operation codes
MATCH, INSERT, DELETE, SUBSTITUTE = 0, 1, 2, 3

# By default, we use BLOSUM62 with affine gap penalties
default_proteins_alphabet: str = "ARNDCQEGHILKMFPSTWYVBZX"
default_proteins_matrix = (
    np.array(
        [
            # fmt: off
            4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4, 
            -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4, 
            -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4, 
            -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4, 
            0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4, 
            -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4, 
            -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4, 
            0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4, 
            -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4, 
            -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4, 
            -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4, 
            -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4, 
            -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4, 
            -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4, 
            -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4, 
            1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4, 
            0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4, 
            -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4, 
            -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4, 
            0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4, 
            -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4, 
            -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4, 
            0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4, 
            -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1,
            # fmt: on
        ],
        dtype=np.int8,
    ).reshape(24, 24)
    * 5
)
default_gap_opening: int = -4 * 5
default_gap_extension: int = int(-0.2 * 5)


def _reconstruct_alignment(
    changes: np.ndarray,
    seq1: np.ndarray,
    seq2: np.ndarray,
    code_to_char: Callable,
    should_continue: Callable,
) -> Tuple[str, str]:

    align1, align2 = "", ""
    i, j = len(seq1), len(seq2)

    # Backtrack to recover the alignment
    while should_continue(i, j):
        if changes[i, j] == DELETE:
            align1 += code_to_char(seq1[i - 1])
            align2 += "-"
            i -= 1
        elif changes[i, j] == INSERT:
            align1 += "-"
            align2 += code_to_char(seq2[j - 1])
            j -= 1
        else:  # MATCH or SUBSTITUTE
            align1 += code_to_char(seq1[i - 1])
            align2 += code_to_char(seq2[j - 1])
            i -= 1
            j -= 1

    return align1[::-1], align2[::-1]


def _translate_sequence(seq: str, alphabet: str) -> np.ndarray:
    # def map_char(char):
    #     offset = alphabet.find(char)
    #     return offset if offset >= 0 else len(seq) - 1
    assert all(char in alphabet for char in seq), f"Found unknown character in sequence: {seq}"
    return np.array([alphabet.index(char) for char in seq], dtype=np.uint8)


def _validate_gotoh_arguments(
    substitution_alphabet: Optional[str] = None,
    substitution_matrix: Optional[np.ndarray] = None,
    gap_opening: Optional[int] = None,
    gap_extension: Optional[int] = None,
    match: Optional[int] = None,
    mismatch: Optional[int] = None,
) -> Tuple[str, np.ndarray, int, int]:
    """Internal method that validates the arguments for the Needleman-Wunsch algorithm."""
    if (match is not None) != (mismatch is not None):
        raise ValueError("Both match and mismatch must be provided.")
    if (match is not None) and (substitution_matrix is not None):
        raise ValueError("Cannot provide both match/mismatch and a substitution matrix.")

    if substitution_alphabet is None:
        substitution_alphabet = default_proteins_alphabet
    if substitution_matrix is None:
        if match is None:
            substitution_matrix = default_proteins_matrix
        else:
            n = len(substitution_alphabet)
            substitution_matrix = np.full((n, n), mismatch)
            substitution_matrix[np.diag_indices(n)] = match
    if gap_opening is None:
        gap_opening = default_gap_opening
    if gap_extension is None:
        gap_extension = default_gap_extension

    return substitution_alphabet, substitution_matrix, gap_opening, gap_extension


@nb.jit(nopython=True)
def _levenshtein_alignment_kernel(seq1: np.ndarray, seq2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns two sequences using Levenshtein's algorithm.
    The returned distance is the minimum number of single-character edits,
    including insertions, deletions, and substitutions, required to change one
    sequence into the other.

    The kernel has quadratic complexity in space and time, as it stores the
    entire scoring matrix and the operations for each cell, to allow the
    reconstruction of the alignment. Should be called through `levenshtein_alignment`.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The matrices for alignment scoring:
        - scores: The primary scoring matrix.
        - changes: The matrix of enums, with cells equal to MATCH, INSERT, DELETE, or SUBSTITUTE.
    """
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    scores = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    changes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    # Initialize the scoring matrix
    scores[0, 0] = 0
    for i in range(1, seq1_len + 1):
        scores[i, 0] = i
        changes[i, 0] = DELETE
    for j in range(1, seq2_len + 1):
        scores[0, j] = j
        changes[0, j] = INSERT

    # Fill the scoring matrix and track operations
    for i in range(1, seq1_len + 1):
        for j in range(1, seq2_len + 1):

            substitution = int(seq1[i - 1] != seq2[j - 1])

            delete = scores[i - 1, j] + 1
            insert = scores[i, j - 1] + 1
            replace = scores[i - 1, j - 1] + substitution
            score = min(replace, delete, insert)
            scores[i, j] = score

            # Determine the minimum cost operation, preserving the operation kind
            if score == replace:
                changes[i, j] = MATCH if seq1[i - 1] == seq2[j - 1] else SUBSTITUTE
            elif score == delete:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

    return scores, changes


def levenshtein_alignment(str1: str, str2: str) -> Tuple[str, str, int]:
    """
    Aligns two sequences using Levenshtein's algorithm.
    The returned distance is the minimum number of single-character edits,
    including insertions, deletions, and substitutions, required to change one
    sequence into the other.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.

    Returns:
    Tuple[str, str, int]: The optimal alignment of the two sequences and the alignment score.
    """
    seq1 = np.array([ord(c) for c in str1], dtype=np.uint32)
    seq2 = np.array([ord(c) for c in str2], dtype=np.uint32)
    scores, changes = _levenshtein_alignment_kernel(seq1, seq2)
    align1, align2 = _reconstruct_alignment(changes, seq1, seq2, chr, lambda i, j: i > 0 and j > 0)
    return align1, align2, int(scores[-1, -1])


@nb.jit(nopython=True)
def _needleman_wunsch_gotoh_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    gap_opening: int,
    gap_extension: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm.

    The kernel has quadratic complexity in space and time, as it stores the
    entire scoring matrix and the operations for each cell, to allow the
    reconstruction of the alignment. Allocates four equivalent-size matrices
    to store the scores, running cost of gaps in the first sequence, running
    cost of gaps in the second sequence, and the operations for each cell.
    Should be called through `needleman_wunsch_gotoh`.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): A substitution matrix for scoring matches/mismatches.
    gap_opening (int): The penalty for opening a gap.
    gap_extension (int): The penalty for extending a gap.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices for alignment scoring:
        - scores: The primary scoring matrix.
        - changes: The matrix for gaps in the first sequence.

    Example usage:
    >>> seq1 = np.array([1, 2, 3])  # Example sequence
    >>> seq2 = np.array([3, 2, 1])  # Example sequence
    >>> substitution_matrix = np.array([[...], [...], [...]])  # Example substitution matrix
    >>> gap_opening = 5
    >>> gap_extension = 2
    >>> scores, changes = _needleman_wunsch_gotoh_kernel(seq1, seq2, substitution_matrix, gap_opening, gap_extension)
    >>> print("Optimal alignment score matrix:\n", scores)

    Notes:
    The basis and recurrence relations for the matrices are as follows:
    Basis:
    - scores[i, 0] = gap_opening + (i - 1) * gap_extension
    - scores[0, j] = gap_opening + (j - 1) * gap_extension
    - deletes[i, 0] = gap_opening + (i - 1) * gap_extension
    - inserts[0, j] = gap_opening + (j - 1) * gap_extension

    Recurrence:
    - deletes[i, j] = max(scores[i - 1, j] + gap_opening, deletes[i - 1, j] + gap_extension)
    - inserts[i, j] = max(scores[i, j - 1] + gap_opening, inserts[i, j - 1] + gap_extension)
    - match = scores[i - 1, j - 1] + substitution_matrix[(seq1[i - 1], seq2[j - 1])]
    - scores[i, j] = max(match, deletes[i, j], inserts[i, j])
    """
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Initialize the scoring matrix, following the suggestions in the paper.
    # There:
    #
    #   v ~ is gap opening penalty (always non-negative in paper, opposite for us)
    #   u ~ is gap extension penalty (always non-positive in paper, opposite for us)
    #   w(k) = u * k + v
    #
    #   D(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n
    #   P(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with a deletion of at least one residue from A, such that A(m)
    #             is aligned with a gap symbol
    #   Q(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with an insertion of at least one residue from B, such that B(n)
    #             is aligned with a gap symbol
    #
    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    scores = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    deletes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    inserts = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    changes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    scores[0, 0] = 0
    for j in range(1, seq2_len + 1):
        scores[0, j] = gap_opening + (j - 1) * gap_extension
        deletes[0, j] = scores[0, j] + gap_opening + gap_extension
        changes[0, j] = INSERT

    # Fill the scoring matrix
    for i in range(1, seq1_len + 1):
        scores[i, 0] = gap_opening + (i - 1) * gap_extension
        inserts[i, 0] = scores[i, 0] + gap_opening + gap_extension
        changes[i, 0] = DELETE

        for j in range(1, seq2_len + 1):
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            delete = max(
                scores[i - 1, j] + gap_opening,
                deletes[i - 1, j] + gap_extension,
            )
            insert = max(
                scores[i, j - 1] + gap_opening,
                inserts[i, j - 1] + gap_extension,
            )
            replace = scores[i - 1, j - 1] + substitution
            score = max(replace, delete, insert)
            scores[i, j] = score
            deletes[i, j] = delete
            inserts[i, j] = insert

            # Track changes
            if score == replace:
                changes[i, j] = MATCH if seq1[i - 1] == seq2[j - 1] else SUBSTITUTE
            elif score == delete:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

    return scores, changes


def needleman_wunsch_gotoh_alignment(
    str1: str,
    str2: str,
    substitution_alphabet: Optional[str] = None,
    substitution_matrix: Optional[np.ndarray] = None,
    gap_opening: Optional[int] = None,
    gap_extension: Optional[int] = None,
    match: Optional[int] = None,
    mismatch: Optional[int] = None,
) -> Tuple[str, str, int]:
    """
    Aligns two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.
    substitution_alphabet (Optional[str]): The optional alphabet used for the substitution matrix.
    substitution_matrix (Optional[np.ndarray]): The optional substitution matrix for scoring matches/mismatches.
    gap_opening (Optional[int]): The penalty for opening a gap.
    gap_extension (Optional[int]): The penalty for extending a gap.
    match (Optional[int]): The score for a match, to compose the substitution matrix.
    mismatch (Optional[int]): The score for a mismatch, to compose the substitution matrix.

    Returns:
    Tuple[str, str, int]: The optimal alignment of the two sequences and the alignment score.

    Default values:
    >>> substitution_alphabet = "ARNDCQEGHILKMFPSTWYVBZX"
    >>> substitution_matrix = BLOSUM62 * 5
    >>> gap_opening = -20
    >>> gap_extension = -1

    Example usage:
    >>> from affine_gaps import needleman_wunsch_gotoh_alignment
    >>> str1 = "GATTACA"
    >>> str2 = "GCATGCU"
    >>> align1, align2, score = needleman_wunsch_gotoh_alignment(str1, str2)
    >>> print("Alignment 1:", align1)
    >>> print("Alignment 2:", align2)
    >>> print("Score:", score)
    """
    substitution_alphabet, substitution_matrix, gap_opening, gap_extension = _validate_gotoh_arguments(
        substitution_alphabet=substitution_alphabet,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
        match=match,
        mismatch=mismatch,
    )

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)
    scores, changes = _needleman_wunsch_gotoh_kernel(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
    )

    align1, align2 = _reconstruct_alignment(
        changes,
        seq1,
        seq2,
        lambda x: substitution_alphabet[x],
        lambda i, j: i > 0 and j > 0,
    )
    return align1, align2, int(scores[-1, -1])


@nb.jit(nopython=True)
def needleman_wunsch_gotoh_score_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    gap_opening: int,
    gap_extension: int,
) -> int:
    """
    Measures the alignment score of two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm. Uses less memory than the alignment function.

    The kernel has quadratic complexity in time and linear in space, as it stores
    only two rows of each matrix. Allocates four equivalent-size matrices
    to store the scores, running cost of gaps in the first sequence, running
    cost of gaps in the second sequence, and the operations for each cell.
    Should be called through `needleman_wunsch_gotoh_score`.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): The substitution matrix for scoring matches/mismatches.
    gap_opening (int): The penalty for opening a gap.
    gap_extension (int): The penalty for extending a gap.

    Returns:
    int: The alignment score.
    """

    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    old_scores = np.empty(seq2_len + 1, dtype=np.int32)
    new_scores = np.empty(seq2_len + 1, dtype=np.int32)
    old_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    new_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    old_inserts = np.empty(seq2_len + 1, dtype=np.int32)
    new_inserts = np.empty(seq2_len + 1, dtype=np.int32)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    old_scores[0] = 0
    for j in range(1, seq2_len + 1):
        old_scores[j] = gap_opening + (j - 1) * gap_extension
        old_deletes[j] = old_scores[j] + gap_opening + gap_extension

    for i in range(1, seq1_len + 1):
        new_scores[0] = gap_opening + (i - 1) * gap_extension
        new_inserts[0] = new_scores[0] + gap_opening + gap_extension

        for j in range(1, seq2_len + 1):
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            delete = max(old_scores[j] + gap_opening, old_deletes[j] + gap_extension)
            insert = max(new_scores[j - 1] + gap_opening, new_inserts[j - 1] + gap_extension)
            replace = old_scores[j - 1] + substitution
            score = max(replace, delete, insert)
            new_scores[j] = score
            new_deletes[j] = delete
            new_inserts[j] = insert

        # Swap rows
        old_scores, new_scores = new_scores, old_scores
        old_deletes, new_deletes = new_deletes, old_deletes
        old_inserts, new_inserts = new_inserts, old_inserts

    return old_scores[-1]


def needleman_wunsch_gotoh_score(
    str1: str,
    str2: str,
    substitution_alphabet: Optional[str] = None,
    substitution_matrix: Optional[np.ndarray] = None,
    gap_opening: Optional[int] = None,
    gap_extension: Optional[int] = None,
    match: Optional[int] = None,
    mismatch: Optional[int] = None,
) -> int:
    """
    Measures the alignment score of two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm. Uses less memory than the alignment function.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.
    substitution_alphabet (Optional[str]): The optional alphabet used for the substitution matrix.
    substitution_matrix (Optional[np.ndarray]): The optional substitution matrix for scoring matches/mismatches.
    gap_opening (Optional[int]): The penalty for opening a gap.
    gap_extension (Optional[int]): The penalty for extending a gap.
    match (Optional[int]): The score for a match, to compose the substitution matrix.
    mismatch (Optional[int]): The score for a mismatch, to compose the substitution matrix.

    Returns:
    int: The alignment score.

    Default values:
    >>> substitution_alphabet = "ARNDCQEGHILKMFPSTWYVBZX"
    >>> substitution_matrix = BLOSUM62 * 5
    >>> gap_opening = -20
    >>> gap_extension = -1

    Example usage:
    >>> from affine_gaps import needleman_wunsch_gotoh_score
    >>> str1 = "GATTACA"
    >>> str2 = "GCATGCU"
    >>> score = needleman_wunsch_gotoh_score(str1, str2)
    >>> print("Alignment 1:", align1)
    >>> print("Alignment 2:", align2)
    >>> print("Score:", score)
    """

    # The inner loop must be the longer one, assuming the latency of calls
    # from Python into the C layer implementation of NumPy, so lets swap
    # the sequences if needed:
    #
    # if (substitution_matrix == substitution_matrix.T).all():
    #     if len(str1) > len(str2):
    #         str1, str2 = str2, str1
    substitution_alphabet, substitution_matrix, gap_opening, gap_extension = _validate_gotoh_arguments(
        substitution_alphabet=substitution_alphabet,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
        match=match,
        mismatch=mismatch,
    )

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)

    score = needleman_wunsch_gotoh_score_kernel(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
    )

    return int(score)


@nb.jit(nopython=True)
def _smith_waterman_gotoh_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    gap_opening: int,
    gap_extension: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns two sequences using Gotoh's affine gap penalty extensions for the
    Smith-Waterman local alignment algorithm.

    The kernel has quadratic complexity in space and time, as it stores the
    entire scoring matrix and the operations for each cell, to allow the
    reconstruction of the alignment. Allocates four equivalent-size matrices
    to store the scores, running cost of gaps in the first sequence, running
    cost of gaps in the second sequence, and the operations for each cell.
    Should be called through `smith_waterman_gotoh`.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): A substitution matrix for scoring matches/mismatches.
    gap_opening (int): The penalty for opening a gap.
    gap_extension (int): The penalty for extending a gap.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices for alignment scoring:
        - scores: The primary scoring matrix.
        - changes: The matrix for gaps in the first sequence.

    Example usage:
    >>> seq1 = np.array([1, 2, 3])  # Example sequence
    >>> seq2 = np.array([3, 2, 1])  # Example sequence
    >>> substitution_matrix = np.array([[...], [...], [...]])  # Example substitution matrix
    >>> gap_opening = 5
    >>> gap_extension = 2
    >>> scores, changes = _smith_waterman_gotoh_kernel(seq1, seq2, substitution_matrix, gap_opening, gap_extension)
    >>> print("Optimal alignment score matrix:\n", scores)

    Notes:
    The basis and recurrence relations for the matrices are as follows:
    Basis:
    - scores[i, 0] = gap_opening + (i - 1) * gap_extension
    - scores[0, j] = gap_opening + (j - 1) * gap_extension
    - deletes[i, 0] = gap_opening + (i - 1) * gap_extension
    - inserts[0, j] = gap_opening + (j - 1) * gap_extension

    Recurrence:
    - deletes[i, j] = max(scores[i - 1, j] + gap_opening, deletes[i - 1, j] + gap_extension)
    - inserts[i, j] = max(scores[i, j - 1] + gap_opening, inserts[i, j - 1] + gap_extension)
    - match = scores[i - 1, j - 1] + substitution_matrix[(seq1[i - 1], seq2[j - 1])]
    - scores[i, j] = max(match, deletes[i, j], inserts[i, j], 0)
    """
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Initialize the scoring matrix, following the suggestions in the paper.
    # There:
    #
    #   v ~ is gap opening penalty (always non-negative in paper, opposite for us)
    #   u ~ is gap extension penalty (always non-positive in paper, opposite for us)
    #   w(k) = u * k + v
    #
    #   D(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n
    #   P(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with a deletion of at least one residue from A, such that A(m)
    #             is aligned with a gap symbol
    #   Q(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with an insertion of at least one residue from B, such that B(n)
    #             is aligned with a gap symbol
    #
    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    scores = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    deletes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    inserts = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    changes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    scores[0, :] = 0
    deletes[0, :] = gap_opening + gap_extension
    changes[0, :] = INSERT

    # Unlike Needleman-Wunsch, we also track the position of the maximum score.
    max_score = 0
    max_pos = (0, 0)

    # Fill the scoring matrix
    for i in range(1, seq1_len + 1):
        scores[i, 0] = 0
        inserts[i, 0] = gap_opening + gap_extension
        changes[i, 0] = DELETE

        for j in range(1, seq2_len + 1):
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            delete = max(
                scores[i - 1, j] + gap_opening,
                deletes[i - 1, j] + gap_extension,
            )
            insert = max(
                scores[i, j - 1] + gap_opening,
                inserts[i, j - 1] + gap_extension,
            )
            replace = scores[i - 1, j - 1] + substitution
            score = max(replace, delete, insert, 0)
            scores[i, j] = score
            deletes[i, j] = delete
            inserts[i, j] = insert

            # Track changes
            if score == replace:
                changes[i, j] = MATCH if seq1[i - 1] == seq2[j - 1] else SUBSTITUTE
            elif score == delete:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

            # Update max score and position
            if score > max_score:
                max_score = score
                max_pos = (i, j)

    return scores, changes, max_pos


def smith_waterman_gotoh_alignment(
    str1: str,
    str2: str,
    substitution_alphabet: Optional[str] = None,
    substitution_matrix: Optional[np.ndarray] = None,
    gap_opening: Optional[int] = None,
    gap_extension: Optional[int] = None,
    match: Optional[int] = None,
    mismatch: Optional[int] = None,
) -> Tuple[str, str, int]:
    """
    Aligns two sequences using the Smith-Waterman algorithm for local alignment.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.
    substitution_alphabet (Optional[str]): The optional alphabet used for the substitution matrix.
    substitution_matrix (Optional[np.ndarray]): The optional substitution matrix for scoring matches/mismatches.
    gap_opening (Optional[int]): The penalty for opening a gap.
    gap_extension (Optional[int]): The penalty for extending a gap.
    match (Optional[int]): The score for a match, to compose the substitution matrix.
    mismatch (Optional[int]): The score for a mismatch, to compose the substitution matrix.

    Returns:
    Tuple[str, str, int]: The optimal local alignment of the two sequences and the alignment score.
    """
    substitution_alphabet, substitution_matrix, gap_opening, gap_extension = _validate_gotoh_arguments(
        substitution_alphabet=substitution_alphabet,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
        match=match,
        mismatch=mismatch,
    )

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)
    scores, changes, max_pos = _smith_waterman_gotoh_kernel(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
    )

    prefix1, prefix2 = max_pos
    align1, align2 = _reconstruct_alignment(
        changes[: prefix1 + 1, : prefix2 + 1],
        seq1[:prefix1],
        seq2[:prefix2],
        lambda x: substitution_alphabet[x],
        lambda i, j: i > 0 and j > 0 and scores[i, j] > 0,
    )
    return align1, align2, int(scores[prefix1, prefix2])


@nb.jit(nopython=True)
def smith_waterman_gotoh_score_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    gap_opening: int,
    gap_extension: int,
) -> int:
    """
    Computes the Smith-Waterman alignment score using Gotoh's affine gap penalty extensions.
    Uses only two rows per matrix to reduce memory usage.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): The substitution matrix for scoring matches/mismatches.
    gap_opening (int): The penalty for opening a gap.
    gap_extension (int): The penalty for extending a gap.

    Returns:
    int: The highest alignment score.
    """
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    old_scores = np.empty(seq2_len + 1, dtype=np.int32)
    new_scores = np.empty(seq2_len + 1, dtype=np.int32)
    old_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    new_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    old_inserts = np.empty(seq2_len + 1, dtype=np.int32)
    new_inserts = np.empty(seq2_len + 1, dtype=np.int32)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    old_scores[0] = 0
    for j in range(1, seq2_len + 1):
        old_scores[j] = 0
        old_deletes[j] = gap_opening + gap_extension

    max_score = 0

    for i in range(1, seq1_len + 1):
        new_scores[0] = 0
        new_inserts[0] = gap_opening + gap_extension

        for j in range(1, seq2_len + 1):
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            delete = max(old_scores[j] + gap_opening, old_deletes[j] + gap_extension)
            insert = max(new_scores[j - 1] + gap_opening, new_inserts[j - 1] + gap_extension)
            replace = old_scores[j - 1] + substitution
            score = max(replace, delete, insert, 0)
            new_scores[j] = score
            new_deletes[j] = delete
            new_inserts[j] = insert

            if score > max_score:
                max_score = score

        # Swap rows
        old_scores, new_scores = new_scores, old_scores
        old_deletes, new_deletes = new_deletes, old_deletes
        old_inserts, new_inserts = new_inserts, old_inserts

    return max_score


def smith_waterman_gotoh_score(
    str1: str,
    str2: str,
    substitution_alphabet: Optional[str] = None,
    substitution_matrix: Optional[np.ndarray] = None,
    gap_opening: Optional[int] = None,
    gap_extension: Optional[int] = None,
    match: Optional[int] = None,
    mismatch: Optional[int] = None,
) -> int:
    """
    Measures the Smith-Waterman local alignment score using Gotoh's affine gap penalty extensions.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.
    substitution_alphabet (Optional[str]): The optional alphabet used for the substitution matrix.
    substitution_matrix (Optional[np.ndarray]): The optional substitution matrix for scoring matches/mismatches.
    gap_opening (Optional[int]): The penalty for opening a gap.
    gap_extension (Optional[int]): The penalty for extending a gap.
    match (Optional[int]): The score for a match, to compose the substitution matrix.
    mismatch (Optional[int]): The score for a mismatch, to compose the substitution matrix.

    Returns:
    int: The highest alignment score.

    Example usage:
    >>> from affine_gaps import smith_waterman_gotoh_score
    >>> str1 = "GATTACA"
    >>> str2 = "GCATGCU"
    >>> score = smith_waterman_gotoh_score(str1, str2)
    >>> print("Score:", score)
    """

    substitution_alphabet, substitution_matrix, gap_opening, gap_extension = _validate_gotoh_arguments(
        substitution_alphabet=substitution_alphabet,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
        match=match,
        mismatch=mismatch,
    )

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)

    score = smith_waterman_gotoh_score_kernel(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
    )

    return int(score)


def colorize_alignment(align1: str, align2: str) -> Tuple[str, str]:
    """
    Colorizes the alignment strings for visual distinction between matches, mismatches, and gaps.

    Parameters:
    align1 (str): The first aligned sequence.
    align2 (str): The second aligned sequence.

    Returns:
    Tuple[str, str]: The colorized alignments.
    """
    colored_align1 = ""
    colored_align2 = ""

    for a, b in zip(align1, align2):
        if a == b and a != "-":
            colored_align1 += Fore.GREEN + a + Style.RESET_ALL
            colored_align2 += Fore.GREEN + b + Style.RESET_ALL
        elif a == "-" or b == "-":
            colored_align1 += Fore.BLACK + a + Style.RESET_ALL
            colored_align2 += Fore.BLACK + b + Style.RESET_ALL
        else:
            colored_align1 += Fore.RED + a + Style.RESET_ALL
            colored_align2 += Fore.RED + b + Style.RESET_ALL

    return colored_align1, colored_align2


def main():
    # Let's parse the input arguments for alignments in CLI
    import argparse

    parser = argparse.ArgumentParser(description="Affine Gaps alignment CLI utility")
    parser.add_argument(
        "seq1",
        type=str,
        help="The first sequence to be aligned, like insulin (GIVEQCCTSICSLYQLENYCN)",
    )
    parser.add_argument(
        "seq2",
        type=str,
        help="The second sequence to be aligned, like glucagon (HSQGTFTSDYSKYLDSRAEQDFV)",
    )
    parser.add_argument(
        "--match",
        type=int,
        default=None,
        help="The score for a match, to compose the substitution matrix; uses scaled BLOSUM62 by default",
    )
    parser.add_argument(
        "--mismatch",
        type=int,
        default=None,
        help="The score for a mismatch, to compose the substitution matrix; uses scaled BLOSUM62 by default",
    )
    parser.add_argument(
        "--gap-opening",
        type=int,
        default=None,
        help=f"The penalty for opening a gap; uses {default_gap_opening} by default",
    )
    parser.add_argument(
        "--gap-extension",
        type=int,
        default=None,
        help=f"The penalty for extending a gap; uses {default_gap_extension} by default",
    )
    parser.add_argument(
        "--substitution-path",
        type=str,
        default=None,
        help="The path to the substitution alphabet and costs matrix file",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use the Smith-Waterman algorithm for local alignment instead of Needleman-Wunsch",
    )
    args = parser.parse_args()

    aligner = smith_waterman_gotoh_alignment if args.local else needleman_wunsch_gotoh_alignment
    try:
        align1, align2, score = aligner(
            args.seq1,
            args.seq2,
            match=args.match,
            mismatch=args.mismatch,
            gap_opening=args.gap_opening,
            gap_extension=args.gap_extension,
        )
    except Exception as exc:
        print("Error:", exc)
        exit(1)

    colored1, colored2 = colorize_alignment(align1, align2)
    print()
    print("Sequence 1:", args.seq1)
    print("Sequence 2:", args.seq2)
    print()
    print("Alignment 1:", colored1)
    print("Alignment 2:", colored2)
    print("Score:      ", score)


if __name__ == "__main__":
    main()
