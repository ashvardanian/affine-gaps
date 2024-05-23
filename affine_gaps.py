from typing import Tuple, Optional

import numpy as np
import numba as nb

_int32_max = np.iinfo(np.int32).max
_int32_min = np.iinfo(np.int32).min


# By default, we use BLOSUM62 with affine gap penalties
default_substitution_alphabet: str = "ARNDCQEGHILKMFPSTWYVBZX"
default_substitution_matrix = (
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


def _translate_sequence(seq: str, alphabet: str) -> np.ndarray:
    # def map_char(char):
    #     offset = alphabet.find(char)
    #     return offset if offset >= 0 else len(seq) - 1
    return np.array([alphabet.index(char) for char in seq], dtype=np.uint8)


@nb.jit(nopython=True)
def needleman_wunsch_gotoh_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    gap_opening: int,
    gap_extension: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements Gotoh's algorithm for sequence alignment with affine gap penalties.
    Returns values equal or higher than BioPython, which contains an initialization
    error.

    This function aligns two sequences using Gotoh's algorithm, an extension of the
    Needleman-Wunsch algorithm, incorporating affine gap penalties. It calculates
    three matrices: scores, gaps1, and gaps2, to determine the optimal alignment
    score. The alignment itself can also be reconstructed using these matrices.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): A substitution matrix for scoring matches/mismatches.
    gap_opening (int): The penalty for opening a gap.
    gap_extension (int): The penalty for extending a gap.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices for alignment scoring:
        - scores: The primary scoring matrix.
        - gaps1: The matrix for gaps in the first sequence.
        - gaps2: The matrix for gaps in the second sequence.

    Example usage:
    >>> seq1 = np.array([1, 2, 3])  # Example sequence
    >>> seq2 = np.array([3, 2, 1])  # Example sequence
    >>> substitution_matrix = np.array([[...], [...], [...]])  # Example substitution matrix
    >>> gap_opening = 5
    >>> gap_extension = 2
    >>> scores, gaps1, gaps2 = needleman_wunsch_gotoh_kernel(seq1, seq2, substitution_matrix, gap_opening, gap_extension)
    >>> print("Optimal alignment score matrix:\n", scores)

    Notes:
    The basis and recurrence relations for the matrices are as follows:
    Basis:
    - scores[i][0] = gap_opening + (i - 1) * gap_extension
    - scores[0][j] = gap_opening + (j - 1) * gap_extension
    - gaps1[i][0] = gap_opening + (i - 1) * gap_extension
    - gaps2[0][j] = gap_opening + (j - 1) * gap_extension

    Recurrence:
    - gaps1[i][j] = max(scores[i - 1][j] + gap_opening, gaps1[i - 1][j] + gap_extension)
    - gaps2[i][j] = max(scores[i][j - 1] + gap_opening, gaps2[i][j - 1] + gap_extension)
    - match = scores[i - 1][j - 1] + substitution_matrix[(seq1[i - 1], seq2[j - 1])]
    - scores[i][j] = max(match, gaps1[i][j], gaps2[i][j])
    """
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    scores = np.zeros((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    gaps1 = np.zeros((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    gaps2 = np.zeros((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    gaps1[:, 0] = _int32_min
    gaps2[0, :] = _int32_min

    # Initialize the scoring matrix
    for i in range(1, seq1_len + 1):
        scores[i][0] = gap_opening + (i - 1) * gap_extension
    for j in range(1, seq2_len + 1):
        scores[0][j] = gap_opening + (j - 1) * gap_extension

    # Fill the scoring matrix
    for i in range(1, seq1_len + 1):
        for j in range(1, seq2_len + 1):
            gaps1[i][j] = max(
                scores[i - 1][j] + gap_opening,
                gaps1[i - 1][j] + gap_extension,
            )
            gaps2[i][j] = max(
                scores[i][j - 1] + gap_opening,
                gaps2[i][j - 1] + gap_extension,
            )
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            match = scores[i - 1][j - 1] + substitution
            scores[i][j] = max(match, gaps1[i][j], gaps2[i][j])

    return scores, gaps1, gaps2


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
    Aligns two sequences using Gotoh's algorithm with affine gap penalties.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.
    substitution_alphabet (str): The optional alphabet used for the substitution matrix.
    substitution_matrix (np.ndarray): The optional substitution matrix for scoring matches/mismatches.
    gap_opening (int): The penalty for opening a gap.
    gap_extension (int): The penalty for extending a gap.
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
    >>> str1 = "GATTACA"
    >>> str2 = "GCATGCU"
    >>> align1, align2, score = gotoh_alignment(str1, str2)
    >>> print("Alignment 1:", align1)
    >>> print("Alignment 2:", align2)
    >>> print("Score:", score)
    """
    if (match is not None) != (mismatch is not None):
        raise ValueError("Both match and mismatch must be provided.")
    if (match is not None) and (substitution_matrix is not None):
        raise ValueError(
            "Cannot provide both match/mismatch and a substitution matrix."
        )

    if substitution_alphabet is None:
        substitution_alphabet = default_substitution_alphabet
    if substitution_matrix is None:
        if match is None:
            substitution_matrix = default_substitution_matrix
        else:
            n = len(substitution_alphabet)
            substitution_matrix = np.full((n, n), mismatch)
            substitution_matrix[np.diag_indices(n)] = match
    if gap_opening is None:
        gap_opening = default_gap_opening
    if gap_extension is None:
        gap_extension = default_gap_extension

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)
    scores, gaps1, gaps2 = needleman_wunsch_gotoh_kernel(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
    )

    seq1_len = len(seq1)
    seq2_len = len(seq2)
    align1, align2 = "", ""
    i, j = seq1_len, seq2_len
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and scores[i][j]
            == scores[i - 1][j - 1] + substitution_matrix[(seq1[i - 1], seq2[j - 1])]
        ):
            align1 += str1[i - 1]
            align2 += str2[j - 1]
            i -= 1
            j -= 1
        elif i > 0 and scores[i][j] == gaps1[i][j]:
            align1 += str1[i - 1]
            align2 += "-"
            i -= 1
        else:
            align1 += "-"
            align2 += str2[j - 1]
            j -= 1

    return align1[::-1], align2[::-1], int(scores[seq1_len][seq2_len])


@nb.jit(nopython=True)
def needleman_wunsch_gotoh_score_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    gap_opening: int,
    gap_extension: int,
) -> int:
    """
    Compute the score of the optimal alignment of two sequences using the Gotoh algorithm,
    while only consuming a linear amount of memory, storing only 2 rows per matrix.
    Returns values equal or higher than BioPython, which contains an initialization
    error.

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

    current_score_row = np.zeros(seq2_len + 1, dtype=np.int32)
    previous_score_row = np.zeros(seq2_len + 1, dtype=np.int32)
    current_gaps1_row = np.zeros(seq2_len + 1, dtype=np.int32)
    previous_gaps1_row = np.zeros(seq2_len + 1, dtype=np.int32)
    current_gaps2_row = np.zeros(seq2_len + 1, dtype=np.int32)
    previous_gaps2_row = np.zeros(seq2_len + 1, dtype=np.int32)

    # Initialize the two rows of the scoring matrix
    previous_gaps2_row[:] = _int32_min
    for j in range(1, seq2_len + 1):
        previous_score_row[j] = gap_opening + (j - 1) * gap_extension

    for i in range(1, seq1_len + 1):
        current_score_row[0] = gap_opening + (i - 1) * gap_extension
        current_gaps1_row[0] = _int32_min

        for j in range(1, seq2_len + 1):
            current_gaps1_row[j] = max(
                previous_score_row[j] + gap_opening,
                previous_gaps1_row[j] + gap_extension,
            )
            current_gaps2_row[j] = max(
                current_score_row[j - 1] + gap_opening,
                current_gaps2_row[j - 1] + gap_extension,
            )

            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            match = previous_score_row[j - 1] + substitution
            current_score_row[j] = max(
                match,
                current_gaps1_row[j],
                current_gaps2_row[j],
            )

        # Swap rows
        previous_score_row, current_score_row = current_score_row, previous_score_row
        previous_gaps1_row, current_gaps1_row = current_gaps1_row, previous_gaps1_row
        previous_gaps2_row, current_gaps2_row = current_gaps2_row, previous_gaps2_row

    return previous_score_row[-1]


def needleman_wunsch_gotoh_score(
    str1: str,
    str2: str,
    substitution_alphabet: str = default_substitution_alphabet,
    substitution_matrix: np.ndarray = default_substitution_matrix,
    gap_opening: int = default_gap_opening,
    gap_extension: int = default_gap_extension,
) -> int:
    """
    Compute the score of the optimal alignment of two sequences using the Gotoh algorithm,
    while only consuming a linear amount of memory, storing only 2 rows per matrix.
    Returns values equal or higher than BioPython, which contains an initialization
    error.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): The substitution matrix for scoring matches/mismatches.
    gap_opening (int): The penalty for opening a gap.
    gap_extension (int): The penalty for extending a gap.

    Returns:
    int: The alignment score.
    """

    # The inner loop must be the longer one, assuming the latency of calls
    # from Python into the C layer implementation of NumPy, so lets swap
    # the sequences if needed:
    #
    # if (substitution_matrix == substitution_matrix.T).all():
    #     if len(str1) > len(str2):
    #         str1, str2 = str2, str1

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
    args = parser.parse_args()

    align1, align2, score = needleman_wunsch_gotoh_alignment(
        args.seq1,
        args.seq2,
        match=args.match,
        mismatch=args.mismatch,
        gap_opening=args.gap_opening,
        gap_extension=args.gap_extension,
    )

    print("Alignment 1:", align1)
    print("Alignment 2:", align2)
    print("Score:", score)


if __name__ == "__main__":
    main()
