"""OfficeQA answer check: scale, sign, percent and list length against gold."""

import pytest

from examples.air_officeqa import answer_check


@pytest.mark.parametrize(
    ("gold", "answer"),
    [
        ("-1299", "-$1,299 million (a net decrease of $1,299 million)"),
        ("-113.42", "-$113.42"),
        ("543 million", "543000000"),
        ("543 million", "543"),
        ("10102000000", "$10.102 billion"),
        ("[10102000000, 4.73]", "[$10,102 million, 4.73%]"),
        ("[10102000000, 4.73]", "[$10,102,000,000 (bids tendered, i.e., $10,102 million), 4.73%]"),
        ("1.5 billion", "1,500 million"),
        ("31.7%", "31.7"),
        ("-18.51%", "−18.51 percent"),
        ("2602", "$2,602 million (calendar 1940, Table 3)"),
        ("1288.33", "$1,288.33 million (FY 1954)"),
        ("[2017, 0.69]", "2017, 0.69"),
        ("March 3, 1977", "March 3, 1977"),
    ],
)
def test_equivalent_answers_are_correct(gold, answer):
    assert answer_check.check(answer, gold) == (True, "")


@pytest.mark.parametrize(
    ("gold", "answer", "reason"),
    [
        ("543 million", "543 billion", answer_check.WRONG),
        ("12.5", "-12.5", answer_check.WRONG),
        ("6.16%", "$6.16 million", answer_check.WRONG),
        ("1, 2", "1, 2, 99", answer_check.EXTRA),
        ("2602", "1,559 or 2,602 million", answer_check.EXTRA),
        ("2602", "total 1,559 (1940: 2,602)", answer_check.NOT_LEADING),
        ("[28, 2444.28]", "28, 2444 (geometric mean, unrounded 2444.28)", answer_check.NOT_LEADING),
        ("March 3, 1977", "April 3, 1977", answer_check.WRONG),
    ],
)
def test_contradicting_or_hedged_answers_are_not(gold, answer, reason):
    assert answer_check.check(answer, gold) == (False, reason)


def test_years_and_numbers_inside_words_are_not_stated_values():
    stated = answer_check.values("FY1984 Series U-1984: 2,602 in 1940")

    assert [(value.number, value.year) for value in stated] == [(1984.0, True), (2602.0, False), (1940.0, True)]
