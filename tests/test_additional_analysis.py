'''Verify mixed-type association helpers.'''

import pandas as pd
from road_severity.additional_analysis import (
    correlation_ratio, cramers_v, mixed_association_matrix,
)


def test_mixed_associations_are_symmetric_and_detect_identical_fields():
    frame = pd.DataFrame({
        'number': [1, 1, 2, 2, 3, 3],
        'number_copy': [1, 1, 2, 2, 3, 3],
        'category': ['a', 'a', 'b', 'b', 'c', 'c'],
        'category_copy': ['a', 'a', 'b', 'b', 'c', 'c'],
    })
    assert cramers_v(frame['category'], frame['category_copy']) == 1.0
    assert correlation_ratio(frame['category'], frame['number']) == 1.0
    matrix = mixed_association_matrix(
        frame, ['number', 'number_copy'], ['category', 'category_copy'])
    pd.testing.assert_frame_equal(matrix, matrix.T)
    assert (matrix.to_numpy().diagonal() == 1).all()
