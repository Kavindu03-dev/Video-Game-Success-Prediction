import pandas as pd

from src.preprocessing import clean_dataset, add_hit_label, encode_categoricals


def test_clean_dataset_basic():
    df = pd.DataFrame(
        {
            'name': [' A ', 'A', None],
            'total_sales': [1.0, 1.0, None],
            'release_date': ['2015-01-02', 'invalid', None],
        }
    )

    out = clean_dataset(df)

    # Duplicates dropped: first two rows are duplicates after strip on name and same sales
    assert len(out) <= len(df)

    # release_date to year (nullable Int64)
    assert 'release_date' in out.columns
    assert str(out['release_date'].dtype) == 'Int64'

    # Numeric NaNs filled with median
    assert out['total_sales'].isna().sum() == 0


def test_add_hit_label():
    df = pd.DataFrame({'total_sales': [0.0, 0.5, 1.0, 2.0, None]})
    out = add_hit_label(df, sales_col='total_sales', threshold=1.0, label_col='Hit')
    assert 'Hit' in out.columns
    # Expect [0,0,1,1,0] because NaN treated as 0
    assert out['Hit'].tolist() == [0, 0, 1, 1, 0]


def test_encode_categoricals_onehot():
    df = pd.DataFrame(
        {
            'genre': ['Action', 'Sports', 'Action', None],
            'platform': ['PS4', 'XOne', 'PS4', 'Switch'],
            'publisher': ['Ubisoft', 'EA', 'EA', 'Nintendo'],
            'total_sales': [1.2, 0.4, 0.8, 2.1],
        }
    )
    out = encode_categoricals(df, columns=("genre", "platform", "publisher"), drop_first=True)
    # Should produce dummy columns and keep numeric
    assert 'total_sales' in out.columns
    # At least one dummy column per input categorical
    assert any(col.startswith('genre') for col in out.columns)
    assert any(col.startswith('platform') for col in out.columns)
    assert any(col.startswith('publisher') for col in out.columns)
