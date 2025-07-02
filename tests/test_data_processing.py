import pytest
import pandas as pd
import numpy as np
from src.data_processing import RFMTransformer, HighRiskLabelGenerator, build_feature_pipeline
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
from sklearn.utils.validation import check_is_fitted

@pytest.fixture
def fixed_now():
    """Single fixed timestamp for all tests"""
    return datetime(2023, 1, 15, 12, 30, 45)  # Fixed reference time with minutes/seconds

@pytest.fixture
def sample_data(fixed_now):
    """Generate synthetic test data with varied timestamps"""
    data = {
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionId': [101, 102, 201, 202, 301],
        'TransactionStartTime': [
            fixed_now - timedelta(days=5, hours=3),    # Jan 10, 09:30:45
            fixed_now - timedelta(days=2, minutes=15), # Jan 13, 12:15:45
            fixed_now - timedelta(days=10, seconds=1), # Jan 5, 12:30:44
            fixed_now - timedelta(hours=18),           # Jan 14, 18:30:45
            fixed_now - timedelta(days=20)             # Dec 26, 12:30:45
        ],
        'Value': [100, 200, 50, 150, 300],
        'ProductCategory': ['A', 'B', 'A', 'C', 'B'],
        'ChannelId': ['Web', 'App', 'App', 'Web', 'App'],
        'PricingStrategy': ['Standard', 'Premium', 'Standard', 'Discount', 'Premium']
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime']) # Ensure datetime type
    return df


def test_rfm_transformer_columns(sample_data, fixed_now):
    """Test all RFM and temporal features are created"""
    transformer = RFMTransformer(snapshot_date=fixed_now)
    transformed = transformer.transform(sample_data.copy()) # Use copy to prevent modifying fixture

    expected_columns = {
        'Recency', 'Frequency', 'MonetarySum',
        'MonetaryMean', 'MonetaryStd',
        'TransactionHour', 'TransactionDay', 'TransactionMonth'
    }
    assert expected_columns.issubset(transformed.columns)
    assert transformed.shape[0] == sample_data.shape[0], "RFMTransformer should maintain the original number of rows."


def test_rfm_calculations(sample_data, fixed_now):
    """Verify RFM calculations for all customer types"""
    transformer = RFMTransformer(snapshot_date=fixed_now)
    transformed = transformer.transform(sample_data.copy())

    # Test Customer 1 (2 transactions)
    cust1_data = sample_data[sample_data['CustomerId'] == 1]
    cust1_transformed = transformed[transformed['CustomerId'] == 1].iloc[0] # Get one row for RFM values
    assert len(transformed[transformed['CustomerId'] == 1]) == 2 # Maintains transaction-level rows

    assert np.allclose(cust1_transformed['MonetarySum'], 300)
    assert np.allclose(cust1_transformed['MonetaryMean'], 150)
    assert np.allclose(cust1_transformed['MonetaryStd'], np.std([100,200], ddof=1))
    assert np.allclose(cust1_transformed['Recency'], (fixed_now - cust1_data['TransactionStartTime'].max()).days)
    assert np.allclose(cust1_transformed['Frequency'], 2)

    # Test Customer 2 (2 transactions with different pattern)
    cust2_data = sample_data[sample_data['CustomerId'] == 2]
    cust2_transformed = transformed[transformed['CustomerId'] == 2].iloc[0]
    assert len(transformed[transformed['CustomerId'] == 2]) == 2

    assert np.allclose(cust2_transformed['MonetarySum'], 200)  # 50 + 150
    assert np.allclose(cust2_transformed['MonetaryMean'], 100)
    assert np.allclose(cust2_transformed['MonetaryStd'], np.std([50,150], ddof=1))
    assert np.allclose(cust2_transformed['Recency'], (fixed_now - cust2_data['TransactionStartTime'].max()).days)
    assert np.allclose(cust2_transformed['Frequency'], 2)

    # Test Customer 3 (single transaction - edge case)
    cust3_data = sample_data[sample_data['CustomerId'] == 3]
    cust3_transformed = transformed[transformed['CustomerId'] == 3].iloc[0]
    assert len(transformed[transformed['CustomerId'] == 3]) == 1

    assert np.allclose(cust3_transformed['MonetarySum'], 300)
    assert np.allclose(cust3_transformed['MonetaryMean'], 300)
    assert np.allclose(cust3_transformed['MonetaryStd'], 0)   # Single transaction
    assert np.allclose(cust3_transformed['Recency'], (fixed_now - cust3_data['TransactionStartTime'].max()).days)
    assert np.allclose(cust3_transformed['Frequency'], 1)


def test_temporal_features(sample_data, fixed_now):
    """Verify precise temporal feature extraction with varied times"""
    transformer = RFMTransformer(snapshot_date=fixed_now)
    transformed = transformer.transform(sample_data.copy())

    # Expected values based on fixed_now (Jan 15, 12:30:45) and timedelta in sample_data
    expected_values = [
        (9, 10, 1),    # Jan 10, 09:30:45 (fixed_now - timedelta(days=5, hours=3))
        (12, 13, 1),   # Jan 13, 12:15:45 (fixed_now - timedelta(days=2, minutes=15))
        (12, 5, 1),    # Jan 5, 12:30:44 (fixed_now - timedelta(days=10, seconds=1))
        (18, 14, 1),   # Jan 14, 18:30:45 (fixed_now - timedelta(hours=18))
        (12, 26, 12)   # Dec 26, 12:30:45 (fixed_now - timedelta(days=20) -> previous year)
    ]

    for idx, (exp_hour, exp_day, exp_month) in enumerate(expected_values):
        assert transformed.loc[idx, 'TransactionHour'] == exp_hour, f"Row {idx}: Hour mismatch. Expected {exp_hour}, Got {transformed.loc[idx, 'TransactionHour']}"
        assert transformed.loc[idx, 'TransactionDay'] == exp_day, f"Row {idx}: Day mismatch. Expected {exp_day}, Got {transformed.loc[idx, 'TransactionDay']}"
        assert transformed.loc[idx, 'TransactionMonth'] == exp_month, f"Row {idx}: Month mismatch. Expected {exp_month}, Got {transformed.loc[idx, 'TransactionMonth']}"


def test_high_risk_labeler(sample_data, fixed_now):
    """Test reproducible risk label generation"""
    rfm_transformer = RFMTransformer(snapshot_date=fixed_now)
    rfm_data = rfm_transformer.transform(sample_data.copy())

    labeler = HighRiskLabelGenerator(n_clusters=2, random_state=42)
    labeled_data = labeler.fit_transform(rfm_data)

    check_is_fitted(labeler, 'kmeans')
    assert hasattr(labeler, 'high_risk_cluster_')
    assert 'is_high_risk' in labeled_data.columns

    # Verify we have both risk classes (0 and 1)
    unique_labels = set(labeled_data['is_high_risk'].unique())
    assert unique_labels == {0, 1}

    # Verify a balanced mix (inclusive on the lower bound for exact 0.2 mean)
    assert 0.2 <= labeled_data['is_high_risk'].mean() < 0.8


def test_full_pipeline_output(sample_data):
    """Test complete pipeline output with exact feature count"""
    pipeline = build_feature_pipeline()
    processed = pipeline.fit_transform(sample_data.copy())
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (5, 16)  # 5 samples × (5 RFM + 3 temporal + 8 encoded)


def test_pipeline_with_missing_data(fixed_now):
    """Test pipeline handles missing values"""
    data = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionId': [101, 102, 201], # Added missing TransactionId
        'TransactionStartTime': [fixed_now] * 3,
        'Value': [100, np.nan, 200],
        'ProductCategory': ['A', np.nan, 'B'], # <--- CORRECTED: Changed None to np.nan
        'ChannelId': ['Web', 'App', np.nan],    # <--- CORRECTED: Changed None to np.nan
        'PricingStrategy': ['Standard', 'Premium', 'Standard']
    })
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    pipeline = build_feature_pipeline()
    processed = pipeline.fit_transform(data)
    assert not np.isnan(processed).any()


def test_pipeline_structure():
    """Verify pipeline composition"""
    pipeline = build_feature_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == [
        'rfm_features',
        'risk_labels',
        'preprocessing'
    ]
