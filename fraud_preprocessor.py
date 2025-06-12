"""
Professional Fraud Detection Preprocessor
=========================================

Engineered for payment fraud detection with focus on:
- Temporal patterns (velocity, recency, frequency)
- Behavioral deviations (amount patterns, merchant patterns)
- Risk aggregations (card-level, merchant-level statistics)
- Production-ready feature engineering

Author: Fraud Detection Specialist
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FraudPreprocessor:
    """
    Production-grade preprocessor for payment fraud detection.
    
    Key capabilities:
    - Velocity features (transactions per hour/day)
    - Recency features (time since last transaction)
    - Amount deviation features (vs user/merchant history)
    - Risk aggregations with statistical moments
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.encoders = {}
        self.scalers = {}
        self.aggregation_stats = {}
        
        if self.verbose:
            print("🔧 FraudPreprocessor initialized")
    
    def engineer_temporal_features(self, df, datetime_col='timestamp', sort_by_time=True):
        """
        Create sophisticated temporal features for fraud detection.
        
        Features engineered:
        - Hour/day cyclical encoding (sine/cosine)
        - Velocity features (transactions per time window)
        - Recency features (time gaps)
        - Business hours indicators
        """
        df_temp = df.copy()
        
        # Ensure datetime column exists and is parsed
        if datetime_col in df_temp.columns:
            df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col])
            
            if sort_by_time:
                df_temp = df_temp.sort_values(datetime_col)
            
            # Extract base temporal components
            df_temp['hour'] = df_temp[datetime_col].dt.hour
            df_temp['day_of_week'] = df_temp[datetime_col].dt.dayofweek
            df_temp['day_of_month'] = df_temp[datetime_col].dt.day
            df_temp['month'] = df_temp[datetime_col].dt.month
            
            # Cyclical encoding (critical for temporal patterns)
            df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp['hour'] / 24)
            df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp['hour'] / 24)
            df_temp['dow_sin'] = np.sin(2 * np.pi * df_temp['day_of_week'] / 7)
            df_temp['dow_cos'] = np.cos(2 * np.pi * df_temp['day_of_week'] / 7)
            
            # Business context indicators
            df_temp['is_weekend'] = (df_temp['day_of_week'] >= 5).astype(int)
            df_temp['is_business_hours'] = (
                (df_temp['hour'] >= 9) & (df_temp['hour'] <= 17) & 
                (df_temp['day_of_week'] < 5)
            ).astype(int)
            df_temp['is_night'] = (
                (df_temp['hour'] >= 22) | (df_temp['hour'] <= 6)
            ).astype(int)
            
            if self.verbose:
                print(f"✅ Temporal features: {12} features created")
        
        return df_temp
    
    def engineer_velocity_features(self, df, user_col='user_id', datetime_col='timestamp'):
        """
        Create velocity features - key for fraud detection.
        
        Velocity = frequency of transactions in time windows
        Higher velocity often indicates fraudulent behavior.
        """
        if user_col not in df.columns or datetime_col not in df.columns:
            return df
        
        df_vel = df.copy()
        df_vel[datetime_col] = pd.to_datetime(df_vel[datetime_col])
        df_vel = df_vel.sort_values([user_col, datetime_col])
        
        # Time since last transaction (per user)
        df_vel['time_since_last_txn'] = (
            df_vel.groupby(user_col)[datetime_col].diff().dt.total_seconds() / 3600
        )
        
        # Velocity windows (transactions per time period)
        for window_hours in [1, 6, 24, 168]:  # 1h, 6h, 1d, 1w
            window_name = f'txn_count_{window_hours}h'
            
            # Rolling count within time window
            df_vel[window_name] = (
                df_vel.groupby(user_col)[datetime_col]
                .rolling(f'{window_hours}H', on=datetime_col)
                .count()
                .reset_index(0, drop=True)
            )
        
        # Velocity ratios (current vs historical)
        df_vel['velocity_1h_vs_6h'] = df_vel['txn_count_1h'] / (df_vel['txn_count_6h'] + 1)
        df_vel['velocity_6h_vs_24h'] = df_vel['txn_count_6h'] / (df_vel['txn_count_24h'] + 1)
        
        if self.verbose:
            print(f"✅ Velocity features: {7} features created")
        
        return df_vel
    
    def engineer_amount_features(self, df, amount_col='amount', user_col='user_id'):
        """
        Amount-based features - core for fraud detection.
        
        Focus on deviations from normal spending patterns.
        """
        df_amt = df.copy()
        
        if amount_col not in df_amt.columns:
            return df_amt
        
        # Basic amount transformations
        df_amt['amount_log'] = np.log1p(df_amt[amount_col])
        df_amt['amount_sqrt'] = np.sqrt(df_amt[amount_col])
        
        # Amount categorization
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        thresholds = df_amt[amount_col].quantile(percentiles)
        
        df_amt['is_micro_amount'] = (df_amt[amount_col] <= thresholds[0.1]).astype(int)
        df_amt['is_small_amount'] = (df_amt[amount_col] <= thresholds[0.25]).astype(int)
        df_amt['is_large_amount'] = (df_amt[amount_col] >= thresholds[0.75]).astype(int)
        df_amt['is_very_large_amount'] = (df_amt[amount_col] >= thresholds[0.95]).astype(int)
        df_amt['is_extreme_amount'] = (df_amt[amount_col] >= thresholds[0.99]).astype(int)
        
        # Round number indicators (fraud pattern)
        df_amt['is_round_100'] = (df_amt[amount_col] % 100 == 0).astype(int)
        df_amt['is_round_50'] = (df_amt[amount_col] % 50 == 0).astype(int)
        df_amt['amount_last_digit'] = (df_amt[amount_col] % 10).astype(int)
        
        # User-specific amount patterns
        if user_col in df_amt.columns:
            user_amount_stats = df_amt.groupby(user_col)[amount_col].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).add_prefix('user_amount_')
            
            df_amt = df_amt.merge(user_amount_stats, left_on=user_col, right_index=True)
            
            # Deviation from user patterns
            df_amt['amount_vs_user_mean'] = (
                df_amt[amount_col] / (df_amt['user_amount_mean'] + 1)
            )
            df_amt['amount_z_score'] = (
                (df_amt[amount_col] - df_amt['user_amount_mean']) / 
                (df_amt['user_amount_std'] + 1)
            )
            
            # Is this amount unusual for this user?
            df_amt['is_unusual_amount'] = (
                (df_amt['amount_z_score'].abs() > 2) & 
                (df_amt['user_amount_count'] >= 10)
            ).astype(int)
        
        if self.verbose:
            print(f"✅ Amount features: {15} features created")
        
        return df_amt
    
    def engineer_merchant_features(self, df, merchant_col='merchant_id', amount_col='amount'):
        """
        Merchant-based risk features.
        
        Different merchants have different fraud risk profiles.
        """
        if merchant_col not in df.columns:
            return df
        
        df_merch = df.copy()
        
        # Merchant statistics
        merchant_stats = df_merch.groupby(merchant_col).agg({
            amount_col: ['count', 'mean', 'std', 'min', 'max'],
            'is_fraud': ['sum', 'mean'] if 'is_fraud' in df_merch.columns else amount_col: ['count']
        }).round(4)
        
        # Flatten column names
        merchant_stats.columns = [f'merchant_{col[0]}_{col[1]}' for col in merchant_stats.columns]
        
        # Merge back
        df_merch = df_merch.merge(merchant_stats, left_on=merchant_col, right_index=True)
        
        # Risk indicators
        df_merch['merchant_risk_score'] = df_merch.get('merchant_is_fraud_mean', 0)
        df_merch['is_high_risk_merchant'] = (
            df_merch['merchant_risk_score'] > df_merch['merchant_risk_score'].quantile(0.9)
        ).astype(int)
        
        # Amount vs merchant patterns
        if f'merchant_{amount_col}_mean' in df_merch.columns:
            df_merch['amount_vs_merchant_mean'] = (
                df_merch[amount_col] / (df_merch[f'merchant_{amount_col}_mean'] + 1)
            )
        
        if self.verbose:
            print(f"✅ Merchant features: {len(merchant_stats.columns) + 3} features created")
        
        return df_merch
    
    def engineer_card_features(self, df, card_cols=['card_type', 'card_brand']):
        """
        Card-based features and risk indicators.
        """
        df_card = df.copy()
        
        # Card type risk encoding (frequency-based)
        for col in card_cols:
            if col in df_card.columns:
                # Frequency encoding
                freq_map = df_card[col].value_counts().to_dict()
                df_card[f'{col}_frequency'] = df_card[col].map(freq_map)
                
                # Risk encoding (if fraud labels available)
                if 'is_fraud' in df_card.columns:
                    risk_map = df_card.groupby(col)['is_fraud'].mean().to_dict()
                    df_card[f'{col}_risk_score'] = df_card[col].map(risk_map)
        
        # Card age indicators (if available)
        if 'card_issuer_country' in df_card.columns:
            df_card['is_domestic_card'] = (
                df_card['card_issuer_country'] == df_card.get('transaction_country', 'domestic')
            ).astype(int)
        
        if self.verbose:
            print(f"✅ Card features: {len(card_cols) * 2} features created")
        
        return df_card
    
    def handle_missing_values(self, df, strategy='intelligent'):
        """
        Intelligent missing value handling for fraud detection.
        
        Strategy varies by feature type and business context.
        """
        df_clean = df.copy()
        
        for col in df_clean.columns:
            missing_rate = df_clean[col].isnull().mean()
            
            if missing_rate == 0:
                continue
            
            # High missing rate columns (>50%) - often intentional nulls
            if missing_rate > 0.5:
                df_clean[f'{col}_is_missing'] = df_clean[col].isnull().astype(int)
            
            # Numerical columns
            if df_clean[col].dtype in ['int64', 'float64']:
                if col.endswith('_count') or col.startswith('txn_count'):
                    # Count features: missing = 0
                    df_clean[col] = df_clean[col].fillna(0)
                elif col.endswith('_score') or col.endswith('_risk'):
                    # Risk scores: missing = median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    # Other numerical: median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            # Categorical columns
            else:
                df_clean[col] = df_clean[col].fillna('unknown')
        
        if self.verbose:
            remaining_nulls = df_clean.isnull().sum().sum()
            print(f"✅ Missing values handled. Remaining nulls: {remaining_nulls}")
        
        return df_clean
    
    def encode_categorical_features(self, df, high_cardinality_threshold=50):
        """
        Encode categorical features with appropriate technique per cardinality.
        
        - Low cardinality: One-hot encoding
        - High cardinality: Target encoding + frequency encoding
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()
            
            if unique_count <= high_cardinality_threshold:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
            else:
                # Frequency encoding for high cardinality
                freq_map = df_encoded[col].value_counts().to_dict()
                df_encoded[f'{col}_frequency'] = df_encoded[col].map(freq_map).fillna(0)
                
                # Target encoding if fraud labels available
                if 'is_fraud' in df_encoded.columns:
                    target_map = df_encoded.groupby(col)['is_fraud'].mean().to_dict()
                    df_encoded[f'{col}_target_enc'] = df_encoded[col].map(target_map)
            
            # Drop original categorical column
            df_encoded = df_encoded.drop(columns=[col])
        
        if self.verbose:
            print(f"✅ Categorical encoding: {len(categorical_cols)} columns processed")
        
        return df_encoded
    
    def create_interaction_features(self, df, max_interactions=20):
        """
        Create meaningful feature interactions for fraud detection.
        
        Focus on business-logical interactions.
        """
        df_interact = df.copy()
        interactions_created = 0
        
        # Time × Amount interactions
        if all(col in df_interact.columns for col in ['is_night', 'is_large_amount']):
            df_interact['night_large_amount'] = (
                df_interact['is_night'] * df_interact['is_large_amount']
            )
            interactions_created += 1
        
        if all(col in df_interact.columns for col in ['is_weekend', 'is_very_large_amount']):
            df_interact['weekend_very_large'] = (
                df_interact['is_weekend'] * df_interact['is_very_large_amount']
            )
            interactions_created += 1
        
        # Velocity × Amount interactions
        if all(col in df_interact.columns for col in ['txn_count_1h', 'is_large_amount']):
            df_interact['high_velocity_large_amount'] = (
                (df_interact['txn_count_1h'] > 2) * df_interact['is_large_amount']
            )
            interactions_created += 1
        
        # User behavior × Merchant risk
        if all(col in df_interact.columns for col in ['is_unusual_amount', 'is_high_risk_merchant']):
            df_interact['unusual_amount_risky_merchant'] = (
                df_interact['is_unusual_amount'] * df_interact['is_high_risk_merchant']
            )
            interactions_created += 1
        
        if self.verbose:
            print(f"✅ Interaction features: {interactions_created} features created")
        
        return df_interact
    
    def transform(self, df, fit=True):
        """
        Full feature engineering pipeline.
        
        Args:
            df: Input dataframe
            fit: Whether to fit transformers (True for train, False for test)
        
        Returns:
            Fully engineered feature set ready for ML
        """
        if self.verbose:
            print(f"🚀 Starting feature engineering pipeline")
            print(f"   Input shape: {df.shape}")
        
        # Apply transformations in sequence
        df_processed = df.copy()
        
        # 1. Temporal features
        df_processed = self.engineer_temporal_features(df_processed)
        
        # 2. Velocity features (requires temporal)
        df_processed = self.engineer_velocity_features(df_processed)
        
        # 3. Amount features
        df_processed = self.engineer_amount_features(df_processed)
        
        # 4. Merchant features
        df_processed = self.engineer_merchant_features(df_processed)
        
        # 5. Card features
        df_processed = self.engineer_card_features(df_processed)
        
        # 6. Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # 7. Encode categoricals
        df_processed = self.encode_categorical_features(df_processed)
        
        # 8. Create interactions
        df_processed = self.create_interaction_features(df_processed)
        
        if self.verbose:
            print(f"✅ Feature engineering complete")
            print(f"   Output shape: {df_processed.shape}")
            print(f"   Features added: {df_processed.shape[1] - df.shape[1]}")
        
        return df_processed


def create_realistic_fraud_dataset(n_samples=50000, fraud_rate=0.025):
    """
    Generate realistic payment fraud dataset for testing.
    
    Includes realistic patterns:
    - Higher fraud at night/weekends
    - Velocity-based fraud patterns
    - Amount-based fraud patterns
    - Merchant risk variations
    """
    np.random.seed(42)
    
    # Base transaction data
    data = {
        'transaction_id': range(1, n_samples + 1),
        'user_id': np.random.randint(1, n_samples//10, n_samples),
        'merchant_id': np.random.randint(1, n_samples//50, n_samples),
        'amount': np.random.lognormal(mean=4, sigma=1.5, size=n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        'card_type': np.random.choice(['credit', 'debit'], n_samples, p=[0.7, 0.3]),
        'card_brand': np.random.choice(['visa', 'mastercard', 'amex'], n_samples, p=[0.5, 0.35, 0.15]),
        'merchant_category': np.random.choice([
            'grocery', 'gas', 'restaurant', 'retail', 'online', 'atm'
        ], n_samples, p=[0.25, 0.15, 0.2, 0.15, 0.2, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic fraud patterns
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Fraud probability based on realistic patterns
    fraud_prob = np.full(n_samples, fraud_rate)
    
    # Night transactions (2x risk)
    fraud_prob *= np.where((df['hour'] >= 22) | (df['hour'] <= 6), 2.0, 1.0)
    
    # Weekend (1.5x risk)
    fraud_prob *= np.where(df['day_of_week'] >= 5, 1.5, 1.0)
    
    # High amounts (3x risk for top 5%)
    fraud_prob *= np.where(df['amount'] > df['amount'].quantile(0.95), 3.0, 1.0)
    
    # Online merchants (2x risk)
    fraud_prob *= np.where(df['merchant_category'] == 'online', 2.0, 1.0)
    
    # AMEX cards (0.5x risk - better security)
    fraud_prob *= np.where(df['card_brand'] == 'amex', 0.5, 1.0)
    
    # Generate fraud labels
    df['is_fraud'] = np.random.binomial(1, np.clip(fraud_prob, 0, 1))
    
    # Clean up temp columns
    df = df.drop(['hour', 'day_of_week'], axis=1)
    
    return df


# Usage example and testing
if __name__ == "__main__":
    print("🧪 Testing Professional Fraud Preprocessor")
    print("=" * 60)
    
    # Generate test data
    df = create_realistic_fraud_dataset(10000)
    print(f"Generated dataset: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3%}")
    
    # Initialize and test preprocessor
    preprocessor = FraudPreprocessor(verbose=True)
    df_processed = preprocessor.transform(df)
    
    print(f"\nFinal dataset: {df_processed.shape}")
    print(f"Feature engineering ratio: {df_processed.shape[1] / df.shape[1]:.1f}x")
    
    # Display feature categories
    feature_categories = {
        'Temporal': [c for c in df_processed.columns if any(x in c for x in ['hour', 'day', 'weekend', 'night', 'business'])],
        'Velocity': [c for c in df_processed.columns if 'txn_count' in c or 'velocity' in c or 'time_since' in c],
        'Amount': [c for c in df_processed.columns if 'amount' in c and c != 'amount'],
        'Merchant': [c for c in df_processed.columns if 'merchant' in c],
        'Card': [c for c in df_processed.columns if 'card' in c],
        'Interaction': [c for c in df_processed.columns if any(x in c for x in ['_large_', '_high_', '_risky_'])]
    }
    
    print("\n📊 Feature Engineering Summary:")
    for category, features in feature_categories.items():
        print(f"{category:12}: {len(features):2d} features")
    
    print("\n✅ Preprocessor ready for production use!")
