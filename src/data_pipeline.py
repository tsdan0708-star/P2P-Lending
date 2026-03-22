import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer

class P2PDataPipeline:
    def __init__(self, num_features, cat_features, user_col='UserName', max_seq_len=6):
        self.num_features = num_features
        self.cat_features = cat_features
        self.user_col = user_col
        self.max_seq_len = max_seq_len
        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self):
        """Xây dựng Pipeline chuẩn hóa Tabular Data"""
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer([
            ('num', num_pipeline, self.num_features),
            ('cat', cat_pipeline, self.cat_features)
        ], remainder='passthrough')

    def fit_transform(self, df):
        """Fit tiền xử lý và trả về DataFrame đã scale"""
        features = df[self.num_features + self.cat_features]
        scaled_features = self.preprocessor.fit_transform(features)
        
        # Lấy lại tên cột sau khi OneHotEncode
        cat_cols = self.preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(self.cat_features)
        all_cols = self.num_features + list(cat_cols)
        
        df_scaled = pd.DataFrame(scaled_features, columns=all_cols, index=df.index)
        df_scaled[self.user_col] = df[self.user_col]
        return df_scaled, all_cols
        
    def transform(self, df):
        """Transform dựa trên pipeline đã fit, trả về DataFrame đã scale cho test data"""
        features = df[self.num_features + self.cat_features]
        scaled_features = self.preprocessor.transform(features)
        
        cat_cols = self.preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(self.cat_features)
        all_cols = self.num_features + list(cat_cols)
        
        df_scaled = pd.DataFrame(scaled_features, columns=all_cols, index=df.index)
        df_scaled[self.user_col] = df[self.user_col]
        return df_scaled, all_cols

    def create_tf_dataset(self, df_scaled, feature_cols, target_col='Default', batch_size=32):
        """Tạo tf.data.Dataset tối ưu hóa bộ nhớ cho chuỗi thời gian"""
        sequences = []
        tabulars = []
        labels = []

        grouped = df_scaled.groupby(self.user_col, sort=False)
        
        for _, group in grouped:
            features = group[feature_cols].values.astype(np.float32)
            label = group[target_col].values[-1] # Lấy nhãn của giao dịch cuối cùng
            
            # Tạo chuỗi tuần tự (Sequential)
            for i in range(1, len(features) + 1):
                seq = features[:i]
                # Pad sequence trực tiếp
                if len(seq) < self.max_seq_len:
                    pad_width = self.max_seq_len - len(seq)
                    seq = np.pad(seq, ((0, pad_width), (0, 0)), constant_values=-999.0)
                else:
                    seq = seq[-self.max_seq_len:]
                
                sequences.append(seq)
                tabulars.append(features[i-1]) # Lấy dòng hiện tại làm Tabular
                labels.append(label)

        # Chuyển thành tf.data.Dataset để tăng tốc độ huấn luyện trên GPU
        dataset = tf.data.Dataset.from_tensor_slices(
            ({'Sequential_Input': sequences, 'Tabular_Input': tabulars}, labels)
        )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
