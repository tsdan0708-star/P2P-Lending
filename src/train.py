from data_pipeline import P2PDataPipeline
from models import build_expert_system_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from config import NUM_COLS, CAT_COLS, USER_COL, MAX_SEQ_LEN, DATA_PATH, BATCH_SIZE, EPOCHS

# 1. Tải dữ liệu
df = pd.read_csv(DATA_PATH)
train_df = df[df['LoanDate'] < '2019-01-01']
test_df = df[df['LoanDate'] >= '2019-01-01']

# 2. Khởi tạo Pipeline
pipeline = P2PDataPipeline(num_features=NUM_COLS, cat_features=CAT_COLS, user_col=USER_COL, max_seq_len=MAX_SEQ_LEN)

# Fit trên Train, Transform trên Test (Chống Data Leakage)
train_scaled, feature_cols = pipeline.fit_transform(train_df)
test_scaled, _ = pipeline.transform(test_df) # Cần thêm hàm transform trong class

# Tạo tf.data.Dataset
train_dataset = pipeline.create_tf_dataset(train_scaled, feature_cols, batch_size=BATCH_SIZE)
test_dataset = pipeline.create_tf_dataset(test_scaled, feature_cols, batch_size=BATCH_SIZE)

# 3. Khởi tạo & Huấn luyện Model
model = build_expert_system_model(seq_len=MAX_SEQ_LEN, num_features=len(feature_cols))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

callbacks = [
    EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)
