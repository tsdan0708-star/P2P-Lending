import tensorflow as tf
from tensorflow.keras import layers, Model

def build_expert_system_model(seq_len, num_features, use_transformer=True, use_lstm=True):
    """
    Hệ thống chuyên gia kết hợp đa phương thức (Multi-modal Fusion)
    """
    seq_input = layers.Input(shape=(seq_len, num_features), name="Sequential_Input")
    tab_input = layers.Input(shape=(num_features,), name="Tabular_Input")
    
    fusion_layers = []
    
    # 1. Luồng Transformer (Phát hiện chu kỳ dài hạn)
    if use_transformer:
        trans_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(seq_input, seq_input)
        trans_out = layers.LayerNormalization()(layers.Add()([seq_input, trans_out]))
        trans_out = layers.GlobalAveragePooling1D()(trans_out)
        trans_out = layers.Dense(64, activation='relu')(trans_out)
        fusion_layers.append(trans_out)
        
    # 2. Luồng LSTM (Phát hiện phụ thuộc ngắn hạn)
    if use_lstm:
        masked_seq = layers.Masking(mask_value=-999.0)(seq_input)
        lstm_out = layers.LSTM(128, return_sequences=False)(masked_seq)
        lstm_out = layers.Dropout(0.3)(layers.Dense(64, activation='relu')(lstm_out))
        fusion_layers.append(lstm_out)

    # 3. Luồng Tabular (Hồ sơ người vay tĩnh)
    mlp_out = layers.Dropout(0.3)(layers.Dense(128, activation='relu')(tab_input))
    mlp_out = layers.Dense(64, activation='relu')(mlp_out)
    fusion_layers.append(mlp_out)

    # 4. Attention-based Feature Fusion (Cơ chế kết hợp có trọng số)
    if len(fusion_layers) > 1:
        # Tính toán trọng số cho từng luồng (Learnable weights)
        weighted_layers = []
        for layer in fusion_layers:
            weight = layers.Dense(1, activation='sigmoid')(layer)
            weighted_layers.append(layers.Multiply()([weight, layer]))
        
        merged = layers.Add()(weighted_layers)
    else:
        merged = fusion_layers[0]

    # Phân loại rủi ro (Default Risk)
    output = layers.Dense(1, activation='sigmoid', name="Default_Probability")(merged)

    return Model(inputs=[seq_input, tab_input], outputs=output, name="ESWA_Fusion_Expert_System")
