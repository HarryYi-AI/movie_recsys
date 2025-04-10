# src/model_din.py
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import logging

# 自定义 Attention 层 
class AttentionUnit(layers.Layer):
    def __init__(self, hidden_units, activation='relu', dropout_rate=0.0, l2_reg=0.0, **kwargs):
        super(AttentionUnit, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation # 使用字符串或 tf.keras.activations
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.dnn_layers = []
        self.dropout_layers = []

    def build(self, input_shape):
        # input_shape 是一个列表: [query_shape, keys_shape]
        # query_shape: (batch_size, embedding_dim)
        # keys_shape: (batch_size, seq_len, embedding_dim)
        query_dim = input_shape[0][-1]
        keys_dim = input_shape[1][-1]
        input_dim = query_dim + keys_dim + keys_dim + keys_dim # query, key, query-key, query*key

        # 创建 DNN 层
        for units in self.hidden_units:
            self.dnn_layers.append(layers.Dense(units,
                                         activation=None, # 先不加激活，后面加 PReLU 或其他
                                         kernel_regularizer=regularizers.l2(self.l2_reg)))
            # 可以考虑在这里加 BatchNormalization
            # self.dnn_layers.append(layers.BatchNormalization())
            self.dnn_layers.append(layers.PReLU() if self.activation == 'prelu' else tf.keras.layers.Activation(self.activation))
            if self.dropout_rate > 0:
                self.dropout_layers.append(layers.Dropout(self.dropout_rate))

        # 最终输出权重层
        self.final_dense = layers.Dense(1, activation=None, kernel_regularizer=regularizers.l2(self.l2_reg)) # 输出 attention score

        super(AttentionUnit, self).build(input_shape) # Be sure to call this at the end

    def call(self, inputs, training=None, mask=None):
        # inputs 是一个列表: [query, keys]
        query, keys = inputs # query: (batch, embed_dim), keys: (batch, seq_len, embed_dim)

        # query 扩展以匹配 keys 的序列长度
        query_expanded = tf.expand_dims(query, axis=1) # (batch, 1, embed_dim)
        seq_len = tf.shape(keys)[1]
        query_tiled = tf.tile(query_expanded, [1, seq_len, 1]) # (batch, seq_len, embed_dim)

        # 拼接特征: query, keys, query-keys, query*keys
        din_input = tf.concat([query_tiled, keys, query_tiled - keys, query_tiled * keys], axis=-1)
        # Shape: (batch, seq_len, 4 * embed_dim)

        # 通过 DNN 计算 Attention Score
        dnn_output = din_input
        for i, layer in enumerate(self.dnn_layers):
            dnn_output = layer(dnn_output, training=training)
            # 应用 Dropout (如果存在)
            if self.dropout_rate > 0 and i < len(self.dropout_layers):
                dnn_output = self.dropout_layers[i](dnn_output, training=training)

        # 计算最终分数
        scores = self.final_dense(dnn_output) # (batch, seq_len, 1)

        # 处理 Masking (如果 keys 的 Embedding 层设置了 mask_zero=True)
        if mask is not None and mask[1] is not None: # mask 通常对应 keys 输入
             # mask[1] shape is (batch, seq_len) boolean
             key_mask = tf.expand_dims(tf.cast(mask[1], tf.float32), axis=-1) # (batch, seq_len, 1)
             # 给 padding 位置设置一个很大的负值，使其在 softmax 后接近 0
             scores = scores * key_mask - (1.0 - key_mask) * 1e9
        else:
            # 如果没有显式 mask，并且 keys 中包含 padding (例如 ID 0)，
            # 我们需要手动处理，确保 padding 不影响 softmax
            # 假设 padding ID 对应的 embedding 是 0 向量，或者我们知道 padding ID
            # 这里简单假设不处理，依赖 padding ID 0 的 embedding 不会太大
             pass


        # 应用 Softmax 得到权重
        weights = tf.nn.softmax(scores, axis=1) # (batch, seq_len, 1)

        # 加权求和 keys
        # output = tf.reduce_sum(weights * keys, axis=1) # (batch, embed_dim)
        # 使用 Batch MatMul 更高效
        output = tf.matmul(tf.transpose(weights, perm=[0, 2, 1]), keys) # (batch, 1, seq_len) x (batch, seq_len, embed_dim) -> (batch, 1, embed_dim)
        output = tf.squeeze(output, axis=1) # (batch, embed_dim)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1]) # (batch_size, embedding_dim)

    def get_config(self):
        config = super(AttentionUnit, self).get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config


def build_din_model(num_users, num_items, embedding_dim=8, seq_len=10,
                    dnn_hidden_units=(64, 32), att_hidden_units=(64, 32),
                    dnn_dropout=0.2, att_dropout=0.1, l2_reg=0.001,
                    use_bn=True, activation='relu'):
    """
    构建 DIN 模型 (TensorFlow 2.x Keras Functional API)
    """
    logging.info("Building DIN model...")
    # --- 输入层 ---
    # 使用编码后的 ID，从 0 开始
    user_input = layers.Input(shape=(1,), name='user_input', dtype=tf.int32)
    item_input = layers.Input(shape=(1,), name='item_input', dtype=tf.int32)
    hist_input = layers.Input(shape=(seq_len,), name='hist_input', dtype=tf.int32) # 用户历史行为序列 (编码后ID)

    # --- Embedding 层 ---
    # 注意：input_dim 需要是 类别数 + 1 (如果ID从0开始编码，且包含0)
    # 如果 LabelEncoder 从 0 到 N-1，则 input_dim 是 N
    # 如果 padding ID 为 0，且 0 是有效 ID，则需要处理。
    # 这里假设 LabelEncoder 输出 0 到 N-1，padding ID 为 0 (需要一个专门的 embedding 或 mask)
    # 为了安全，input_dim 设为 类别数 + 1，并使用 mask_zero=True (假设 0 是 padding)
    user_embedding_layer = layers.Embedding(
        input_dim=num_users + 1, # 加 1 考虑 padding ID 0 或确保范围正确
        output_dim=embedding_dim,
        name='user_embedding',
        embeddings_regularizer=regularizers.l2(l2_reg),
        mask_zero=False # User ID 通常不为 0，不需 mask
    )
    item_embedding_layer = layers.Embedding(
        input_dim=num_items + 1, # +1 for padding ID 0
        output_dim=embedding_dim,
        name='item_embedding',
        embeddings_regularizer=regularizers.l2(l2_reg),
        mask_zero=True # IMPORTANT: hist_input uses 0 for padding
    )

    user_embed = layers.Flatten()(user_embedding_layer(user_input)) # (batch, embed_dim)
    item_embed = layers.Flatten()(item_embedding_layer(item_input))   # (batch, embed_dim)
    hist_embed = item_embedding_layer(hist_input)             # (batch, seq_len, embed_dim), mask 会传递

    # --- Attention 机制 ---
    attention_layer = AttentionUnit(hidden_units=att_hidden_units,
                                    activation=activation,
                                    dropout_rate=att_dropout,
                                    l2_reg=l2_reg,
                                    name='attention_unit')
    # Attention 输入: query=目标物品embedding, keys=历史物品embedding
    # mask 会自动从 hist_embed 传递给 AttentionUnit 的 call 方法
    att_output = attention_layer([item_embed, hist_embed]) # (batch, embed_dim)

    # --- DNN 部分 ---
    # 拼接所有特征: 用户 embedding, 物品 embedding, Attention 输出
    dnn_input = layers.Concatenate()([user_embed, item_embed, att_output])

    # DNN 层
    x = dnn_input
    for units in dnn_hidden_units:
        x = layers.Dense(units, activation=None, kernel_regularizer=regularizers.l2(l2_reg))(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        # x = tf.keras.layers.Activation(activation)(x) # 使用标准的 Activation
        x = layers.PReLU()(x) # 或者使用 PReLU
        if dnn_dropout > 0:
            x = layers.Dropout(dnn_dropout)(x)

    # --- 输出层 ---
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    # --- 构建模型 ---
    model = keras.Model(inputs=[user_input, item_input, hist_input], outputs=output)

    # --- 编译模型 ---
    # 使用 Adam 优化器，可以调整学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')]) # 使用 Keras 内置 AUC

    logging.info("DIN model built and compiled successfully.")
    # model.summary() # 打印模型结构

    return model

if __name__ == '__main__':
    # 假设用户和电影数量 (应从数据集中计算获得)
    num_users_example = 6040
    num_items_example = 3706
    seq_len_example = 15

    # 构建模型
    try:
        din_model = build_din_model(
            num_users=num_users_example,
            num_items=num_items_example,
            embedding_dim=16,
            seq_len=seq_len_example,
            dnn_hidden_units=(128, 64),
            att_hidden_units=(64, 32),
            dnn_dropout=0.3,
            att_dropout=0.2,
            l2_reg=0.001
        )
        print("\n--- DIN Model Summary ---")
        din_model.summary()

        # 简单测试输入输出形状 (模拟数据)
        batch_size = 4
        dummy_user = tf.random.uniform((batch_size, 1), minval=1, maxval=num_users_example + 1, dtype=tf.int32)
        dummy_item = tf.random.uniform((batch_size, 1), minval=1, maxval=num_items_example + 1, dtype=tf.int32)
        # 模拟历史序列，包含 padding (0)
        dummy_hist = tf.random.uniform((batch_size, seq_len_example), minval=0, maxval=num_items_example + 1, dtype=tf.int32)

        # 预测
        preds = din_model.predict([dummy_user, dummy_item, dummy_hist])
        print("\n--- Dummy Prediction Output ---")
        print(f"Input shapes: user={dummy_user.shape}, item={dummy_item.shape}, hist={dummy_hist.shape}")
        print(f"Output prediction shape: {preds.shape}")
        print(f"Sample predictions:\n{preds}")

    except Exception as e:
        print(f"Error building or testing DIN model: {e}")
        import traceback
        traceback.print_exc()