import os
from pennylane import numpy as np


def process_data(batch_data, audio_encoder, text_encoder):
    audio_names, text, text_ids = batch_data
    audio_names = list(audio_names)

    batch_size = len(audio_names)

    feature_dir = "/mnt/d/ethanfolder/qml/features/"
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    audio_feature_dir = os.path.join(feature_dir, "audio")
    text_feature_dir = os.path.join(feature_dir, "text")
    if not os.path.exists(audio_feature_dir):
        os.makedirs(audio_feature_dir)
    if not os.path.exists(text_feature_dir):
        os.makedirs(text_feature_dir)

    # NOTE: audio features
    all_features_exist = True
    for audio_file in audio_names:
        audio_feature_path = os.path.join(audio_feature_dir, f"AF_{os.path.basename(audio_file)}.npy")
        if not os.path.exists(audio_feature_path):
            all_features_exist = False
            break

    if not all_features_exist:
        # Calculate and save audio features if any are missing
        audio_feature = audio_encoder(audio_names)
        audio_feature = audio_feature.view(batch_size, -1, audio_feature.shape[1])
        audio_feature = audio_feature.mean(dim=1)
        audio_feature = audio_feature.detach().cpu().numpy()
        
        for idx, audio_file in enumerate(audio_names):
            audio_feature_path = os.path.join(audio_feature_dir, f"AF_{os.path.basename(audio_file)}.npy")
            with open(audio_feature_path, 'wb') as af_file:
                np.save(af_file, audio_feature[idx])
    else:
        # Load existing audio features
        audio_feats = []
        for audio_file in audio_names:
            audio_feature_path = os.path.join(audio_feature_dir, f"AF_{os.path.basename(audio_file)}.npy")
            with open(audio_feature_path, 'rb') as af_file:
                audio_feats.append(np.load(af_file))
        print(f"Loaded audio features from files")
        audio_feature = np.array(audio_feats)


    # NOTE: text features
    all_text_features_exist = True
    for t_id in text_ids:
        text_feature_path = os.path.join(text_feature_dir, f"TF_{t_id}.npy")
        if not os.path.exists(text_feature_path):
            all_text_features_exist = False
            break

    if not all_text_features_exist:
        text_feature = text_encoder.encode(text)
        
        for idx, t_id in enumerate(text_ids):
            t_f_path = os.path.join(text_feature_dir, f"TF_{t_id}.npy")
            with open(t_f_path, 'wb') as tf_file:
                np.save(tf_file, text_feature[idx])
    else:
        t_feats = []
        for t_id in text_ids:
            t_f_path = os.path.join(text_feature_dir, f"TF_{t_id}.npy")
            with open(t_f_path, 'rb') as tf_file:
                t_feats.append(np.load(tf_file))
        print(f"Loaded text features from files")
        text_feature = np.array(t_feats)
        

    return audio_feature, text_feature
    # print("audio shape", audio_feature.shape) # 128
    # print("text shape", text_feature.shape) # 384