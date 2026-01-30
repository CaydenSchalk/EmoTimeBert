from datasets import load_dataset


def load_empath_dfs(trust_remote_code=True):
    full_empath_data = load_dataset(
        "facebook/empathetic_dialogues",
        trust_remote_code=trust_remote_code,
    )
    empath_data = full_empath_data["train"].to_pandas()
    val_empath_data = full_empath_data["validation"].to_pandas()
    return empath_data, val_empath_data


def load_empath_split(split, trust_remote_code=True):
    full_empath_data = load_dataset(
        "facebook/empathetic_dialogues",
        trust_remote_code=trust_remote_code,
    )
    return full_empath_data[split].to_pandas()


def build_emotion_mapping(empath_data):
    emotion_labels = empath_data["context"].unique().tolist()
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
    return emotion_labels, emotion_to_id


def build_conversations(empath_data, emotion_to_id):
    grouped = empath_data.groupby("conv_id")
    conversations = []

    for _, df_conv in grouped:
        texts = df_conv["utterance"].tolist()
        labels = [emotion_to_id[x] for x in df_conv["context"]]
        timestamps = df_conv["utterance_idx"].tolist()
        speakers = (
            df_conv["speaker_idx"]
            .rank(method="dense")
            .astype(int)
            .sub(1)
            .tolist()
        )
        conversations.append(
            {
                "texts": texts,
                "labels": labels,
                "timestamps": timestamps,
                "speakers": speakers,
            }
        )

    return conversations


def load_empath_conversations(trust_remote_code=True):
    empath_data, val_empath_data = load_empath_dfs(
        trust_remote_code=trust_remote_code
    )
    emotion_labels, emotion_to_id = build_emotion_mapping(empath_data)

    conversations = build_conversations(empath_data, emotion_to_id)
    val_conversations = build_conversations(val_empath_data, emotion_to_id)

    return conversations, val_conversations, emotion_labels, emotion_to_id


def load_empath_test_conversations(emotion_to_id, trust_remote_code=True):
    test_empath_data = load_empath_split("test", trust_remote_code=trust_remote_code)
    return build_conversations(test_empath_data, emotion_to_id)
