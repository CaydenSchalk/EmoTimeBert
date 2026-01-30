from datasets import load_dataset

def load_meld_dfs(trust_remote_code=True):
    full_meld_data = load_dataset("./data/meld/", trust_remote_code=trust_remote_code)

    meld_data = full_meld_data["train"].to_pandas()
    val_meld_data = full_meld_data["validation"].to_pandas()

    return meld_data, val_meld_data


def load_meld_split(split, trust_remote_code=True):
    full_meld_data = load_dataset("./data/meld/", trust_remote_code=trust_remote_code)

    return full_meld_data[split].to_pandas()


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
    meld_data, val_meld_data = load_meld_dfs(
        trust_remote_code=trust_remote_code
    )
    emotion_labels, emotion_to_id = build_emotion_mapping(meld_data)

    conversations = build_conversations(meld_data, emotion_to_id)
    val_conversations = build_conversations(val_meld_data, emotion_to_id)

    return conversations, val_conversations, emotion_labels, emotion_to_id


def load_empath_test_conversations(emotion_to_id, trust_remote_code=True):
    test_meld_data = load_meld_split("test", trust_remote_code=trust_remote_code)
    return build_conversations(test_meld_data, emotion_to_id)
