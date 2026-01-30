from datasets import load_dataset, VerificationMode

def load_meld_dfs(trust_remote_code=True):
    full_meld_data = load_dataset("./data/meld/", trust_remote_code=trust_remote_code,
                                  verification_mode=VerificationMode.NO_CHECKS)

    meld_data = full_meld_data["train"].to_pandas()
    val_meld_data = full_meld_data["validation"].to_pandas()

    return meld_data, val_meld_data


def load_meld_split(split, trust_remote_code=True):
    full_meld_data = load_dataset("./data/meld/", trust_remote_code=trust_remote_code,
                                  verification_mode=VerificationMode.NO_CHECKS)

    return full_meld_data[split].to_pandas()

def build_conversations(meld_data):
    grouped = meld_data.groupby("dialogue_id")
    conversations = []

    for _, df_conv in grouped:
        texts = df_conv["utterance"].tolist()
        labels = df_conv["emotion"].tolist()
        timestamps = df_conv["start_time"].tolist()
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


def load_meld_conversations(trust_remote_code=True):
    meld_data, val_meld_data = load_meld_dfs(
        trust_remote_code=trust_remote_code
    )

    conversations = build_conversations(meld_data)
    val_conversations = build_conversations(val_meld_data, emotion_to_id)

    return conversations, val_conversations, emotion_labels, emotion_to_id


def load_meld_test_conversations(emotion_to_id, trust_remote_code=True):
    test_meld_data = load_meld_split("test", trust_remote_code=trust_remote_code)
    return build_conversations(test_meld_data, emotion_to_id)
