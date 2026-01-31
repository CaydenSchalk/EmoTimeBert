from datasets import load_dataset, VerificationMode
import numpy as np

emotion_id_to_label = np.array([
    "neutral neutral",
    "surprise positive",
    "fear negative",
    "surprise negative",
    "sadness negative",
    "joy positive",
    "disgust negative",
    "anger negative",
], dtype=object)

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
    conversations = []

    for idx, df_conv in meld_data.iterrows():
        texts = df_conv["utterance"].tolist()
        labels = df_conv["emotion"].tolist()
        timestamps = df_conv["start_time"].tolist()
        speakers = df_conv["speaker"].tolist()
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
    val_conversations = build_conversations(val_meld_data)
    emotion_classifiers = meld_data["emotion"].explode().unique() # get the unique ints across all rows
    emotion_labels = emotion_id_to_label[emotion_classifiers.astype(int)]

    return conversations, val_conversations, emotion_classifiers, emotion_labels


def load_meld_test_conversations(trust_remote_code=True):
    test_meld_data = load_meld_split("test", trust_remote_code=trust_remote_code)
    return build_conversations(test_meld_data)
