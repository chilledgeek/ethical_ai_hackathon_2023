import streamlit as st

import os
import sys
import librosa
import torch
import numpy as np
import scipy
from tqdm import tqdm

# didn't have time to turb this into a proper module
sys.path.append("./laughter/")
import laugh_segmenter

import configs

from utils import audio_utils, data_loaders, torch_utils
from functools import partial
from moviepy.editor import VideoFileClip


sample_rate = 8000

import os


@st.cache_resource
def load_model(
    model_path="./laughter/checkpoints/in_use/resnet_with_augmentation",
    config="resnet_with_augmentation",
):
    device = torch.device("cpu")
    print(f"Using device {device}")

    # Load the Model
    config_data = configs.CONFIG_MAP[config]
    model = config_data["model"](
        dropout_rate=0.0,
        linear_layer_size=config_data["linear_layer_size"],
        filter_sizes=config_data["filter_sizes"],
    )
    feature_fn = config_data["feature_fn"]
    model.set_device(device)

    if os.path.exists(model_path):
        torch_utils.load_checkpoint(model_path + "/best.pth.tar", model)
        model.eval()
    else:
        raise Exception(f"Model checkpoint not found at {model_path}")

    return model, feature_fn, config_data


def extract_audio_from_video(video_file, audio_output):
    # Load video file
    video = VideoFileClip(video_file)

    # Extract audio
    audio = video.audio

    # Save audio to the specified output file
    audio.write_audiofile(audio_output)

    # Close the clips
    audio.close()
    video.close()


def segment_laughter(
    input_audio_file,
    model,
    feature_fn,
    config_data,
    threshold=0.5,
    min_length=0.2,
    output_dir=None,
):
    device = torch.device("cpu")
    print(f"Using device {device}")

    st.write(os.getcwd())

    # Load the audio file and features
    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=input_audio_file, feature_fn=feature_fn, sr=sample_rate
    )

    collate_fn = partial(
        audio_utils.pad_sequences_with_labels,
        expand_channel_dim=config_data["expand_channel_dim"],
    )

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset,
        num_workers=4,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Make Predictions
    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape) == 0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = audio_utils.get_audio_length(input_audio_file)

    fps = len(probs) / float(file_length)

    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_laughter_instances(
        probs, threshold=threshold, min_length=min_length, fps=fps
    )

    results = []

    if len(instances) > 0:
        full_res_y, full_res_sr = librosa.load(input_audio_file, sr=44100)
        wav_paths = []
        maxv = np.iinfo(np.int16).max

        base_name = os.path.splitext(os.path.basename(input_audio_file))[0]

        if output_dir is None:
            raise Exception("Need to specify an output directory to save audio files")
        else:
            os.system(f"mkdir -p {output_dir}")
            for index, instance in enumerate(instances):
                laughs = laugh_segmenter.cut_laughter_segments(
                    [instance], full_res_y, full_res_sr
                )
                wav_path = os.path.join(output_dir, f"{base_name}_laugh_{index}.wav")
                scipy.io.wavfile.write(
                    wav_path, full_res_sr, (laughs * maxv).astype(np.int16)
                )
                results.append(
                    {
                        "file_name": wav_path,
                        "start": instance[0],
                        "end": instance[1],
                    }
                )

    return results


# Main Streamlit app
# Streamlit Title
st.title("Laughter Detection")
st.divider()

st.write(
    "Please note this is just a simple interface to play with the laughter detection algorthim (https://github.com/jrgillick/laughter-detection) "
)
st.write(
    "All credits to the author. The paper on the algorithm can be found here:  https://www.isca-speech.org/archive/pdfs/interspeech_2021/gillick21_interspeech.pdf"
)
st.write("note: set to run in cpu only for now. But the algorithm can run on GPU.")
st.divider()

# load model
st.write("loading model...")
model, feature_fn, config_data = load_model()
st.write("model load successfull...")

# parameters
st.subheader("Adjust Model Parameters")
threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
min_length = st.slider("Minimum Length", 0.0, 1.0, 0.2)
st.divider()

# User input
st.subheader("Upload File (audio or video)")

# Add video file types to the file uploader
uploaded_file = st.file_uploader(
    "Choose an audio or video file", type=["wav", "mp3", "mp4", "avi", "mov", "mkv"]
)

UPLOAD_AUDIO_DIR = "upload_audio"
if not os.path.exists(UPLOAD_AUDIO_DIR):
    os.makedirs(UPLOAD_AUDIO_DIR)


UPLOAD_VIDEO_DIR = "upload_video"
if not os.path.exists(UPLOAD_VIDEO_DIR):
    os.makedirs(UPLOAD_VIDEO_DIR)


if uploaded_file:
    # Determine if the uploaded file is a video
    if uploaded_file.type in ["video/mp4", "video/avi", "video/mov", "video/mkv"]:
        # Save the uploaded video to the "upload_video" directory
        upload_video_path = os.path.join(UPLOAD_VIDEO_DIR, uploaded_file.name)
        with open(upload_video_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Extract audio from the video
        audio_output_path = os.path.join(UPLOAD_AUDIO_DIR, uploaded_file.name + ".wav")
        extract_audio_from_video(upload_video_path, audio_output_path)
        upload_file_path = audio_output_path
    else:
        # Save the uploaded audio file to the "upload_audio" directory
        upload_file_path = os.path.join(UPLOAD_AUDIO_DIR, uploaded_file.name)
        with open(upload_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

    # Play the uploaded audio file
    st.audio(upload_file_path, format="audio/wav")

    # Button to run detection
    if st.button("Run Detection"):
        with st.spinner("Segmenting laughter..."):
            results = segment_laughter(
                input_audio_file=upload_file_path,
                model=model,
                feature_fn=feature_fn,
                config_data=config_data,
                threshold=threshold,
                min_length=min_length,
                output_dir="./segmented_laughters",
            )

        if results:
            st.write("Laughter segments found:")

            # Create a table with the results
            table_data = []
            for res in results:
                row = {
                    "File": res["file_name"],
                    "Start": res["start"],
                    "End": res["end"],
                    # "Play": st.audio(res["file_name"], format="audio/wav"),
                }
                table_data.append(row)
                st.audio(res["file_name"], format="audio/wav")

            st.table(table_data)
        else:
            st.write("No laughter segments found.")
