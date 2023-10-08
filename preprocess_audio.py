"""
VCTK
    https://datashare.ed.ac.uk/handle/10283/3443
VCTK trim info
    https://github.com/nii-yamagishilab/vctk-silence-labels
    
Warning! This code is not properly debugged.
It is recommended to run it only once for the initial state of the audio file (flac or wav).
If executed repeatedly, consecutive application of "trim" may potentially damage the audio file.

>>> $ pip install librosa==0.9.2 numpy==1.23.5 scipy==1.9.1 tqdm # [option]
>>> $ cd /path/to/the/your/vits2
>>> $ ln -s /path/to/the/VCTK/* DUMMY2/
>>> $ git clone https://github.com/nii-yamagishilab/vctk-silence-labels filelists/vctk-silence-labels
>>> $ python preprocess_audio.py --filelists <~/filelist.txt> --config <~/config.json> --trim <~/info.txt>
"""

import argparse
import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm.auto import tqdm

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filelists",
        nargs="+",
        default=[
            "filelists/vctk_audio_sid_text_test_filelist.txt",
            "filelists/vctk_audio_sid_text_val_filelist.txt",
            "filelists/vctk_audio_sid_text_train_filelist.txt",
        ],
    )
    parser.add_argument("--config", default="configs/vctk_base2.json", type=str)
    parser.add_argument(
        "--trim",
        default="filelists/vctk-silence-labels/vctk-silences.0.92.txt",
        type=str,
    )
    args = parser.parse_args()

    with open(args.trim, "r", encoding="utf8") as f:
        lines = list(filter(lambda x: len(x) > 0, f.read().split("\n")))
        trim_info = {}
        for line in lines:
            line = line.split(" ")
            trim_info[line[0]] = (float(line[1]), float(line[2]))

    hps = utils.get_hparams_from_file(args.config)
    for filelist in args.filelists:
        print("START:", filelist)
        with open(filelist, "r", encoding="utf8") as f:
            lines = list(filter(lambda x: len(x) > 0, f.read().split("\n")))

            for line in tqdm(lines, total=len(lines), desc=filelist):
                src_filename = line.split("|")[0]
                if not os.path.isfile(src_filename):
                    if os.path.isfile(src_filename.replace(".wav", "_mic1.flac")):
                        src_filename = src_filename.replace(".wav", "_mic1.flac")
                    else:
                        continue

                if src_filename.endswith("_mic1.flac"):
                    tgt_filename = src_filename.replace("_mic1.flac", ".wav")
                else:
                    tgt_filename = src_filename

                basename = os.path.splitext(os.path.basename(src_filename))[0].replace(
                    "_mic1", ""
                )
                if trim_info.get(basename) is None:
                    print(
                        f"file info: '{src_filename}' doesn't exist in trim info '{args.trim}'"
                    )
                    continue

                start, end = trim_info[basename][0], trim_info[basename][1]

                # warning: it could be make the file to unacceptable
                y, _ = librosa.core.load(
                    src_filename,
                    sr=hps.data.sampling_rate,
                    mono=True,
                    res_type="scipy",
                    offset=start,
                    duration=end - start,
                )

                # y, _ = librosa.effects.trim(
                #     y=y,
                #     frame_length=4096,
                #     hop_length=256,
                #     top_db=35,
                # )

                if y.shape[-1] < hps.train.segment_size:
                    continue

                y = y * hps.data.max_wav_value
                wavfile.write(
                    filename=tgt_filename,
                    rate=hps.data.sampling_rate,
                    data=y.astype(np.int16),
                )
