import argparse

import numpy as np
import onnxruntime
import torch
from scipy.io.wavfile import write

import commons
import utils
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (.onnx)")
    parser.add_argument(
        "--config-path", required=True, help="Path to model config (.json)"
    )
    parser.add_argument(
        "--output-wav-path", required=True, help="Path to write WAV file"
    )
    parser.add_argument("--text", required=True, type=str, help="Text to synthesize")
    parser.add_argument("--sid", required=False, type=int, help="Speaker ID to synthesize")
    args = parser.parse_args()

    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession(str(args.model), sess_options=sess_options, providers=["CPUExecutionProvider"])

    hps = utils.get_hparams_from_file(args.config_path)

    phoneme_ids = get_text(args.text, hps)
    text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    text_lengths = np.array([text.shape[1]], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    sid = np.array([int(args.sid)]) if args.sid is not None else None

    audio = model.run(
        None,
        {
            "input": text,
            "input_lengths": text_lengths,
            "scales": scales,
            "sid": sid,
        },
    )[0].squeeze((0, 1))

    write(data=audio, rate=hps.data.sampling_rate, filename=args.output_wav_path)


if __name__ == "__main__":
    main()
