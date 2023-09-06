from text.text_preprocessing import available_phonemizers, available_symbol_set


def parse_args(parser):
    """Parse commandline arguments."""

    parser.add_argument(
        "--dataset",
        default="ljspeech",
        choices=["ljspeech"],
        type=str,
        help="select dataset to train with",
    )
    parser.add_argument(
        "--logging-dir", type=str, default=None, help="directory to save the log files"
    )
    parser.add_argument(
        "--dataset-path", type=str, default="./", help="path to dataset"
    )
    parser.add_argument(
        "--val-ratio",
        default=0.1,
        type=float,
        help="the ratio of waveforms for validation",
    )

    parser.add_argument(
        "--anneal-steps", nargs="*", help="epochs after which decrease learning rate"
    )
    parser.add_argument(
        "--anneal-factor",
        type=float,
        choices=[0.1, 0.3],
        default=0.1,
        help="factor for annealing learning rate",
    )
    preprocessor = parser.add_argument_group("text preprocessor setup")
    preprocessor.add_argument(
        "--text-preprocessor",
        default="english_characters",
        type=str,
        choices=available_symbol_set,
        help="select text preprocessor to use.",
    )
    preprocessor.add_argument(
        "--phonemizer",
        type=str,
        choices=available_phonemizers,
        help='select phonemizer to use, only used when text-preprocessor is "english_phonemes"',
    )
    preprocessor.add_argument(
        "--phonemizer-checkpoint",
        type=str,
        help="the path or name of the checkpoint for the phonemizer, "
        'only used when text-preprocessor is "english_phonemes"',
    )
    preprocessor.add_argument(
        "--cmudict-root",
        default="./",
        type=str,
        help="the root directory for storing cmudictionary files",
    )

    # training
    training = parser.add_argument_group("training setup")
    training.add_argument("--epochs", type=int, help="number of total epochs to run")

    training.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Directory to save model and training state checkpoints",
    )

    training.add_argument(
        "--workers", default=8, type=int, help="number of data loading workers"
    )

    training.add_argument(
        "--validate-freq",
        default=10,
        type=int,
        metavar="N",
        help="validation frequency in epochs",
    )

    training.add_argument(
        "--checkpoint-freq",
        default=10,
        type=int,
        metavar="N",
        help="checkpoint frequency in iterations",
    )

    training.add_argument(
        "--logging-freq",
        default=10,
        type=int,
        metavar="N",
        help="logging frequency in epochs",
    )

    optimization = parser.add_argument_group("optimization setup")
    optimization.add_argument(
        "--learning-rate", default=1e-3, type=float, help="initial learing rate"
    )
    optimization.add_argument(
        "--weight-decay", default=1e-6, type=float, help="weight decay"
    )
    optimization.add_argument(
        "--batch-size", default=32, type=int, help="batch size per GPU"
    )
    optimization.add_argument(
        "--grad-clip",
        default=5.0,
        type=float,
        help="clipping gradient with maximum gradient norm value",
    )

    # model parameters
    model = parser.add_argument_group("model parameters")
    model.add_argument(
        "--mask-padding", action="store_true", default=False, help="use mask padding"
    )
    model.add_argument(
        "--symbols-embedding-dim",
        default=512,
        type=int,
        help="input embedding dimension",
    )

    # encoder
    model.add_argument(
        "--encoder-embedding-dim",
        default=512,
        type=int,
        help="encoder embedding dimension",
    )
    model.add_argument(
        "--encoder-n-convolution",
        default=3,
        type=int,
        help="number of encoder convolutions",
    )
    model.add_argument(
        "--encoder-kernel-size", default=5, type=int, help="encoder kernel size"
    )
    # decoder
    model.add_argument(
        "--n-frames-per-step",
        default=1,
        type=int,
        help="number of frames processed per step (currently only 1 is supported)",
    )
    model.add_argument(
        "--decoder-rnn-dim",
        default=1024,
        type=int,
        help="number of units in decoder LSTM",
    )
    model.add_argument(
        "--decoder-dropout",
        default=0.1,
        type=float,
        help="dropout probability for decoder LSTM",
    )
    model.add_argument(
        "--decoder-max-step",
        default=2000,
        type=int,
        help="maximum number of output mel spectrograms",
    )
    model.add_argument(
        "--decoder-no-early-stopping",
        action="store_true",
        default=False,
        help="stop decoding only when all samples are finished",
    )

    # attention model
    model.add_argument(
        "--attention-hidden-dim",
        default=128,
        type=int,
        help="dimension of attention hidden representation",
    )
    model.add_argument(
        "--attention-rnn-dim",
        default=1024,
        type=int,
        help="number of units in attention LSTM",
    )
    model.add_argument(
        "--attention-location-n-filter",
        default=32,
        type=int,
        help="number of filters for location-sensitive attention",
    )
    model.add_argument(
        "--attention-location-kernel-size",
        default=31,
        type=int,
        help="kernel size for location-sensitive attention",
    )
    model.add_argument(
        "--attention-dropout",
        default=0.1,
        type=float,
        help="dropout probability for attention LSTM",
    )

    model.add_argument(
        "--prenet-dim",
        default=256,
        type=int,
        help="number of ReLU units in prenet layers",
    )

    # mel-post processing network parameters
    model.add_argument(
        "--postnet-n-convolution",
        default=5,
        type=float,
        help="number of postnet convolutions",
    )
    model.add_argument(
        "--postnet-kernel-size", default=5, type=float, help="postnet kernel size"
    )
    model.add_argument(
        "--postnet-embedding-dim",
        default=512,
        type=float,
        help="postnet embedding dimension",
    )

    model.add_argument(
        "--gate-threshold",
        default=0.5,
        type=float,
        help="probability threshold for stop token",
    )

    # audio parameters
    audio = parser.add_argument_group("audio parameters")
    audio.add_argument("--sample-rate", default=22050, type=int, help="Sampling rate")
    audio.add_argument("--n-fft", default=1024, type=int, help="Filter length for STFT")
    audio.add_argument(
        "--hop-length", default=256, type=int, help="Hop (stride) length"
    )
    audio.add_argument("--win-length", default=1024, type=int, help="Window length")
    audio.add_argument("--n-mels", default=80, type=int, help="")
    audio.add_argument(
        "--mel-fmin", default=0.0, type=float, help="Minimum mel frequency"
    )
    audio.add_argument(
        "--mel-fmax", default=8000.0, type=float, help="Maximum mel frequency"
    )

    return parser
