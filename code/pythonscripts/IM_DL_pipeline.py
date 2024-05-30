import warnings
import math
from torch import nn, Tensor
from einops.layers.torch import Rearrange
import numpy as np
from datetime import datetime, timedelta
import torch

# Get location of this file to find path to models
from inspect import getsourcefile
from os.path import dirname
currentDic = dirname(getsourcefile(lambda:0))
modelsDic = dirname(currentDic) + "/models"

# Path to the debug log file
debug_log_file = f"{currentDic}/output/IM_DL_debug_log.txt"
open(debug_log_file, 'w').close()

def log_debug_message(message):
    with open(debug_log_file, 'a') as f:
        f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {message}\n')

def remove_arch_prefix(key):
    return key.replace('arch.', '') if key.startswith('arch.') else key

class MyOVBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)
        self.signalHeader = None
        self.outputHeader = None
        log_debug_message("Initializing MyOVBox...")
        self.processor = EEGProcessor(
            model_path=f"{modelsDic}/ComfyNet_best_8_channels.ckpt",
            output_file=f"{currentDic}/output/IM_DL_predictions.txt"
        )
        log_debug_message("MyOVBox initialized.")

    def process(self):
        for chunkIndex in range(len(self.input[0])):
            log_debug_message(f"Processing chunk index: {chunkIndex}")
            if type(self.input[0][chunkIndex]) == OVSignalHeader:
                log_debug_message("Received OVSignalHeader")
                self.signalHeader = self.input[0].pop()
                outputHeader = OVSignalHeader(
                    self.signalHeader.startTime,
                    self.signalHeader.endTime,
                    self.signalHeader.dimensionSizes,
                    self.signalHeader.dimensionLabels,
                    self.signalHeader.samplingRate)
                self.nbEEGChannels = self.signalHeader.dimensionSizes[0] - 2
                
            elif type(self.input[0][chunkIndex]) == OVSignalBuffer:
                log_debug_message("Received OVSignalBuffer")
                chunk = self.input[0][chunkIndex] 
                concatenated = np.array(chunk).tolist()
                to_process = []
                for i in range(self.nbEEGChannels):
                    start_idx = i * int(len(concatenated)/self.signalHeader.dimensionSizes[0])
                    end_idx = start_idx + int(len(concatenated)/self.signalHeader.dimensionSizes[0])
                    # last_sample = concatenated[end_idx]
                    channel = concatenated[start_idx:end_idx]
                    last_sample = channel[-1]
                    channel.append(last_sample)
                    to_process.append(channel)
                to_process = np.array([to_process], dtype=object)
                # self.output[0].append(chunk)  # Append the received buffer to the output
                if chunkIndex + 1 == len(self.input[0]):
                    self.processor.process_chunk(to_process)  

            elif type(self.input[0][chunkIndex]) == OVSignalEnd:
                log_debug_message("Received OVSignalEnd")
                self.output[0].append(self.input[0].pop())


class EEGProcessor:
    def __init__(self, model_path, output_file, fs=160, interval=4):
        log_debug_message("Initializing EEGProcessor...")
        self.model = self.load_model(model_path)
        self.fs = fs
        self.interval = interval
        self.buffer = []
        self.output_file = output_file
        self.initialize_output_file()
        log_debug_message(f"EEGProcessor initialized with model_path: {model_path}")
        self.lastPredictTime = datetime.now()

    def load_model(self, model_path):
        try:
            model = ComfyNet(n_outputs=int(3), n_chans=int(8), n_times=int(321))
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            modified_state_dict = {remove_arch_prefix(k): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(modified_state_dict)
            model.eval()
            # model = torch.load(model_path, map_location=torch.device('cpu'))
            log_debug_message(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            log_debug_message(f"Error loading model from {model_path}: {e}")
            return None

    def initialize_output_file(self):
        try:
            with open(self.output_file, 'w') as f:
                f.write('Timestamp,Prediction\n')
            log_debug_message("Output file initialized.")
        except Exception as e:
            log_debug_message(f"Error initializing output file: {e}")

    def append_to_output_file(self, prediction):
        if prediction is None:
            log_debug_message("No prediction to append to the output file.")
            return
        try:
            with open(self.output_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')[:-4]
                # print(f"Write to predictions file: {timestamp}")
                f.write(f'{timestamp},{prediction}\n')
            log_debug_message(f"Prediction {prediction} appended to output file at {timestamp}.")
        except Exception as e:
            log_debug_message(f"Error appending to output file: {e}")

    def process_chunk(self, chunk):
        # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')[:-4]
        # print(f"Process chunk at time {timestamp}")
        now = datetime.now()
        if (now-self.lastPredictTime) > timedelta(milliseconds=900):
            self.lastPredictTime = now
            log_debug_message(f"Processing chunk with length: {len(chunk)}")
            data = np.array( chunk, dtype = np.float64)
            log_debug_message(data)
            log_debug_message(f"Data prepared for prediction: {data.shape}")
            prediction = self.predict(data)
            log_debug_message(f"Prediction made: {prediction}")
            self.append_to_output_file(prediction)

    def predict(self, data):
        log_debug_message('Entering Predict')
        # Check if model is loaded
        if self.model is None:
            log_debug_message("Model is not loaded. Cannot make predictions.")
            return None
        try:
            # Convert data to tensor and cast to float
            data_tensor = torch.tensor(data).float()
            output = self.model(data_tensor)
            log_debug_message(f"Model raw output: {output}")
            # Extract prediction
            prediction = output.argmax(dim=1).item()

            # 0: neutral, 1: left, 2: right
            if prediction == 1:
                # Left action
                print("LEFT: Cast sign.")
                pressKey(SPELL_KEY)
            elif prediction == 2:
                # Right action
                print("RIGHT: Call horse")
                pressKey(HORSE_KEY)
                # else:
                #     print(f"No action {timestamp}")
                log_debug_message(f"Prediction: {prediction}")
            return prediction
        except Exception as e:
            log_debug_message(f"An error occurred: {str(e)}")

class ComfyNet(nn.Module):

    """Convolutional Transformer for EEG decoding.
 
    The paper and original code with more details about the methodological

    choices are available at the [Song2022]_ and [ConformerCode]_.
 
    This neural network architecture receives a traditional braindecode input.

    The input shape should be three-dimensional matrix representing the EEG

    signals.
 
         `(batch_size, n_channels, n_timesteps)`.
 
    The EEG Conformer architecture is composed of three modules:

        - PatchEmbedding

        - TransformerEncoder

        - ClassificationHead
 
    Notes

    -----

    The authors recommend using data augmentation before using Conformer,

    e.g. segmentation and recombination,

    Please refer to the original paper and code for more details.
 
    The model was initially tuned on 4 seconds of 250 Hz data.

    Please adjust the scale of the temporal convolutional layer,

    and the pooling layer for better performance.
 
    We aggregate the parameters based on the parts of the models, or

    when the parameters were used first, e.g. n_filters_time.
 
    Parameters

    ----------

    n_filters_time: int

        Number of temporal filters, defines also embedding size.

    filter_time_length: int

        Length of the temporal filter.

    pool_time_length: int

        Length of temporal pooling filter.

    pool_time_stride: int

        Length of stride between temporal pooling filters.

    drop_prob: float

        Dropout rate of the convolutional layer.

    att_depth: int

        Number of self-attention layers.

    att_heads: int

        Number of attention heads.

    return_features: bool

        If True, the forward method returns the features before the

        last classification layer. Defaults to False.

    References

    ----------

    .. [Song2022] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG

       conformer: Convolutional transformer for EEG decoding and visualization.

       IEEE Transactions on Neural Systems and Rehabilitation Engineering,

       31, pp.710-719. https://ieeexplore.ieee.org/document/9991178

    .. [ConformerCode] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG

       conformer: Convolutional transformer for EEG decoding and visualization.
https://github.com/eeyhsong/EEG-Conformer.

    """
 
    def __init__(

        self,

        n_outputs=None,

        n_chans=None,

        n_filters_time=40,

        filter_time_length=25,

        pool_time_length=75,

        pool_time_stride=15,

        drop_prob=0.5,

        att_depth=2,

        att_heads=4,

        n_times=None,

        return_features=False,

    ):

        super().__init__()
 
        self.n_outputs = n_outputs

        self.n_chans = n_chans

        self.n_times = n_times
 
 
        self.mapping = {

            "classification_head.fc.6.weight": "final_layer.final_layer.0.weight",

            "classification_head.fc.6.bias": "final_layer.final_layer.0.bias",

        }
 
        if not (self.n_chans <= 64):

            warnings.warn(

                "This model has only been tested on no more "

                + "than 64 channels. no guarantee to work with "

                + "more channels.",

                UserWarning,

            )
 
        self.patch_embedding = _PatchEmbedding(

            n_filters_time=n_filters_time,

            filter_time_length=filter_time_length,

            n_channels=n_chans,

            pool_time_length=pool_time_length,

            stride_avg_pool=pool_time_stride,

            drop_prob=drop_prob,

        )
 
        self.positional_encoding = PositionalEncoding(

            n_filters_time,

            n_times,

        )
 
        final_fc_length = self.get_fc_size()
 
 
        self.tranformer_layer = nn.TransformerEncoderLayer(d_model=n_filters_time, nhead=att_heads, batch_first=True)

        self.transformer = nn.TransformerEncoder(self.tranformer_layer, num_layers=att_depth)
 
 
        self.fc = _FullyConnected(final_fc_length=final_fc_length)
 
        self.final_layer = _FinalLayer(

            n_classes=self.n_outputs,

            return_features=return_features,

        )
 
    def forward(self, x: Tensor) -> Tensor:

        x = torch.unsqueeze(x, dim=1)  # add one extra dimension

        x = self.patch_embedding(x)

        x = self.positional_encoding(x)

        x = self.transformer(x)

        x = self.fc(x)

        x = self.final_layer(x)

        return x
 
    def get_fc_size(self):

        out = self.patch_embedding(torch.ones((1, 1, self.n_chans, self.n_times)))

        size_embedding_1 = out.cpu().data.numpy().shape[1]

        size_embedding_2 = out.cpu().data.numpy().shape[2]
 
        return size_embedding_1 * size_embedding_2
 
 
class _PatchEmbedding(nn.Module):

    """Patch Embedding.
 
    The authors used a convolution module to capture local features,

    instead of position embedding.
 
    Parameters

    ----------

    n_filters_time: int

        Number of temporal filters, defines also embedding size.

    filter_time_length: int

        Length of the temporal filter.

    n_channels: int

        Number of channels to be used as number of spatial filters.

    pool_time_length: int

        Length of temporal poling filter.

    stride_avg_pool: int

        Length of stride between temporal pooling filters.

    drop_prob: float

        Dropout rate of the convolutional layer.
 
    Returns

    -------

    x: torch.Tensor

        The output tensor of the patch embedding layer.

    """
 
    def __init__(

        self,

        n_filters_time,

        filter_time_length,

        n_channels,

        pool_time_length,

        stride_avg_pool,

        drop_prob,

    ):

        super().__init__()
 
        self.shallownet = nn.Sequential(

            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),

            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),

            nn.BatchNorm2d(num_features=n_filters_time),

            nn.ELU(),

            nn.AvgPool2d(

                kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool)

            ),

            # pooling acts as slicing to obtain 'patch' along the

            # time dimension as in ViT

            nn.Dropout(p=drop_prob),

        )
 
        self.projection = nn.Sequential(

            nn.Conv2d(

                n_filters_time, n_filters_time, (1, 1), stride=(1, 1)

            ),  # transpose, conv could enhance fitting ability slightly

            Rearrange("b d_model 1 seq -> b seq d_model"),

        )
 
    def forward(self, x: Tensor) -> Tensor:

        x = self.shallownet(x)

        x = self.projection(x)

        return x
 
 
class PositionalEncoding(torch.nn.Module):

    def __init__(self, emb_size, max_len):

        super(PositionalEncoding, self).__init__()
 
        # Create a long enough `pe` matrix with shape (max_len, d_model)

        pe = torch.zeros(max_len, emb_size)
 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(

            torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size)

        )
 
        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)
 
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape (max_len, 1, d_model)

        self.register_buffer(

            "pe", pe

        )  # Not a trainable parameter but buffered for state tracking
 
    def forward(self, x):

        # `x` shape: (sequence_length, batch_size, d_model)

        return x + self.pe[: x.size(0), :]
 
 
class _FullyConnected(nn.Module):

    def __init__(

        self,

        final_fc_length,

        drop_prob_1=0.5,

        drop_prob_2=0.3,

        out_channels=256,

        hidden_channels=32,

    ):

        """Fully-connected layer for the transformer encoder.
 
        Parameters

        ----------

        final_fc_length : int

            Length of the final fully connected layer.

        drop_prob_1 : float

            Dropout probability for the first dropout layer.

        drop_prob_2 : float

            Dropout probability for the second dropout layer.

        out_channels : int

            Number of output channels for the first linear layer.

        hidden_channels : int

            Number of output channels for the second linear layer.

        """
 
        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(final_fc_length, out_channels),

            nn.ELU(),

            nn.Dropout(drop_prob_1),

            nn.Linear(out_channels, hidden_channels),

            nn.ELU(),

            nn.Dropout(drop_prob_2),

        )
 
    def forward(self, x):

        x = x.contiguous().view(x.size(0), -1)

        out = self.fc(x)

        return out
 
 
class _FinalLayer(nn.Module):

    def __init__(

            self, n_classes, hidden_channels=32, return_features=False

    ):

        """Classification head for the transformer encoder.
 
        Parameters

        ----------

        n_classes : int

            Number of classes for classification.

        hidden_channels : int

            Number of output channels for the second linear layer.

        return_features : bool

            Whether to return input features.

        """
 
        super().__init__()

        self.final_layer = nn.Sequential(

            nn.Linear(hidden_channels, n_classes),

        )

        self.return_features = return_features
 
        # WARNING : The original code uses a custom loss function

        # classification = nn.CrossEntropyLoss()

        #

        # if not self.return_features:

        #     self.final_layer.add_module("classification", classification)
 
    def forward(self, x):

        if self.return_features:

            out = self.final_layer(x)

            return out, x

        else:

            out = self.final_layer(x)

            return out



# Notify OpenViBE that the box instance 'box' is now an instance of MyOVBox
box = MyOVBox()

