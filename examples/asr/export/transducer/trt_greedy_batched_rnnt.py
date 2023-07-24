import tensorrt as trt
import torch
from nemo.utils import logging
from nemo.collections.asr.parts.utils import rnnt_utils
from typing import Optional
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

class ExportedModelGreedyBatchedRNNTInfer:
    def __init__(self, encoder_model: str, decoder_joint_model: str, max_symbols_per_step: Optional[int] = None):
        self.encoder_model_path = encoder_model
        self.decoder_joint_model_path = decoder_joint_model
        self.max_symbols_per_step = max_symbols_per_step

        # Will be populated at runtime
        self._blank_index = None

    def __call__(self, audio_signal: torch.Tensor, length: torch.Tensor):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-regressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        with torch.no_grad():
            # Apply optional preprocessing
            # print("GALVEZ: original inputs=", audio_signal.shape, length.max())
            encoder_output, encoded_lengths = self.run_encoder(audio_signal=audio_signal, length=length)
            # print("GALVEZ: encoded shapes=", encoder_output.shape, encoded_lengths.max())
            if torch.is_tensor(encoder_output):
                encoder_output = encoder_output.transpose(1, 2)
            else:
                encoder_output = encoder_output.transpose([0, 2, 1])  # (B, T, D)
            logitlen = encoded_lengths

            inseq = encoder_output  # [B, T, D]
            hypotheses, timestamps = self._greedy_decode(inseq, logitlen)

            # Pack the hypotheses results
            packed_result = [rnnt_utils.Hypothesis(score=-1.0, y_sequence=[]) for _ in range(len(hypotheses))]
            for i in range(len(packed_result)):
                packed_result[i].y_sequence = torch.tensor(hypotheses[i], dtype=torch.long)
                packed_result[i].length = timestamps[i]

            del hypotheses

        return packed_result

    def _greedy_decode(self, x, out_len):
        # x: [B, T, D]
        # out_len: [B]

        # Initialize state
        batchsize = x.shape[0]
        hidden = self._get_initial_states(batchsize)
        target_lengths = torch.ones(batchsize, dtype=torch.int32)

        # Output string buffer
        label = [[] for _ in range(batchsize)]
        timesteps = [[] for _ in range(batchsize)]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long).numpy()
        if torch.is_tensor(x):
            last_label = torch.from_numpy(last_label).to(self.device)

        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool).numpy()

        # Get max sequence length
        # Is this longer than f's length in some cases?
        max_out_len = out_len.max()
        # This assertion is currently not true
        # assert max_out_len.item() <= x.shape[1]
        max_out_len = min(max_out_len, x.shape[1])
        for time_idx in range(max_out_len):
            f = x[:, time_idx : time_idx + 1, :]  # [B, 1, D]
            # print("GALVEZ:max_out_len=", max_out_len)
            # print("GALVEZ:xshape=", x.shape)
            # print("GALVEZ:fshape=", f.shape)

            if torch.is_tensor(f):
                f = f.transpose(1, 2)
            else:
                f = f.transpose([0, 2, 1])

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask *= False

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
            blank_mask = time_idx >= out_len
            # Start inner loop
            while not_blank and (self.max_symbols_per_step is None or symbols_added < self.max_symbols_per_step):

                # Batch prediction and joint network steps
                # If very first prediction step, submit SOS tag (blank) to pred_step.
                # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                if time_idx == 0 and symbols_added == 0:
                    g = torch.tensor([self._blank_index] * batchsize, dtype=torch.int32).view(-1, 1)
                else:
                    if torch.is_tensor(last_label):
                        g = last_label.type(torch.int32)
                    else:
                        g = last_label.astype(np.int32)
                # Batched joint step - Output = [B, V + 1]
                joint_out, hidden_prime = self.run_decoder_joint(f, g, target_lengths, *hidden)
                logp, pred_lengths = joint_out
                logp = logp[:, 0, 0, :]

                # Get index k, of max prob for batch
                if torch.is_tensor(logp):
                    v, k = logp.max(1)
                else:
                    k = np.argmax(logp, axis=1).astype(np.int32)


                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == self._blank_index
                blank_mask |= k_is_blank
                del k_is_blank
                del logp

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False

                else:
                    # Collect batch indices where blanks occurred now/past
                    if torch.is_tensor(blank_mask):
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
                    else:
                        blank_indices = blank_mask.astype(np.int32).nonzero()

                    if type(blank_indices) in (list, tuple):
                        blank_indices = blank_indices[0]

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
                        for state_id in range(len(hidden)):
                            hidden_prime[state_id][:, blank_indices, :] = hidden[state_id][:, blank_indices, :]

                    elif len(blank_indices) > 0 and hidden is None:
                        # Reset state if there were some blank and other non-blank predictions in batch
                        # Original state is filled with zeros so we just multiply
                        # LSTM has 2 states
                        for state_id in range(len(hidden_prime)):
                            hidden_prime[state_id][:, blank_indices, :] *= 0.0

                    # Recover prior predicted label for all samples which predicted blank now/past
                    k[blank_indices] = last_label[blank_indices, 0]

                    # Update new label and hidden state for next iteration
                    if torch.is_tensor(k):
                        last_label = k.clone().reshape(-1, 1)
                    else:
                        last_label = k.copy().reshape(-1, 1)
                    hidden = hidden_prime

                    # Update predicted labels, accounting for time mask
                    # If blank was predicted even once, now or in the past,
                    # Force the current predicted label to also be blank
                    # This ensures that blanks propogate across all timesteps
                    # once they have occured (normally stopping condition of sample level loop).
                    for kidx, ki in enumerate(k):
                        if blank_mask[kidx] == 0:
                            label[kidx].append(ki)
                            timesteps[kidx].append(time_idx)

                    symbols_added += 1

        return label, timesteps

    def _setup_blank_index(self):
        raise NotImplementedError()

    def run_encoder(self, audio_signal, length):
        raise NotImplementedError()

    def run_decoder_joint(self, enc_logits, targets, target_length, *states):
        raise NotImplementedError()

    def _get_initial_states(self, batchsize):
        raise NotImplementedError()


class TRTGreedyBatchedRNNTInfer(ExportedModelGreedyBatchedRNNTInfer):
    def __init__(self, encoder_model: str, decoder_joint_model: str, max_symbols_per_step: Optional[int] = 10):
        super().__init__(
            encoder_model=encoder_model,
            decoder_joint_model=decoder_joint_model,
            max_symbols_per_step=max_symbols_per_step,
        )

        with open(encoder_model, "rb") as f:
            serialized_engine = f.read()
        encoder_engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.encoder_context = encoder_engine.create_execution_context()
        self.encoder_engine = encoder_engine

        with open(decoder_joint_model, "rb") as f:
            serialized_engine = f.read()
        decoder_joint_engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.decoder_joint_context = decoder_joint_engine.create_execution_context()
        self.decoder_engine = decoder_joint_engine
        stream = torch.cuda.Stream()
        self.enc_stream = stream
        self.enc_stream_ptr = stream.cuda_stream

        stream2 = torch.cuda.Stream()
        self.dec_stream = stream2
        self.dec_stream_ptr = stream2.cuda_stream

        #logging.info("Successfully loaded encoder, decoder and joint onnx models !")

        # Will be populated at runtime
        self._blank_index = None
        self.max_symbols_per_step = max_symbols_per_step
        self.device = 'cpu'
        self._setup_encoder_input_output_keys()
        self._setup_decoder_joint_input_output_keys()
        self._setup_blank_index()

    def _setup_encoder_input_output_keys(self):
        self.encoder_inputs = ["audio_signal", "length"]
        self.encoder_outputs = ["outputs", "encoded_lengths"]

    def _setup_decoder_joint_input_output_keys(self):
        self.decoder_joint_inputs = ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"]
        self.decoder_joint_outputs = ["outputs", "prednet_lengths", "output_states_1", "output_states_2"]

    def _setup_blank_index(self):
        # ASSUME: Single input with no time length information
        # I'm confused
        dynamic_dim = 16
        seq_len = 96
        ip_shape = [dynamic_dim, 80, seq_len]
        enc_logits, encoded_length = self.run_encoder(
            audio_signal=torch.randn(*ip_shape), length=torch.randint(0, 1, size=(dynamic_dim,))
        )

        # prepare states
        states = self._get_initial_states(batchsize=dynamic_dim)

        # run decoder 1 step
        joint_out, states = self.run_decoder_joint(enc_logits[:,:,0:1], None, None, *states)
        log_probs, lengths = joint_out

        self._blank_index = log_probs.shape[-1] - 1  # last token of vocab size is blank token
        logging.info(
            f"Enc-Dec-Joint step was evaluated, blank token id = {self._blank_index}; vocab size = {log_probs.shape[-1]}"
        )

    def run_encoder(self, audio_signal, length):
        audio_signal = audio_signal.contiguous().cuda()
        length = length.contiguous().cuda()
       
        idx0 = self.encoder_engine.get_binding_index(self.encoder_inputs[0])
       
        idx1 = self.encoder_engine.get_binding_index(self.encoder_inputs[1])
        #idx2 = self.encoder_engine.get_binding_index(self.encoder_outputs[0])
        #idx3 = self.encoder_engine.get_binding_index(self.encoder_outputs[1])
       
        self.encoder_context.set_binding_shape(idx0, tuple(audio_signal.shape))
        self.encoder_context.set_binding_shape(idx1, tuple(length.shape))

        #for binding in self.encoder_engine:
        #    binding_idx = self.encoder_engine.get_binding_index(binding)
        #    size = trt.volume(self.encoder_context.get_binding_shape(binding_idx))
        #    dtype = trt.nptype(self.encoder_engine.get_binding_dtype(binding))
       
        # subsampling rate is 4 -> is there some way to do this sort of shape
        # inference automatically? Maybe via onnx?
        # https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md
        audio_len = audio_signal.shape[-1]
        # This appears to be incorrect...
        # output_length = ((audio_len + 1) // 2 + 1 ) // 2
        output_length = (audio_len - 1) // 2 + 1
        print(output_length)
        output_length = (output_length - 1) // 2 + 1
        print(output_length)
        # output_length = ((((audio_len - 1) // 2 + 1 ) - 1 ) // 2) + 1
        output1_shape = (audio_signal.shape[0], 512, output_length)
        output2_shape = (audio_signal.shape[0],)
        
        enc_out = torch.zeros(output1_shape, dtype=torch.float32).contiguous().cuda()
        encoded_length = torch.zeros(output2_shape, dtype=torch.int64).contiguous().cuda()
        enc_out_ptr = enc_out.data_ptr()
        encoded_length_ptr = encoded_length.data_ptr()
        
        buffers = [audio_signal.data_ptr(), length.data_ptr(), enc_out_ptr, encoded_length_ptr]
        for _ in range(3):
            self.encoder_context.execute_async_v2(buffers, self.enc_stream_ptr)
            self.enc_stream.synchronize()

        torch.cuda.cudart().cudaProfilerStart()
        self.encoder_context.execute_async_v2(buffers, self.enc_stream_ptr)
        self.enc_stream.synchronize()
        enc_out = enc_out.to(self.device)
        encoded_length = encoded_length.to(self.device)
        # print("GALVEZ:enc_out=", enc_out.shape)
        # print("GALVEZ:encoded_length=", encoded_length)
        torch.cuda.cudart().cudaProfilerStop()
        return enc_out, encoded_length

    def run_decoder_joint(self, enc_logits, targets, target_length, *states):
        # ASSUME: Decoder is RNN Transducer
        if targets is None:
            targets = torch.zeros(enc_logits.shape[0], 1, dtype=torch.int32)
            target_length = torch.ones(enc_logits.shape[0], dtype=torch.int32)

        # print("GALVEZ0:", enc_logits.shape)

        enc_logits = enc_logits.contiguous().cuda()
        targets = targets.contiguous().cuda()
        target_length = target_length.contiguous().cuda()
        state0 = states[0].contiguous().cuda()
        state1 = states[1].contiguous().cuda()
       
        idx0 = self.decoder_engine.get_binding_index(self.decoder_joint_inputs[0])
        idx1 = self.decoder_engine.get_binding_index(self.decoder_joint_inputs[1])
        idx2 = self.decoder_engine.get_binding_index(self.decoder_joint_inputs[2])
        idx3 = self.decoder_engine.get_binding_index(self.decoder_joint_inputs[3])
        idx4 = self.decoder_engine.get_binding_index(self.decoder_joint_inputs[4])


        idx5 = self.decoder_engine.get_binding_index(self.decoder_joint_outputs[0])
        idx6 = self.decoder_engine.get_binding_index(self.decoder_joint_outputs[1])
        idx7 = self.decoder_engine.get_binding_index(self.decoder_joint_outputs[2])
        idx8 = self.decoder_engine.get_binding_index(self.decoder_joint_outputs[3])
        # 0 1 2 3 4 7 8 5 6
        # binding order is different

        # print("GALVEZ:shape=", tuple(enc_logits.shape))
        self.decoder_joint_context.set_binding_shape(idx0, tuple(enc_logits.shape))
        self.decoder_joint_context.set_binding_shape(idx1, tuple(targets.shape))
        self.decoder_joint_context.set_binding_shape(idx2, tuple(target_length.shape))
        self.decoder_joint_context.set_binding_shape(idx3, tuple(state0.shape))
        self.decoder_joint_context.set_binding_shape(idx4, tuple(state1.shape))
       
        B = enc_logits.shape[0]
        outputs = torch.zeros((B, 1, 1, 1025), dtype=torch.float32).contiguous().cuda()
        prednet_lengths = torch.zeros((B,), dtype=torch.int32).contiguous().cuda()
        output_states_1 = torch.zeros((1, B, 640), dtype=torch.float32).contiguous().cuda()
        output_states_2 = torch.zeros((1, B, 640), dtype=torch.float32).contiguous().cuda()
        
        buffers = []
        
        buffers = [enc_logits.data_ptr(), targets.data_ptr(), 
                   target_length.data_ptr(), state0.data_ptr(), state1.data_ptr(), 
                   output_states_1.data_ptr(), output_states_2.data_ptr(),
                   outputs.data_ptr(), prednet_lengths.data_ptr()]

        self.decoder_joint_context.execute_async_v2(buffers, self.dec_stream_ptr)
        self.dec_stream.synchronize()
        dec_out = [outputs.to(self.device), prednet_lengths.to(self.device)]
        new_states = [output_states_1.to(self.device), output_states_2.to(self.device)]

        return dec_out, new_states

    def _get_initial_states(self, batchsize):

        input_states = [torch.zeros(1, batchsize, 640),
                        torch.zeros(1, batchsize, 640)]

        return input_states
