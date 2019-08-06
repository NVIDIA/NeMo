
# Simplest Example
Just learns simple function `y=sin(x)`.
Simply run from `examples/start_here` folder.

# ChatBot Example
This is an adaptation of [Pytorch Chatbot tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
Simply run from `examples/start_here` folder.

During training it will print **SOURCE**, **PREDICTED RESPONSE** and **TARGET**.

* **SOURCE** it what the model's input was (e.g. what user said)
* **PREDICTED RESPONSE** is what model said
* **TARGET** is what model should have said.

Closer to the end of training (~15-20 epochs) you should see **PREDICTED RESPONSE** become more and more similar to **TARGET**.  

# ChatBot Example 2
The example demonstrates NEMO's flexibility - here, you will use two encoders instead of one like in
previous example (and in a typical seq2seq model).
```python
...
# Instance one on EncoderRNN
encoder1 = neural_factory.get_module(
  name="EncoderRNN", collection="tutorials",
  params=config)
# Instance two on EncoderRNN. It will have different weights from instance one
encoder2 = neural_factory.get_module(
  name="EncoderRNN", collection="tutorials",
  params=config)
mixer = neural_factory.get_module(
  name="SimpleCombiner", collection="other",
  params={}
)
...
encoder_outputs1, encoder_hidden1 = encoder1(input_seq=src,
                                             input_lengths=src_lengths)
encoder_outputs2, encoder_hidden2 = encoder2(input_seq=src,
                                             input_lengths=src_lengths)
encoder_outputs = mixer(x1=encoder_outputs1, x2=encoder_outputs2)
outputs, hidden = decoder(targets=tgt,
                          encoder_outputs=encoder_outputs,
                          max_target_len=max_tgt_length)
...                          
```
Simply run from `examples/start_here` folder.