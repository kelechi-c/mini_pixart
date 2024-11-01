###### ...my minimal iplementation of Pixart-alpha (a 'distillation' of the main codebase)

_start_ -> [27-10-2024]

This project aims to be a **hackable** and stripped down impl. of the **Pixart-alpha** paper, 
for efficient **text2image generation**.

I could use the code to train a text2image model but..
trust me, the original codebase is something else, definitely not minimal.
I now decided to spend time understanding it. It's more fun to create my own version anyway **:)**

As of **November 1st**, the model was ready(at least for now), next is the training and sampling part..
which is just a brain-wreck for me. I will take a day or two break for now.


#### Timeline
- 27 - read/digest paper/components.
- 28 - Implement labelling pipeline described in paper, Attention blocks, T5 encoding
- 29 - DiT blocks, Full Pixart DiT model.
- 30 - MLP layers, Time/text conditioning embedder.
- 31 - What did I even do...hmm. _oh_, other parts of the model, initialization...
- Nov 1 - Dataloader. Preprocessing function, part of trainig code. The sampling/loss part offed me...for now
- 

#### Notes
* - **Modulation**: adjusts attention weights, using constant scale(multiply) and shift(add) values.
* - **Kaiming initialization**: parameters are initialized using a normal distribution with a mean of 0 and a standard deviation of `1 / sqrt(hidden_size)`