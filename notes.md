###### ...my minimal iplementation of Pixart-alpha (a 'distillation' of the main codebase)

_start_ -> [27-10-2024]

This project aims to be a **hackable** and stripped down impl. of the **Pixart-alpha** paper, 
for efficient **text2image generation**.

#### Timeline
- 27 - read/digest paper/components
- 28 - Implement labelling pipeline described in paper, Attention blocks, T5 encoder
- 29 - 

#### Notes
* - **Modulation**: adjusts attention weights, using constant scale(multiply) and shift(add) values.
* - **Kaiming initialization**: parameters are initialized using a normal distribution with a mean of 0 and a standard deviation of `1 / sqrt(hidden_size)`