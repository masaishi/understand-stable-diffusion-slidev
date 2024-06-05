---
# try also 'default' to start simple
theme: seriph
colorSchema: 'light'

# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: backgrounds/understand-sd.webp
# some information about your slides, markdown enabled
title: Understand Stable Diffusion from Code
info: This slide explains image generation using Latent Diffusion Models through source code. This slide explains the mechanism of image generation through the code of a library called parediffusers, which simplifies diffusers.

author: masaishi
keywords: [ Stable Diffusion, Diffusers, parediffusers, AI, ML, Generative Models ]

favicon: 'images/icon_tea_light.webp'

export:
  format: pdf
  timeout: 30000
  dark: false
  withClicks: false

lineNumbers: true

# apply any unocss classes to the current slide
class: text-center
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# https://sli.dev/guide/drawing
drawings:
  persist: true
# slide transition: https://sli.dev/guide/animations#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/guide/syntax#mdc-syntax
mdc: true

fonts:
  sans: Noto Serif JP, serif
---

# Understand Stable Diffusion from Code

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Understand Stable Diffusion from code, cyberpunk theme, best quality, high resolution, concept art</p>

---
title: Introduction
---

# 2. Masamune Ishihara
<div class="[&>*]:important-leading-10 opacity-80">
Computer Engineering Undergrad at University of California, Santa Cruz <br />
I'm interested in AI/ML and GIS.<br />

<br />

#### Likes:
- Tea
- Tennis
</div>

<div class="mt-10 flex flex-col gap-2">
  <div>
		<mdi-github-circle />
		<a href="https://github.com/masaishi" target="_blank" class="ml-1.5 border-none! font-300">masaishi</a>
	</div>
	<div>
		<mdi-twitter />
		<a href="https://twitter.com/masaishi2001" target="_blank" class="ml-1.5 border-none! font-300">@masaishi2001</a>
	</div>
	<div>
		<mdi-linkedin />
		<a href="https://www.linkedin.com/in/masamune-ishihara" target="_blank" class="ml-1.5 border-none! font-300">masamune-ishihara</a>
	</div>
	<div class="flex items-center">
		<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 512" class="h-5 w-5"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M304.2 501.5L158.4 320.3 298.2 185c2.6-2.7 1.7-10.5-5.3-10.5h-69.2c-3.5 0-7 1.8-10.5 5.3L80.9 313.5V7.5q0-7.5-7.5-7.5H21.5Q14 0 14 7.5v497q0 7.5 7.5 7.5h51.9q7.5 0 7.5-7.5v-109l30.8-29.3 110.5 140.6c3 3.5 6.5 5.3 10.5 5.3h66.9q5.3 0 6-3z"/></svg>
		<a href="https://www.kaggle.com/masaishi" target="_blank" class="ml-1.5 border-none! font-300">masaishi</a>
	</div>
</div>

<img src="/images/icon_tea_light.webp" class="rounded-full w-35 abs-tr mt-12 mr-24" />

---
level: 2
layout: center
---

Purpose

# Introduce image generation process with code

---
level: 2
layout: center
transition: fade
---

# About
In writing this slide, I realized many things that I did not understand. I may have explained things incorrectly, so I would appreciate it if you could tell me if I am unclear or if I have made any mistakes by clicking on the link below.

<br />

### [<mdi-github-circle />understand-stable-diffusion-slidev](https://github.com/masaishi/understand-stable-diffusion-slidev)

<br />

[<mdi-radiobox-marked />Issues](https://github.com/masaishi/understand-stable-diffusion-slidev/issues): Please let me know if you find any mistakes.

[<mdi-message-text-outline />Discussions](https://github.com/masaishi/understand-stable-diffusion-slidev/discussions): Please let me know if you have questions.

[<mdi-source-pull />Pull Requests](https://github.com/masaishi/understand-stable-diffusion-slidev/pulls): Please let me know if you have any improvements.

---
level: 2
layout: center
---

# About
The concept of this slide is to introduce the flow of image generation with code, so basically all the code in this slide can be actually run.

<br />

### Repository list

[<mdi-github-circle />understand-stable-diffusion-slidev](https://github.com/masaishi/understand-stable-diffusion-slidev): Repository of this slide.

[<mdi-github-circle />understand-stable-diffusion-slidev-notebooks](https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks): Notebooks for generating sample images and gifs.

[<mdi-github-circle />parediffusers](https://github.com/masaishi/parediffusers): Simple library for generating images without using huggingface/diffusers.

---
layout: center
title: Table of Contents
---

# Table of Contents
<Toc minDepth="1" maxDepth="1"></Toc>

---
layout: cover
title: Flow of Image Generation
background: /backgrounds/stable-diffusion.webp
---

# 4. Flow of Image Generation

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Stable Diffusion, watercolor painting, best quality, high resolution</p>

---
level: 2
layout: center
---

# What is Stable Diffusion?

- An image generation model based on the Latent Diffusion Model (LDM) developed by Stability AI.
- It can be used for Text-to-Image, Image-to-Image.
- It can be easily moved by using Diffusers.
- https://arxiv.org/abs/2112.10752

---
level: 2
layout: center
---

# What is Diffusers?

- Library for Diffusion Models developed by Hugging Face🤗.
- Easy to run many image generation models.
- <mdi-github-circle /> https://github.com/huggingface/diffusers

---
level: 2
layout: image-right
image: /exps/d-sd2-sample-42.webp
---

# [<mdi-github-circle />Diffusers](https://github.com/huggingface/diffusers)
## <!-- TODO: Find better way, currently for avoide below becomes subtitle -->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EbqeoWL5kPaDA8INLWl8g34v3vn83AQ5?usp=sharing)

Install the Diffusers library:
```python
!pip install transformers diffusers accelerate -U
```

Generate an image from text:
```python {all|4-7|8|10|all}{lines:true}
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-2",
  dtype=torch.float16,
).to(device=torch.device("cuda"))
prompt = "painting depicting the sea, sunrise, ship, artstation, 4k, concept art"

image = pipe(prompt, width=512, height=512).images[0]
display(image)
```

---
level: 2
layout: center
---

# Diffusers are highly flexible, but <br />
# that's why understand the code is difficult.

---
level: 2
---

# [<mdi-github-circle />diffusers/.../pipeline_stable_diffusion.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)

<iframe frameborder="0" scrolling="yes" class="overflow-scroll mt-10" style="width:100%; height:85%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fhuggingface%2Fdiffusers%2Fblob%2Fmain%2Fsrc%2Fdiffusers%2Fpipelines%2Fstable_diffusion%2Fpipeline_stable_diffusion.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

# [<mdi-github-circle />parediffusers/.../pipeline.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/pipeline.py)

<iframe frameborder="0" scrolling="yes" class="overflow-scroll mt-10" style="width:100%; height:85%;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fpipeline.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
layout: image-right
image: /exps/p-sd2-sample-43.webp
---

# [<mdi-github-circle />PareDiffusers](https://github.com/masaishi/parediffusers)
## <!-- TODO: Find better way, currently for avoide below becomes subtitle -->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I-qU3hfF19T42ksIh5FC0ReyKZ2hsJvx?usp=sharing)

Install the PareDiffusers library:
```python
!pip install parediffusers
```

Generate an image from text:
```python {all}{lines:true}
import torch
from parediffusers import PareDiffusionPipeline

pipe = PareDiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-2",
  device=torch.device("cuda"),
  dtype=torch.float16,
)
prompt = "painting depicting the sea, sunrise, ship, artstation, 4k, concept art"

image = pipe(prompt, width=512, height=512)
display(image)
```

---
level: 2
layout: center
---

# How is image generation performed?

---
level: 2
layout: image-right
image: /exps/p-sd2-sample-43.webp
---

# [<mdi-github-circle />PareDiffusers](https://github.com/masaishi/parediffusers)
## <!-- TODO: Find better way, currently for avoide below becomes subtitle -->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I-qU3hfF19T42ksIh5FC0ReyKZ2hsJvx?usp=sharing)

Install the PareDiffusers library:
```python
!pip install parediffusers
```

Generate an image from text:
```python {11}{lines:true}
import torch
from parediffusers import PareDiffusionPipeline

pipe = PareDiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-2",
  device=torch.device("cuda"),
  dtype=torch.float16,
)
prompt = "painting depicting the sea, sunrise, ship, artstation, 4k, concept art"

image = pipe(prompt, width=512, height=512)
display(image)
```

---
level: 2
layout: center
transition: fade
---

<div v-click=1 v-click.hide=2>

[<mdi-github-circle />pipeline.py#L117-L135](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L117-L135)

</div>

````md magic-move {style:'--slidev-code-font-size: 1.2rem; --slidev-code-line-height: 1.5;'}
```python {all}
image = pipe(prompt, width=512, height=512)
```
```python {all}
def __call__(self, prompt: str, height: int = 512, width: int = 512, ...):
	prompt_embeds = self.encode_prompt(prompt)
	latents = self.get_latent(width, height).unsqueeze(dim=0)
	latents = self.denoise(latents, prompt_embeds, ...)
	image = self.vae_decode(latents)
	return image
```
```md {all}
1. `encode_prompt` : Convert Prompt to Embedding.
2. `get_latent` : Create random Latent.
3. `denoise` : Denosing by using Scheduler and UNet.
4. `vae_decode` : Decode to pixel space by VAE.
```
````

---
level: 2
layout: center
---

```md {all|1|2|3|4|all}{lines:false, style:'--slidev-code-font-size: 1.2rem; --slidev-code-line-height: 1.5;'}
1. `encode_prompt` : Convert Prompt to Embedding.
2. `get_latent` : Create random Latent.
3. `denoise` : Denosing by using Scheduler and UNet.
4. `vae_decode` : Decode to pixel space by VAE.
```

<img src="/images/ldm-4step-figure.webp" class="mt-5" />

---
level: 2
layout: center
---

# A Briefly Theory

---
level: 2
layout: center
---

What is Latent Diffusion Model (LDM)?

# Models Which Run <span v-mark.yellow="1">DDPM</span> on Latent Space

---
level: 2
---

What is Denoising Diffusion Probabilistic Model (DDPM)?

<h1 class="!text-6.6"><span v-mark.red="1">Adding noise</span> to the image and <span v-mark.blue="2">restoring the original image from it</span>.</h1>

<p>It is used for audio and other data in general, but this slide will discuss images.</p>

<ul>
	<li><span v-mark.red="1">Diffusion process</span> is used to preprocess training data. Stochastic process（Markov chain)</li>
	<li><span v-mark.blue="2">Reverse process</span> is used to recover the original data from the noisy data.</li>
</ul>

<br />

<img src="/images/ddpm-figure.webp" class="abs-b mb-10 ml-auto mr-auto w-5/6" />

<!-- Reference -->
<p class="text-black text-xs abs-bl w-full mb-6 text-center">
Jonathan Ho, Ajay Jain, Pieter Abbeel: “Denoising Diffusion Probabilistic Models”, 2020; <a href='http://arxiv.org/abs/2006.11239'>arXiv:2006.11239</a>.
</p>

---
level: 2
layout: center
---

# It is interesting that it is called a Diffusion Model
# even if diffusion is not NN just processing.

---
level: 2
layout: center
---

What is Latent Diffusion Model (LDM)?

# Models Which Run DDPM on <span v-mark.green="1">Latent Space</span>


---
level: 2
layout: center
transition: fade
---

Difference of Loss Functions

<h2>
$$
L_{DM} := \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0, 1),  t}\Big[ \Vert \epsilon - \epsilon_\theta(x_{t},t) \Vert_{2}^{2}\Big] \, .
$$
</h2>

<h2>
$$
L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0, 1),  t}\Big[ \Vert \epsilon - \epsilon_\theta(z_{t},t) \Vert_{2}^{2}\Big] \, .
$$
</h2>

---
level: 2
layout: center
transition: fade
---

Latent Diffusion Model (LDM)

<h2>
$$
L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0, 1),  t}\Big[ \Vert \epsilon - \epsilon_\theta(z_{t},t) \Vert_{2}^{2}\Big] \, .
$$
</h2>

<!--
\mathcal{E} = VAE Encorder
\mathcal{N} = Gaussian noise
\epsilon_\theta = VAE Decoder

E(Expected): Input data is converted to latent space through Encorder.
When passing through the VAE Decoder, t=timestep is considered.
-->

---
level: 2
layout: center
---

Latent Diffusion Model (LDM) with Conditioning

<h2>
$$
L_{LDM} := \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0, 1), t }\Big[ \Vert \epsilon - \epsilon_\theta(z_{t},t, \tau_\theta(y)) \Vert_{2}^{2}\Big] \, ,
$$
</h2>

<v-clicks every="1" at="1">

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V
$$

$$
\begin{equation*}
Q = W^{(i)}_Q \cdot  \varphi_i(z_t), \; K = W^{(i)}_K \cdot \tau_\theta(y),
  \; V = W^{(i)}_V \cdot \tau_\theta(y) . \nonumber
%
\end{equation*}
$$

</v-clicks>

<!--
Conditioning, i.e., prompts, semantic maps, repres entations, images, etc. are considered.
-->

---
level: 2
transition: fade
---

<div class="flex flex-col !justify-between w-full h-120">
	<div>
		<img src="/images/ddpm-figure.webp" class="ml-auto mr-auto h-26" />
		<!-- Reference -->
		<p class="text-black text-xs w-full mt-6 text-center">
		Jonathan Ho, Ajay Jain, Pieter Abbeel: “Denoising Diffusion Probabilistic Models”, 2020; <a href='http://arxiv.org/abs/2006.11239'>arXiv:2006.11239</a>.
		</p>
	</div>
	<div v-click>
		<img src="/images/stable-diffusion-figure.webp" alt="Stable Diffusion Figure" class="ml-auto mr-auto h-48 object-contain" />
		<p class="text-black text-xs w-full mt-6 text-center">
		Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer: “High-Resolution Image Synthesis with Latent Diffusion Models”, 2021; <a href='http://arxiv.org/abs/2112.10752'>arXiv:2112.10752</a>.
		</p>
	</div>
</div>

---
level: 2
layout: center
---

<iframe frameborder="0" scrolling="no" style="width:100%; height:163px;" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2F035772c684ae8d16c7c908f185f6413b72658126%2Fsrc%2Fparediffusers%2Fpipeline.py%23L131-L134&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<div class="w-full flex flex-col justify-center mt-10.7">
<img src="/images/stable-diffusion-figure.webp" alt="Stable Diffusion Figure" class="h-48 object-contain" />
<p class="text-black text-xs w-full mt-6 text-center">
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer: “High-Resolution Image Synthesis with Latent Diffusion Models”, 2021; <a href='http://arxiv.org/abs/2112.10752'>arXiv:2112.10752</a>.
</p>
</div>

---
level: 2
layout: center
---

What is Latent Space?

# Features of the Input Image are Extracted


---
level: 2
layout: center
transition: fade
---

The flow of image generation in 4 steps

<h1 class="!text-7">
Step 1: Convert Prompt to Embedding.<br />
Step 2: Create random Latent.<br />
Step 3: Denosing by using Scheduler and UNet.<br />
Step 4: Decode to pixel space by VAE.
</h1>

---
level: 2
layout: center
---

The flow of image generation in 4 steps

<h1 class="!text-7">
Step 1: encode_prompt<br />
Step 2: get_latent<br />
Step 3: denoise<br />
Step 4: vae_decode
</h1>

---
layout: cover
title: "Step 1: encode_prompt"
background: /backgrounds/pipeline.webp
---

<h1>Step 1: encode_prompt</h1>

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Pipeline, cyberpunk theme, best quality, high resolution, concept art</p>

---
level: 2
layout: center
transition: fade
---

Step 1: encode_prompt
# Convert Prompt to Embedding.

---
level: 2
layout: center
---

Step 1: encode_prompt
# Convert prompts into a form that is easy for the model to handle.

---
level: 2
layout: center
---

Necessities
# - [CLIPTokenizer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/tokenization_clip.py#L251)
# - [CLIPTextModel](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py)
##
From [<mdi-github-circle />huggingface/transformers](https://github.com/huggingface/transformers/tree/main)

---
level: 2
layout: two-cols
transition: fade
---

<h1 class="!text-8.3">Step 1: encode_prompt</h1>
<p>Calling another function twice within encode_prompt</p>

::right::

[<mdi-github-circle />pipeline.py#L41-L48](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L41-L48)

```python {all|45|45-46}{lines:true,startLine:41,at:1}
def encode_prompt(self, prompt: str):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	prompt_embeds = self.get_embes(prompt, self.tokenizer.model_max_length)
	negative_prompt_embeds = self.get_embes([''], prompt_embeds.shape[1])
	prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
	return prompt_embeds
```

---
level: 2
layout: two-cols
transition: fade
---

<h1 class="!text-8.3">Step 1: encode_prompt</h1>
<p>Where are Necessities used?</p>

::right::

[<mdi-github-circle />pipeline.py#L41-L57](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L41-L57)

```python {all|54|54,56}{lines:true,startLine:41,at:1}
def encode_prompt(self, prompt: str):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	prompt_embeds = self.get_embes(prompt, self.tokenizer.model_max_length)
	negative_prompt_embeds = self.get_embes([''], prompt_embeds.shape[1])
	prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
	return prompt_embeds
 
def get_embes(self, prompt, max_length):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
	text_input_ids = text_inputs.input_ids.to(self.device)
	prompt_embeds = self.text_encoder(text_input_ids)[0].to(dtype=self.dtype, device=self.device)
	return prompt_embeds
```

---
level: 2
layout: two-cols
transition: fade
---

<h1 class="!text-8.3">Step 1: encode_prompt</h1>
<p>Where are Necessities used?</p>

<v-clicks every="1" at="1">

- L54: `CLIPTokenizer`: Token into text (Prompt). By making it a vector, it makes it easier to handle AI.

- L56: `CLIPTextModel`: Multi -modal model of language and image. In the image generation, the expression (embedding) of the image you want to make at the prompt is extracted.

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L21-L39](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L21-L39)

```python {4,5|4|4,5}{lines:false,at:1}
@classmethod
def from_pretrained(cls, model_name, device=torch.device("cuda"), dtype=torch.float16):
	# Ommit comments
	tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
	text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
	scheduler = PareDDIMScheduler.from_config(model_name, subfolder="scheduler")
	unet = PareUNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
	vae = PareAutoencoderKL.from_pretrained(model_name, subfolder="vae")
	return cls(tokenizer, text_encoder, scheduler, unet, vae, device, dtype)
```

[<mdi-github-circle />pipeline.py#L50-L57](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L50-L57)

```python {54,56|54|54,56}{lines:true,startLine:50,at:1}
def get_embes(self, prompt, max_length):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
	text_input_ids = text_inputs.input_ids.to(self.device)
	prompt_embeds = self.text_encoder(text_input_ids)[0].to(dtype=self.dtype, device=self.device)
	return prompt_embeds
```

---
level: 2
layout: two-cols
---

<h1 class="!text-8.3">Step 1: encode_prompt</h1>
<p>Read the code and understand the whole flow</p>

- L54: `CLIPTokenizer`: Token into text (Prompt). By making it a vector, it makes it easier to handle AI.

- L56: `CLIPTextModel`: Multi -modal model of language and image. In the image generation, the expression (embedding) of the image you want to make at the prompt is extracted.


<v-clicks every="1" at="2">

- L46: Negative_prompt is an empty character string to make it simple.

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L34-L35](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L34-L35)

```python {all}{lines:true,startLine:34,at:1}
	tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
	text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
```

[<mdi-github-circle />pipeline.py#L41-L57](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L41-L57)

```python {|all|46|all}{lines:true,startLine:41,at:1}
def encode_prompt(self, prompt: str):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	prompt_embeds = self.get_embes(prompt, self.tokenizer.model_max_length)
	negative_prompt_embeds = self.get_embes([''], prompt_embeds.shape[1])
	prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
	return prompt_embeds
 
def get_embes(self, prompt, max_length):
	"""
	Encode the text prompt into embeddings using the text encoder.
	"""
	text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
	text_input_ids = text_inputs.input_ids.to(self.device)
	prompt_embeds = self.text_encoder(text_input_ids)[0].to(dtype=self.dtype, device=self.device)
	return prompt_embeds
```

---
level: 2
---

<iframe frameborder="0" scrolling="no" class="emg-iframe-text-inputs" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Funderstand-stable-diffusion-slidev-notebooks%2Fblob%2Fmain%2Fembed%2Fch5-text_inputs.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>


<style>
	.emg-iframe-text-inputs {
		transform: scale(0.9) translate(-50%, -50%); /* Apply both transformations */
		transform-origin: top left;
		position: absolute;
		top: 50%;
		left: 50%;
		width: 100%;
		height: 100%;
	}
</style>

---
level: 2
---

<iframe frameborder="0" scrolling="no" class="emg-iframe-prompt-embeds" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Funderstand-stable-diffusion-slidev-notebooks%2Fblob%2Fmain%2Fembed%2Fch5-prompt_embeds.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<style>
	.emg-iframe-prompt-embeds {
		transform: scale(0.8) translate(-50%, -50%);
		transform-origin: top left;
		position: absolute;
		top: 57%;
		left: 50%;
		width: 100%;
		height: 130%;
	}
</style>

---
level: 2
layout: center
---

<iframe frameborder="0" scrolling="yes" class="overflow-scroll emg-iframe-play-prompt-embeds" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fnotebooks%2Fch0.0.2_Play_prompt_embeds.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<style>
	.emg-iframe-play-prompt-embeds {
		transform: scale(0.5) translate(-50%, -50%); /* Apply both transformations */
		transform-origin: top left;
		position: absolute;
		top: 50%;
		left: 50%;
		width: 100%;
		height: 160%;
	}
</style>

<!--
まるでWord2Vec
-->

---
layout: cover
title: "Step 2: get_latent"
background: /backgrounds/scheduler.webp
---

# Step 2: get_latent

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Scheduler, flat vector illustration, best quality, high resolution</p>

---
level: 2
layout: center
---

Step 2: get_latent
# Generate random tensor of 1/8 size

---
level: 2
layout: center
---

Necessities
# torch.randn

---
level: 2
layout: custom-two-cols
leftPercent: 0.4
---

<h1 class="!text-8.3">Step 2: get_latent</h1>
<p>Read the code and understand the whole flow</p>

<v-clicks every="1">

- L63: Generate random tensor of 1/8 size

<img src="/exps/latent.webp" class="mt-5 h-48 object-contain" />

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L59-L65](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L59-L65)


```python {all|63|all}{lines:true,startLine:59,at:1,style:'--slidev-code-font-size: 1rem; --slidev-code-line-height: 1.5;'}
def get_latent(self, width: int, height: int):
	"""
	Generate a random initial latent tensor to start the diffusion process.
	"""
	return torch.randn((4, width // 8, height // 8)).to(
		device=self.device, dtype=self.dtype
	)
```

---
layout: cover
title: "Step 3: denoise"
background: /backgrounds/unet.webp
---

# Step 3: denoise

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: UNet, watercolor painting, detailed, brush strokes, best quality, high resolution</p>

---
level: 2
layout: center
---

Step 3: denoise
# Denosing by using Scheduler and UNet.

---
level: 2
layout: center
---

<img src="/exps/denoised_latents_with_index.webp" class="h-96 object-contain mr-auto ml-auto" />

[<mdi-github-circle />understand-stable-diffusion-slidev-notebooks/denoise.ipynb](https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/denoise.ipynb)

---
level: 2
layout: center
---

<img src="/exps/decoded_images_with_index.webp" class="h-100 object-contain mr-auto ml-auto" />

[<mdi-github-circle />understand-stable-diffusion-slidev-notebooks/denoise.ipynb](https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/denoise.ipynb)

---
level: 2
layout: center
---

Necessities
# [<mdi-github-circle />scheduler.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/scheduler.py)
# [<mdi-github-circle />unet.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/unet.py)

---
level: 2
layout: center
---

Step 3: denoise
# Aside from Necessities, the whole flow

---
level: 2
layout: custom-two-cols
leftPercent: 0.5
transition: fade
---

# Step 3: denoise
Where are Necessities used?

<v-clicks every="1">

- L86: UNet

- L91: Scheduler

</v-clicks>

::right::


[<mdi-github-circle />pipeline.py#L75-L93](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L75-L93)

```python {all|86|86,91}{lines:true,startLine:75,at:1}
@torch.no_grad()
def denoise(self, latents, prompt_embeds, num_inference_steps=50, guidance_scale=7.5):
	"""
	Iteratively denoise the latent space using the diffusion model to produce an image.
	"""
	timesteps, num_inference_steps = self.retrieve_timesteps(num_inference_steps)

	for t in timesteps:
		latent_model_input = torch.cat([latents] * 2)
		
		# Predict the noise residual for the current timestep
		noise_residual = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)
		uncond_residual, text_cond_residual = noise_residual.chunk(2)
		guided_noise_residual = uncond_residual + guidance_scale * (text_cond_residual - uncond_residual)

		# Update latents by reversing the diffusion process for the current timestep
		latents = self.scheduler.step(guided_noise_residual, t, latents)[0]

	return latents
```

---
level: 2
layout: custom-two-cols
leftPercent: 0.5
transition: fade
---

# Step 3: denoise
Where are Necessities used?

- L86: UNet2DConditionModel

- L91: DDIMScheduler


::right::

[<mdi-github-circle />pipeline.py#L21-L39](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py#L21-L39)

```python {6,7}{lines:false,at:1}
@classmethod
def from_pretrained(cls, model_name, device=torch.device("cuda"), dtype=torch.float16):
	# Ommit comments
	tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
	text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
	scheduler = PareDDIMScheduler.from_config(model_name, subfolder="scheduler")
	unet = PareUNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
	vae = PareAutoencoderKL.from_pretrained(model_name, subfolder="vae")
	return cls(tokenizer, text_encoder, scheduler, unet, vae, device, dtype)
```

[<mdi-github-circle />pipeline.py#L82-L93](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L82-L93)

```python {86,91}{lines:true,startLine:82,at:1}
	for t in timesteps:
		latent_model_input = torch.cat([latents] * 2)
		
		# Predict the noise residual for the current timestep
		noise_residual = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)
		uncond_residual, text_cond_residual = noise_residual.chunk(2)
		guided_noise_residual = uncond_residual + guidance_scale * (text_cond_residual - uncond_residual)

		# Update latents by reversing the diffusion process for the current timestep
		latents = self.scheduler.step(guided_noise_residual, t, latents)[0]

	return latents
```

---
level: 2
layout: custom-two-cols
leftPercent: 0.5
---

# Step 3: denoise
Read the code and understand the whole flow

<v-clicks every="1">

- L80: Acquisition of timesteps using Scheduler<br />(<span class="text-sm">Scheduler will be described later</span>)

- L82: timesteps length loop<br />(<span class="text-sm">timesteps length = num_inference_steps</span>)

- L86: Denose by UNet<br />(<span class="text-sm">UNet will be described later</span>)

- L88: Calculate how much considering the prompt<br />(<span class="text-3">Reference: 
Jonathan Ho, Tim Salimans: “Classifier-Free Diffusion Guidance”, 2022; <a href='http://arxiv.org/abs/2207.12598'>arXiv:2207.12598</a>.</span>)

- L91: The strength of the deny is determined by Scheduler.

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L82-L93](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L82-L93)

```python {all|80|82|86|88|91|all}{lines:true,startLine:75,at:1}
@torch.no_grad()
def denoise(self, latents, prompt_embeds, num_inference_steps=50, guidance_scale=7.5):
	"""
	Iteratively denoise the latent space using the diffusion model to produce an image.
	"""
	timesteps, num_inference_steps = self.retrieve_timesteps(num_inference_steps)

	for t in timesteps:
		latent_model_input = torch.cat([latents] * 2)
		
		# Predict the noise residual for the current timestep
		noise_residual = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)
		uncond_residual, text_cond_residual = noise_residual.chunk(2)
		guided_noise_residual = uncond_residual + guidance_scale * (text_cond_residual - uncond_residual)

		# Update latents by reversing the diffusion process for the current timestep
		latents = self.scheduler.step(guided_noise_residual, t, latents)[0]

	return latents
```

---
level: 2
---

# <span class="text-3xl">[<mdi-github-circle />scheduler.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/scheduler.py)</span>
Determine the strength of denoising

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fscheduler.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
layout: custom-two-cols
leftPercent: 0.5
---

# Scheduler

<v-clicks every="1">

- L49: Get alpha_prod_t(0~1.0)<br />(<span class="text-sm">Indicates how much the original data is retained.</span>)

- L50: Get alpha_prod_t_prev(0~1.0)

- L52: alpha_prod_t + beta_prod_t = 1

- L53: Estimate the original sample from the current sample and model output.

- L54: Calculate the estimated value of the added noise.

- L56: Calculate the direction to restore it to the original image.

- L57: Calculate the sample that goes one step further in the deming by combining the estimated original sample and the update direction.

</v-clicks>

::right::

[<mdi-github-circle />scheduler.py#L40-L59](https://github.com/masaishi/parediffusers/blob/17e8ece5e6104fbec34d64c4d87545f340b0ea50/src/parediffusers/scheduler.py#L40-L59)

```python {all|49|50|52|53|54|56|57|all}{lines:true, at:1, startLine:40, style:'--slidev-code-font-size: 0.7rem; --slidev-code-line-height: 1.5;'}
def step(
	self,
	model_output: torch.FloatTensor,
	timestep: int,
	sample: torch.FloatTensor,
) -> list:
	"""Perform a single step of denoising in the diffusion process."""
	prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

	alpha_prod_t = self.alphas_cumprod[timestep]
	alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

	beta_prod_t = 1 - alpha_prod_t
	pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
	pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

	pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
	prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

	return prev_sample, pred_original_sample
```

---
level: 2
---

[<mdi-github-circle />scheduler.py#L40-L59](https://github.com/masaishi/parediffusers/blob/17e8ece5e6104fbec34d64c4d87545f340b0ea50/src/parediffusers/scheduler.py#L40-L59)

```python {all|49|49,50}{lines:true, startLine:40}
def step(
	self,
	model_output: torch.FloatTensor,
	timestep: int,
	sample: torch.FloatTensor,
) -> list:
	"""Perform a single step of denoising in the diffusion process."""
	prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

	alpha_prod_t = self.alphas_cumprod[timestep]
	alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

	beta_prod_t = 1 - alpha_prod_t
	pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
	pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

	pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
	prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

	return prev_sample, pred_original_sample
```

---
level: 2
layout: center
transition: fade
---

<div class="flex content-around gap-6">

<img src="/exps/alpha_prod_t.webp" class="h-64 object-contain ml-auto mr-auto" />

<img src="/exps/alpha_prod_t_prev.webp" class="h-64 object-contain ml-auto mr-auto" />

</div>

<p class="text-center">
<a src="https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/scheduler.ipynb"><mdi-github-circle />understand-stable-diffusion-slidev-notebooks/scheduler.ipynb</a>
</p>

---
level: 2
layout: center
transition: fade
---

<div class="flex content-around gap-6">
<h1 class="!text-16 !mt-auto !mb-auto">−</h1>
<img src="/exps/alpha_prod_t.webp" class="h-64 object-contain ml-auto mr-auto" />
<h1 class="!text-16 !mt-auto !mb-auto">+</h1>
<img src="/exps/alpha_prod_t_prev.webp" class="h-64 object-contain ml-auto mr-auto" />
</div>

<p class="text-center">
<a src="https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/scheduler.ipynb"><mdi-github-circle />understand-stable-diffusion-slidev-notebooks/scheduler.ipynb</a>
</p>

---
level: 2
layout: center
---

<div class="flex content-around gap-6">
<img src="/exps/alpha_diff.webp" class="h-64 object-contain ml-auto mr-auto" />

</div>

<p class="text-center">
<a src="https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/scheduler.ipynb"><mdi-github-circle />understand-stable-diffusion-slidev-notebooks/scheduler.ipynb</a>
</p>

---
level: 2
---

<iframe frameborder="0" scrolling="no" class="scale-40 -translate-y-1/2 absolute top-50% right-25% w-full h-240%" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Funderstand-stable-diffusion-slidev-notebooks%2Fblob%2Fmain%2Fembed%2Fwithout_scheduler.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<iframe frameborder="0" scrolling="no" class="scale-40 -translate-y-1/2 absolute top-54% left-25% w-full h-240%" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Funderstand-stable-diffusion-slidev-notebooks%2Fblob%2F606a033780f0c9aa0681fd1468f91f3961a73a3f%2Fembed%2Fwith_scheduler.ipynb&style=github&type=ipynb&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<!--
I feel like I'm learning and comparing it without using Scheduler.
Still, you can generate an image of contours. However, you can clearly see that without the Scheduler, a beautiful image has not been generated.
-->

---
level: 2
layout: center
---

<h1 class="mb-0">I don't have any idea why <code>ratio = 1.5</code> looks good.</h1>

<img class="h-100 object-contain -mb-5 ml-auto mr-auto" src="/exps/custom_denoise_different_ratio.webp" />

<p class="text-center">
	<a src="https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/scheduler_necessity.ipynb">
		<mdi-github-circle />understand-stable-diffusion-slidev-notebooks/scheduler_necessity.ipynb
	</a>
</p>

---
level: 2
---

# [<mdi-github-circle />unet.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/unet.py)
Using for Denoising

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Funet.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
layout: image
image: /images/unet-figure.webp
backgroundSize: 70%
class: 'text-black'
---

<!-- Reference -->
<p class="text-black text-xs abs-bl w-full mb-6 text-center">
Olaf Ronneberger, Philipp Fischer, Thomas Brox: “U-Net: Convolutional Networks for Biomedical Image Segmentation”, 2015; <a href='http://arxiv.org/abs/1505.04597'>arXiv:1505.04597</a>.
</p>

---
level: 2
---

# Create UNet using Resnet and Transformer

<iframe frameborder="0" scrolling="yes" class="emg-res-transformer" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2F675b3fdaf4435e9982f92ff933f78db64f16a980%2Fsrc%2Fparediffusers%2Fmodels%2Funet_2d_blocks.py%23L114-L141&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

<style>
	.emg-res-transformer {
		transform: scale(0.68) translate(-50%, -50%);
		transform-origin: top left;
		position: absolute;
		top: 63%;
		left: 50%;
		width: 100%;
		height: 130%;
	}
</style>

---
layout: cover
title: "Step 4: vae_decode"
background: /backgrounds/vae.webp
---

# Step 4: vae_decode

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: VAE, abstract style, highly detailed, colors and shapes</p>

---
level: 2
layout: center
---

Step 4: vae_decode
# Decode into the image with VAE

---
level: 2
layout: custom-two-cols
leftPercent: 0.4
---

<h1 class="!text-7.8">Step 4: vae_decode</h1>
<p>Read the code and understand the whole flow</p>

<v-clicks every="1">

- L112: Decode into the image with VAE

<img src="/exps/vae_decode.webp" class="mb-5 h-28 object-contain" />

- L113: Since we are learning and learning, it is necessary to reverse.

<img src="/exps/vae_denormalize.webp" class="mb-5 h-28 object-contain" />

- L114: Convert from tensor to Pil.image

</v-clicks>

::right::

[<mdi-github-circle />pipeline.py#L107-L105](https://github.com/masaishi/parediffusers/blob/035772c684ae8d16c7c908f185f6413b72658126/src/parediffusers/pipeline.py#L107-L115)

```python {all|112|113|114}{lines:true,startLine:107,at:1}
@torch.no_grad()
def vae_decode(self, latents):
	"""
	Decode the latent tensors using the VAE to produce an image.
	"""
	image = self.vae.decode(latents / self.vae.config.scaling_factor)[0]
	image = self.denormalize(image)
	image = self.tensor_to_image(image)
	return image
```

---
level: 2
layout: center
---

# Variational Autoencoder (VAE)

---
level: 2
---

# [<mdi-github-circle />vae.py](https://github.com/masaishi/parediffusers/blob/main/src/parediffusers/vae.py)

<iframe frameborder="0" scrolling="yes" class="overflow-scroll iframe-full-code" allow="clipboard-write" src="https://emgithub.com/iframe.html?target=https%3A%2F%2Fgithub.com%2Fmasaishi%2Fparediffusers%2Fblob%2Fmain%2Fsrc%2Fparediffusers%2Fvae.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></iframe>

---
level: 2
---

<div class="w-full h-90% flex content-around gap-2 justify-center items-center">
	<div>
		<p class="text-1 text-center !line-height-0 !mt-0 !mb-1.5">[1, 512, 64, 64]</p>
		<img src="/exps/vae_mid_0.webp" class="h-8 object-contain ml-auto mr-auto" />
	</div>
	<div>
		<p class="text-1 text-center !line-height-0 !mt-0 !mb-1.5">[1, 512, 64, 64]</p>
		<img src="/exps/vae_mid_1.webp" class="h-8 object-contain ml-auto mr-auto" />
	</div>
	<div>
		<p class="text-2 text-center !line-height-0 !mt-0 !mb-2">[1, 512, 128, 128]</p>
		<img src="/exps/vae_mid_2.webp" class="h-16 object-contain ml-auto mr-auto" />
	</div>
	<div>
		<p class="text-3 text-center !line-height-0 !mt-0 !mb-2.5">[1, 512, 256, 256]</p>
		<img src="/exps/vae_mid_3.webp" class="h-32 object-contain ml-auto mr-auto" />
	</div>
	<div>
		<p class="text-4 text-center !line-height-0 !mt-0 !mb-3">[1, 256, 512, 512]</p>
		<img src="/exps/vae_mid_4.webp" class="h-64 object-contain ml-auto mr-auto" />
	</div>
	<div>
		<p class="text-4 text-center !line-height-0 !mt-0 !mb-3">[1, 128, 512, 512]</p>
		<img src="/exps/vae_denormalize.webp" class="h-64 object-contain ml-auto mr-auto" />
	</div>
</div>

<div class="w-full flex justify-center items-center">
	<a class="" src="https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/vae.ipynb">
		<mdi-github-circle />understand-stable-diffusion-slidev-notebooks/vae.ipynb
	</a>
</div>

---
layout: cover
title: Conclusion
background: /backgrounds/summary.webp
---

# 9. Conclusion

<p class="text-xs abs-bl w-full mb-6 text-center">Prompt: Summary, long-exposure photography, masterpieces</p>

---
level: 2
layout: center
---

# It's fun to read the library code!

---
level: 2
layout: center
---

## A dissertation quoted everywhere in the library

[<mdi-github-circle />diffusers/.../pipeline.py](https://github.com/masaishi/parediffusers/blob/9e32721a4b1a63baf499517384e2a2acd9c08dae/src/parediffusers/pipeline.py)

<img src="/images/diffusers-code-arxiv.webp" class="mt-5 h-92 object-contain" />

---
level: 2
layout: center
---

Conclusion
# Step 1: Convert Prompt to Embedding.
# Step 2: Create random Latent.<br />
# Step 3: Denosing by using Scheduler and UNet.<br />
# Step 4: Decode to pixel space by VAE.

---
level: 1
layout: center
---

# Appendix

<Toc mode="onlyCurrentTree" maxDepth="2"></Toc>

---
level: 2
layout: center
---

# Other denoising samples

<div class="flex content-around gap-6">

<img src="/exps/text_cond_residuals_with_index.webp" class="h-64 object-contain ml-auto mr-auto" />

<img src="/exps/uncond_residuals_with_index.webp" class="h-64 object-contain ml-auto mr-auto" />

</div>

[<mdi-github-circle />understand-stable-diffusion-slidev-notebooks/denoise.ipynb](https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/denoise.ipynb)

---
level: 2
layout: center
---

# Other decorded denoising samples

<div class="flex content-around gap-6">

<img src="/exps/decoded_text_cond_residuals.webp" class="h-64 object-contain ml-auto mr-auto" />

<img src="/exps/decoded_uncond_residuals.webp" class="h-64 object-contain ml-auto mr-auto" />

</div>

[<mdi-github-circle />understand-stable-diffusion-slidev-notebooks/denoise.ipynb](https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/denoise.ipynb)

---
level: 2
layout: center
---

# UNet is really U form?

```python
init             torch.Size([2, 4, 64, 64])
conv_in          torch.Size([2, 320, 64, 64])

down_blocks_0    torch.Size([2, 320, 32, 32])
down_blocks_1    torch.Size([2, 640, 16, 16])
down_blocks_2    torch.Size([2, 1280, 8, 8])
down_blocks_3    torch.Size([2, 1280, 8, 8])

mid_block        torch.Size([2, 1280, 8, 8])

up_blocks0       torch.Size([2, 1280, 16, 16])
up_blocks1       torch.Size([2, 1280, 32, 32])
up_blocks2       torch.Size([2, 640, 64, 64])
up_blocks3       torch.Size([2, 320, 64, 64])

conv_out         torch.Size([2, 4, 64, 64])
```

[<mdi-github-circle />understand-stable-diffusion-slidev-notebooks/unet.ipynb](https://github.com/masaishi/understand-stable-diffusion-slidev-notebooks/blob/main/unet.ipynb)
