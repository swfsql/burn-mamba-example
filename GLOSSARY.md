# burn-mamba-example

A demo app that runs pretrained Mamba checkpoints to generate text, natively
or in the browser.

This glossary defines only the terms of this app. The other terms come from two
glossaries, and this app uses them with the same meaning:

- The terms of the blocks (plant, SSD path, MIMO rank, rotation, …) are in the
  [burn-mamba glossary](../burn-mamba/GLOSSARY.md).
- The terms of the containers (family, block, layer, cache, forward, step,
  chunk, …) are in the [burn-stack glossary](../burn-stack/GLOSSARY.md).

## Language

### Checkpoints

**Checkpoint**:
One published set of trained weights, with the topology that the weights need.
Each checkpoint has one family. Two checkpoints can have the same family, as the
Mamba-3 SISO and MIMO checkpoints do.
_Avoid_: model (alone), variant, version

**Preferred checkpoint**:
The checkpoint with the highest priority among the checkpoints that a build
includes. The build runs it, unless the user selects another checkpoint.
_Avoid_: default model

**Manifest**:
The tensor names and shapes of a checkpoint, copied into the repo from the
header of its weights file.
_Avoid_: index

**Key remapping**:
The rules that change each tensor name of a checkpoint into a parameter path
of the network.
_Avoid_: key mapping, renaming

### Tokenizers

**Tokenizer pipeline**:
The normalizer, the pre-tokenizer, the model and the post-processor that a
tokenizer file declares. The app supports two pipelines: gpt-neox and
Llama-3.1.
_Avoid_: tokenizer config

### Run modes

**Sequential run**:
A generation in which each token goes through one step. This includes each
token of the prompt.
_Avoid_: recurrent mode

**Parallel run**:
A generation that runs one forward over the whole prefix for each new token,
with no cache.
_Avoid_: chunked mode

**Run agreement**:
The same text from a sequential run and a parallel run of one prompt. It shows
that a backend gives the same values in the two run modes. It does not show that
the values are correct.

### The browser app

**Asset**:
One file that the browser app needs for a checkpoint: the tokenizer file or the
weights file. The user can download, delete, load and unload each asset
separately.
_Avoid_: resource

**Download chunk**:
A 10MB byte range of one remote file. The browser downloads and caches a file
one download chunk at a time.
_Avoid_: chunk (alone, which is a part of a sequence)
