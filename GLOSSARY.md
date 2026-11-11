# burn-mamba-example

A demo app that runs pretrained Mamba checkpoints to generate text, natively
or in the browser.

This glossary defines only the terms of this app. The other terms come from two
glossaries, and this app uses them with the same meaning:

- The terms of the blocks (plant, SSD path, MIMO rank, rotation, …) are in the
  [burn-mamba glossary](../burn-mamba/GLOSSARY.md).
- The terms of the containers (family, block, layer, network, cache, forward,
  step, prefill, decode, chunk, …) are in the
  [burn-stack glossary](../burn-stack/GLOSSARY.md).

## Language

### Checkpoints

**Checkpoint**:
One published set of trained weights, with the topology that the weights need.
Each checkpoint has one family. Two checkpoints can have the same family, as the
Mamba-3 SISO and MIMO checkpoints do.
_Avoid_: model (alone), variant, version

**Topology**:
The shape of the network and the config of its block that the weights of a
checkpoint need.
_Avoid_: config (alone), architecture

**Checkpoint feature**:
A cargo feature that adds one checkpoint to a build. Its name is the id of the
checkpoint, for example `mamba2-130m`. It also adds the family feature of the
checkpoint.
_Avoid_: model feature

**Family feature**:
A cargo feature that adds the blocks of one family to a build, but no
checkpoint: `mamba1`, `mamba2` or `mamba3`.

**Preferred checkpoint**:
The checkpoint with the highest priority among the checkpoints that a build
includes. The build runs it, unless the user selects another checkpoint.
_Avoid_: default model

**Manifest**:
The names and shapes of the tensors of one checkpoint, copied into the repo
from the header of its weights file. A manifest holds the tensors outside the
layers and the tensors of the first layer. Each other layer repeats the first
layer.
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
A decode that also steps each token of the prompt, with no prefill.
_Avoid_: recurrent mode, inference mode

**Parallel run**:
A generation that runs one forward over the whole sequence so far for each new
token, with no cache.
_Avoid_: chunked mode, training mode

**Run agreement**:
The same text from a sequential run and a parallel run of one prompt. It shows
that a backend gives the same values in the two run modes. It does not show that
the values are correct.

**Reference backend**:
A backend whose text the app takes as correct: flex or ndarray. A run on another
backend is correct when its text is the text of a reference backend.

### Front-ends

**Front-end**:
One way to run the app: the native binary, the browser console or the browser
page. Each front-end has a feature: `native`, `console` or `yew`.
_Avoid_: target, UI (alone)

### Downloads

**Hub cache**:
The local copy of the files of the HuggingFace hub. Natively, it is a directory
that other HuggingFace tools can share. In the browser, it is IndexedDB.
_Avoid_: cache (alone, which is the state of a block)

**Download chunk**:
A 10MB byte range of one file of the hub. The hub cache of the browser holds a
file as its download chunks.
_Avoid_: chunk (alone, which is a part of a sequence)

**Fetch**:
To copy a file from the hub into the hub cache. The browser page fetches one
download chunk at a time.
_Avoid_: download (the browser page uses it for a save to disk)

**Plan**:
The download chunks of one file, in order, each marked as in the hub cache or
not. A fetch copies only the download chunks that the hub cache does not hold.
_Avoid_: download chunk list

### The browser page

**Asset**:
One file that the browser page needs for a checkpoint: the tokenizer file or the
weights file. The user can fetch, erase, load and unload each asset separately.
_Avoid_: resource, model data

**Erase**:
The opposite of a fetch. After an erase, the hub cache holds no part of the
asset.
_Avoid_: evict

**Load**:
To read a cached asset into memory, as bytes.
_Avoid_: read

**Build**:
To make the tokenizer or the network from the loaded bytes of an asset.

**Unload**:
To free the memory of an asset: its bytes, and the tokenizer or the network
that the page built from them.
