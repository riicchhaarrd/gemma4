# gemma4

Single-file Gemma 4 inference experiments for GGUF models.

## WebGPU Demo

Try the browser build on GitHub Pages:

https://riicchhaarrd.github.io/gemma4.c/

The Pages build can load the tested model:

https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q4_K_XL.gguf?download=true

The page does not download the model until you click **Load Default Model** or choose a local `.gguf`.

That GGUF is about 3 GB. The first cached load can take several minutes because the browser downloads the whole model before uploading weights to WebGPU. After that, refreshes can reuse the browser cache. You can also choose the streaming mode to fetch model ranges without storing the whole file first.

Add `?autoload=1` to the Pages URL only if you intentionally want the default model load to start automatically.

## Native C Runner

Build on Linux:

```sh
gcc -O2 -o gemma4 gemma4.c -lm -lpthread
```

Run:

```sh
./gemma4 gemma-4-E2B-it-UD-Q4_K_XL.gguf -p "Hello" -n 128 -t 0.7
```

## Local WebGPU

Serve the repository from localhost, then open `gemma4-webgpu.html`:

```sh
python3 -m http.server 8000
```

Then visit:

```text
http://localhost:8000/gemma4-webgpu.html
```

WebGPU requires a compatible browser and GPU. Chrome or Edge on a recent desktop GPU is the most likely path to success.

## Tested Model

- `gemma-4-E2B-it-UD-Q4_K_XL.gguf`
