"""Interactive REPL to "chat with" a WaveLM checkpoint (master plan Part 10).

WaveLM is a byte-level *document-continuation* model (V=256, no tokenizer),
trained on raw text (TinyStories / the Pile). It is NOT instruction-tuned: there
is no chat template, no role tokens, no notion of "turns". The honest interaction
model is therefore collaborative text continuation:

  - the running conversation is ONE byte stream (the "context");
  - each line you type is appended to the context verbatim (terminal labels like
    `you>` are NOT fed to the model -- they are display only);
  - the model streams a continuation of that byte stream, which is appended back
    into the context so the next turn sees everything so far;
  - the context is a sliding window capped at cfg.T bytes (1024 for the released
    d=2048 checkpoints). When the window fills, the oldest bytes drop off.

Generation reuses the streaming inference path in wave/infer.py
(WaveStreamer.step), whose logits are bit-identical to the parallel training
forward (test_stream_equals_parallel). Sampling (temperature / top-k / greedy)
mirrors wave.infer.generate exactly.

Runs on CPU by default so it can be used while a GPU training run is in progress.

Example:
    python scripts/chat_wave.py                       # longest run (Pile, 12000 steps)
    python scripts/chat_wave.py --ckpt checkpoints/wavelm_run4/best.pt
    python scripts/chat_wave.py --temperature 0.7 --top_k 40 --max_new 200
"""

from __future__ import annotations

import argparse
import codecs
import os
import random
import sys
from typing import List, Optional

# Make `import wave.*` resolve to the repo package, not this scripts/ dir or any
# similarly named site package (same shim as scripts/generate_wave.py).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(_here))

import torch
import torch.nn.functional as F

from wave.config import WaveConfig
from wave.model import WaveModel
from wave.infer import WaveStreamer
from wave.data import encode_str, decode_bytes


# -----------------------------------------------------------------------------
# Sampling (identical policy to wave.infer.generate, factored out for streaming)
# -----------------------------------------------------------------------------
def _sample_next(logits_row: torch.Tensor, temperature: float, top_k: Optional[int],
                 greedy: bool, gen: torch.Generator) -> int:
    """Pick the next byte id from one (V,) logit row. CPU float32 throughout."""
    lg = logits_row.float().cpu()
    if greedy:
        return int(lg.argmax())
    lg = lg / max(1e-6, temperature)
    if top_k is not None and top_k > 0:
        v, ix = lg.topk(min(top_k, lg.numel()))
        mask = torch.full_like(lg, float("-inf"))
        mask[ix] = v
        lg = mask
    probs = F.softmax(lg, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen))


# -----------------------------------------------------------------------------
# Chat session: a sliding byte-window over the WaveLM context length cfg.T
# -----------------------------------------------------------------------------
class ChatSession:
    """Holds the running conversation as a byte stream and drives generation.

    Correctness note: the WaveStreamer cache is append-only up to cfg.T positions
    (it raises once full). To support a conversation longer than the window we
    keep the full transcript in `self.context` but, every generation call, build
    a *fresh* streamer primed only on the trailing window that fits. This re-feeds
    the window each turn (O(window * n_sub * delta * d)); for the small T=1024
    window this is a fraction of a second on CPU and keeps the logic provably
    correct rather than tracking incremental cache eviction.
    """

    def __init__(self, model: WaveModel, device: str):
        self.model = model
        self.cfg: WaveConfig = model.cfg
        self.device = device
        self.context: List[int] = []   # full transcript (unbounded; for /save & display)

    # -- context management ---------------------------------------------------
    def append_text(self, text: str) -> None:
        """Append user text to the context, inserting a separating space only if
        needed so successive turns/words do not mash together."""
        new = encode_str(text)
        if self.context and new:
            prev_ws = self.context[-1] in (0x20, 0x09, 0x0A, 0x0D)
            new_ws = new[0] in (0x20, 0x09, 0x0A, 0x0D)
            if not prev_ws and not new_ws:
                self.context.append(0x20)  # single space
        self.context.extend(new)

    def reset(self) -> None:
        self.context = []

    def _window(self, max_new: int) -> List[int]:
        """Trailing slice of the context that leaves room to generate max_new
        bytes inside the cfg.T cache (need len(prompt) + max_new <= cfg.T)."""
        budget = self.cfg.T - max_new - 1
        if budget < 1:
            budget = 1
        return self.context[-budget:]

    # -- generation -----------------------------------------------------------
    @torch.no_grad()
    def generate_stream(self, max_new: int, temperature: float, top_k: Optional[int],
                        greedy: bool, seed: int):
        """Yield (byte_id, decoded_str_fragment) as the model streams a
        continuation of the current context. Generated bytes are appended to the
        context as they are produced. UTF-8 is decoded incrementally so multi-byte
        characters are not split across yields."""
        self.model.eval()
        streamer = WaveStreamer(self.model, batch_size=1, device=self.device)

        prompt = self._window(max_new)
        logits = None
        for b in prompt:
            logits = streamer.step(torch.tensor([b], device=self.device, dtype=torch.long))

        gen = torch.Generator(device="cpu").manual_seed(seed)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        for _ in range(max_new):
            if streamer.length >= self.cfg.T:   # cache exhausted (window == T)
                break
            if logits is None:                  # empty context: seed with newline byte
                logits = streamer.step(torch.tensor([0x0A], device=self.device, dtype=torch.long))
            nxt = _sample_next(logits[0], temperature, top_k, greedy, gen)
            self.context.append(nxt)
            frag = decoder.decode(bytes([nxt]))
            if frag:
                yield nxt, frag
            if streamer.length >= self.cfg.T:
                break
            logits = streamer.step(torch.tensor([nxt], device=self.device, dtype=torch.long))

        tail = decoder.decode(b"", final=True)  # flush any dangling partial char
        if tail:
            yield -1, tail


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------
def load_model(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}\n"
            f"Available .pt checkpoints under checkpoints/ (the 6000-step run3 "
            f"TinyStories model was not saved to disk; the longest saved run is "
            f"wavelm_pile_42 at 12000 steps)."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: WaveConfig = ckpt["cfg"]
    model = WaveModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt


# -----------------------------------------------------------------------------
# REPL
# -----------------------------------------------------------------------------
HELP = """\
Commands (everything else is appended to the context and continued by the model):
  /help                 show this help
  /reset                clear the conversation context
  /context              show context length (bytes / cfg.T) and a tail preview
  /temp <float>         set sampling temperature   (e.g. /temp 0.7)
  /topk <int>           set top-k   (0 disables top-k)
  /max  <int>           set max new bytes per turn
  /seed <int|random>    fix the RNG seed, or 'random' for a fresh seed each turn
  /greedy <on|off>      toggle greedy (argmax) decoding
  /save <path>          write the full transcript (decoded UTF-8) to a file
  /params               show current sampling parameters
  /quit  (or /exit)     leave
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat (text-continuation) with a WaveLM checkpoint.")
    ap.add_argument("--ckpt", default="checkpoints/wavelm_pile_42/best.pt",
                    help="checkpoint .pt (default: longest run = Pile, 12000 steps)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--max_new", type=int, default=160, help="bytes generated per turn")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40, help="0 disables top-k")
    ap.add_argument("--greedy", action="store_true", help="argmax decoding (ignores temp/top_k)")
    ap.add_argument("--seed", type=int, default=None,
                    help="fixed RNG seed; default: a fresh random seed each turn")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    model, cfg, ckpt = load_model(args.ckpt, args.device)

    # mutable session params (editable via slash-commands)
    temperature = args.temperature
    top_k = args.top_k
    max_new = args.max_new
    greedy = args.greedy
    fixed_seed = args.seed

    print("=" * 78)
    print(f"WaveLM chat  |  {args.ckpt}")
    print(f"  step={ckpt.get('step')}  val_bpb={ckpt.get('val_bpb'):.4f}"
          if isinstance(ckpt.get('val_bpb'), float) else f"  step={ckpt.get('step')}")
    print(f"  d={cfg.d}  T={cfg.T} (context window in bytes)  n_sub={cfg.n_sub}  "
          f"mix_rank={getattr(cfg, 'mix_rank', 0)}  device={args.device}")
    print("  Byte-level continuation model: it CONTINUES your text, it does not")
    print("  follow instructions. Type text and it writes more. /help for commands.")
    print("=" * 78)

    session = ChatSession(model, args.device)

    def next_seed() -> int:
        return fixed_seed if fixed_seed is not None else random.randint(0, 2**31 - 1)

    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        stripped = line.strip()
        if stripped.startswith("/"):
            parts = stripped.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/quit", "/exit"):
                print("bye.")
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/reset":
                session.reset()
                print("[context cleared]")
            elif cmd == "/context":
                n = len(session.context)
                tail = decode_bytes(session.context[-200:])
                print(f"[context: {n} bytes total; window cap = {cfg.T}]")
                print(f"...{tail!r}" if n else "[empty]")
            elif cmd == "/temp":
                try:
                    temperature = float(arg); print(f"[temperature = {temperature}]")
                except ValueError:
                    print("usage: /temp <float>")
            elif cmd == "/topk":
                try:
                    top_k = int(arg); print(f"[top_k = {top_k}{' (disabled)' if top_k <= 0 else ''}]")
                except ValueError:
                    print("usage: /topk <int>")
            elif cmd == "/max":
                try:
                    max_new = max(1, int(arg)); print(f"[max_new = {max_new}]")
                except ValueError:
                    print("usage: /max <int>")
            elif cmd == "/seed":
                if arg.lower() in ("random", "rand", "none", ""):
                    fixed_seed = None; print("[seed = random each turn]")
                else:
                    try:
                        fixed_seed = int(arg); print(f"[seed = {fixed_seed}]")
                    except ValueError:
                        print("usage: /seed <int|random>")
            elif cmd == "/greedy":
                greedy = arg.lower() in ("on", "true", "1", "yes")
                print(f"[greedy = {greedy}]")
            elif cmd == "/params":
                print(f"[temperature={temperature} top_k={top_k} max_new={max_new} "
                      f"greedy={greedy} seed={'random' if fixed_seed is None else fixed_seed}]")
            elif cmd == "/save":
                path = arg or "wavelm_chat_transcript.txt"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(decode_bytes(session.context))
                print(f"[saved {len(session.context)} bytes -> {path}]")
            else:
                print(f"unknown command {cmd!r}; /help for the list")
            continue

        # Non-command line: append to context (empty line => just continue).
        if line:
            session.append_text(line)
        elif not session.context:
            print("[empty context — type something for the model to continue]")
            continue

        print("wave>", end=" ", flush=True)
        for _bid, frag in session.generate_stream(max_new, temperature,
                                                   (None if greedy else top_k),
                                                   greedy, next_seed()):
            print(frag, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
