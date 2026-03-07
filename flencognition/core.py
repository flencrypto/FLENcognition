"""
FLENcognition core inference engine.

This module provides the :class:`FLENcognition` class, which wraps the
FireRed-OCR model and exposes a clean API for converting document images
to Markdown.  The model is loaded lazily – it is only downloaded and
initialised the first time :meth:`FLENcognition.process_image` (or
:meth:`FLENcognition.process_images`) is called.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .conv_for_infer import generate_conv

DEFAULT_MODEL_DIR = "FireRedTeam/FireRed-OCR"


class FLENcognition:
    """Document-image-to-Markdown OCR engine backed by FireRed-OCR.

    Parameters
    ----------
    model_dir:
        Hugging Face model repository or local path.  Defaults to the
        official ``"FireRedTeam/FireRed-OCR"`` checkpoint.
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``) or
        :class:`torch.device`.  When *None* (the default) the device is
        chosen automatically: CUDA if available, otherwise CPU.
    output_dir:
        Default directory used when saving Markdown files via
        ``save_markdown=True``.
    max_new_tokens:
        Maximum number of tokens the model may generate per image.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: Optional[str | torch.device] = None,
        output_dir: str = "md_output",
        max_new_tokens: int = 8192,
    ) -> None:
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the model and processor (idempotent)."""
        if self._model is not None:
            return

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print(f"🔥 Loading FLENcognition model from {self.model_dir}...")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        ).to(self.device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )
        self._model.eval()

    @property
    def model(self):
        """The underlying :class:`~transformers.PreTrainedModel` (lazy)."""
        self._load_model()
        return self._model

    @property
    def processor(self):
        """The associated :class:`~transformers.AutoProcessor` (lazy)."""
        self._load_model()
        return self._processor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_path: str,
        save_markdown: bool = False,
    ) -> dict:
        """Run OCR on a single document image.

        Parameters
        ----------
        image_path:
            Path to the image file to process.
        save_markdown:
            When *True* the recognised Markdown is written to
            ``<output_dir>/<basename>.md`` and the path is returned in
            the result dict.

        Returns
        -------
        dict
            ``{"markdown": str, "latex": str, "file": str | None}``

            * **markdown** – recognised text in Markdown format.
            * **latex** – same text with ``$$`` LaTeX delimiters instead of
              fenced code blocks (suitable for Gradio's Markdown renderer).
            * **file** – absolute path of the saved ``.md`` file, or *None*
              when ``save_markdown=False``.
        """
        messages = generate_conv({"image_path": image_path})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        latex_text = text.replace("```markdown", "$$").replace("```", "$$")

        file_path = None
        if save_markdown:
            os.makedirs(self.output_dir, exist_ok=True)
            basename = os.path.splitext(os.path.basename(image_path))[0]
            file_path = os.path.join(self.output_dir, f"{basename}.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

        return {"markdown": text, "latex": latex_text, "file": file_path}

    def process_images(
        self,
        image_paths: list[str],
        save_markdown: bool = False,
    ) -> list[dict]:
        """Run OCR on a list of document images.

        Parameters
        ----------
        image_paths:
            Paths to the image files to process.
        save_markdown:
            When *True* each result is written to ``<output_dir>/<basename>.md``.

        Returns
        -------
        list[dict]
            One result dict per image (see :meth:`process_image`).
        """
        return [self.process_image(path, save_markdown=save_markdown) for path in image_paths]
