"""Training-format options for experiment E1.

The completed run trained on a stream with no document boundaries, no prompt
template and no prompt masking (see docs/AIVORA_TRAINING_PIPELINE_AUDIT.md).
Three measured consequences: `<|endoftext|>` never appeared in the data and the
model emitted it 0 times in 204 generations, answers ran to the token cap 98.5%
of the time, and model-only extraction accuracy was 0 of 25 - it continues text
rather than answering.

E1 tests whether those three choices, rather than the corpus size, are what
holds the model back. Everything here is OFF by default, so arm 1 of the
experiment reproduces the existing behaviour exactly and arm 2 differs only in
these three things.

    fmt = TrainingFormat(document_separator=True, template=True, mask_prompt=True)
    ids, mask = fmt.encode(record)
"""

from dataclasses import dataclass

from .tokenizer import get_encoding

# Matches models/model.py: F.cross_entropy(..., ignore_index=-1)
IGNORE_INDEX = -1

# One template, used when preparing data AND when prompting at inference. The
# serving path already uses exactly this string; under the old preparation the
# model had never seen it.
PROMPT_TEMPLATE = "Question: {prompt}\nAnswer:"


@dataclass
class TrainingFormat:
    document_separator: bool = False
    template: bool = False
    mask_prompt: bool = False

    def describe(self):
        return {"document_separator": self.document_separator,
                "template": self.template, "mask_prompt": self.mask_prompt,
                "prompt_template": PROMPT_TEMPLATE if self.template else None,
                "separator_token": "<|endoftext|>" if self.document_separator else None}

    def encode(self, record):
        """Tokenize one record into (ids, mask).

        mask[i] is 1 where the token should contribute to the loss and 0 where
        it should be ignored. With mask_prompt off, every token is 1 and the
        mask costs nothing but a byte per token.

        A record may carry `prompt`/`completion` (instruction-style) or just
        `text`. Only the former can be templated or masked; plain text is
        unchanged apart from the separator.
        """
        enc = get_encoding()
        prompt = (record.get("prompt") or "").strip()
        completion = (record.get("completion") or "").strip()

        if self.template and prompt and completion:
            prompt_text = PROMPT_TEMPLATE.format(prompt=prompt)
            prompt_ids = enc.encode_ordinary(prompt_text)
            completion_ids = enc.encode_ordinary(" " + completion)
            ids = prompt_ids + completion_ids
            if self.mask_prompt:
                # The model is not asked to predict the question it was given.
                mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            else:
                mask = [1] * len(ids)
        else:
            text = record.get("text")
            if not text:
                text = "\n".join(p for p in (prompt, completion) if p)
            ids = enc.encode_ordinary(text)
            mask = [1] * len(ids)

        if self.document_separator and ids:
            # Trained ON the separator, not merely separated by it: the model
            # has to learn that an answer ends.
            ids = ids + [enc.eot_token]
            mask = mask + [1]

        return ids, mask
