class TokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def build_prompt(self, user_text: str) -> str:
        """
        优先使用 chat template。
        如果 tokenizer 不支持，就退化成普通文本。
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_text}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return text
            except Exception:
                return user_text
        return user_text
