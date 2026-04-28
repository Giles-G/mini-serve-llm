"""Tokenizer 适配器

对 HuggingFace tokenizer 进行封装，提供统一的 encode/decode/build_prompt 接口，
屏蔽不同 tokenizer 的差异。
"""


class TokenizerAdapter:
    """Tokenizer 适配器，封装编码、解码和 prompt 构建逻辑

    Attributes:
        tokenizer: HuggingFace tokenizer 实例
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        """将文本编码为 token id 列表

        Args:
            text: 输入文本

        Returns:
            token id 列表（不含 special tokens）
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """将 token id 列表解码为文本

        Args:
            token_ids: token id 列表

        Returns:
            解码后的文本（跳过 special tokens）
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def build_prompt(self, user_text: str) -> str:
        """构建完整的对话 prompt

        优先使用 tokenizer 的 chat template（如 <|im_start|> 格式），
        如果不支持则退化成原始文本。

        Args:
            user_text: 用户输入的原始文本

        Returns:
            构建好的完整 prompt 字符串
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_text}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,          # 返回文本而非 token ids
                    add_generation_prompt=True,  # 追加 assistant 开头标记
                )
                return text
            except Exception:
                # chat template 失败时退化为原始文本
                return user_text
        return user_text
