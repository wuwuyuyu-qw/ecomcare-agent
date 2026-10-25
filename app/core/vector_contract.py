"""The current knowledge collection uses bge-m3 dense vectors.

Changing this value requires a separately rebuilt collection with the matching
embedding model; it is not an automatic migration of existing Milvus data.
"""
import math

DENSE_DIMENSION = 1024
FLOAT32_MAX = 3.4028234663852886e38


def validate_embedding_batch(data: list, count: int) -> list[list[float]]:
    """Reorder by provider index; reject missing, malformed or unusable vectors."""
    if not isinstance(data, list) or len(data) != count:
        raise ValueError("嵌入响应数量与输入不一致")
    ordered: list[list[float] | None] = [None] * count
    for item in data:
        index = getattr(item, "index", None)
        vector = getattr(item, "embedding", None)
        if (not isinstance(index, int) or isinstance(index, bool)
                or not 0 <= index < count or ordered[index] is not None):
            raise ValueError("嵌入响应索引缺失、重复或越界")
        if not isinstance(vector, list) or len(vector) != DENSE_DIMENSION:
            raise ValueError(f"嵌入维度必须与当前知识库一致：{DENSE_DIMENSION}")
        numbers = []
        for value in vector:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("嵌入向量包含非数值元素")
            try:
                number = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("嵌入向量元素超出浮点范围") from exc
            if not math.isfinite(number) or abs(number) > FLOAT32_MAX:
                raise ValueError("嵌入向量元素超出 Milvus FLOAT_VECTOR 的有限 float32 范围")
            numbers.append(number)
        if not any(numbers):
            raise ValueError("余弦检索不能使用零向量")
        ordered[index] = numbers
    # Unique in-range indices and an exact count guarantee every slot is filled.
    if any(vector is None for vector in ordered):
        raise ValueError("嵌入响应缺少输入对应的向量")
    return [vector for vector in ordered if vector is not None]
