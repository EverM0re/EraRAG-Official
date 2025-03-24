from Core.Chunk.ChunkFactory import register_chunking_method

@register_chunking_method("chunking_by_token_size")
async def chunking_by_token_size(tokens_list: list[list[int]], doc_keys, tiktoken_model, overlap_token_size=128,
                                 max_token_size=1024, title_list=None):
    """
    Chunking by token size.

    This method will split the tokens list into chunks given a max token size.

    :return: A list of chunks.
    :rtype: list
    """

    results = []
    idx_cnt = 0
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start: start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "index": idx_cnt,
                    "doc_id": doc_keys[index],
                    "title": title_list[index],
                }
            )
            idx_cnt += 1
   
    return results