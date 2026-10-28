"""Write compressed MLA rows without a dynamic DCP ownership selection."""

import triton as tr
import triton.language as tl


@tr.jit
def _store(
    X,
    K,
    LOC,
    PAGES,
    CACHE,
    X_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    LATENT: tl.constexpr,
    KEY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    INDIRECT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    loc = tl.load(LOC + row)
    owned = (loc >= 0) & (loc % CP_SIZE == CP_RANK)
    page = loc // PAGE_SIZE
    if INDIRECT:
        page = tl.load(PAGES + page, owned, 0)
    offset = (loc % PAGE_SIZE) // CP_SIZE
    x = tl.load(X + row * X_STRIDE + col, col < LATENT, 0)
    k = tl.load(K + row * K_STRIDE + col - LATENT, (col >= LATENT) & (col < LATENT + KEY), 0)
    value = tl.where(col < LATENT, x, k)
    tl.store(
        CACHE + page * PAGE_STRIDE + offset * TOKEN_STRIDE + col,
        value,
        owned & (col < LATENT + KEY),
    )


def store_mla(latent, key, locations, pages, cache, *, page_size, cp_size, cp_rank):
    _store[(len(locations),)](
        latent,
        key,
        locations,
        pages if pages is not None else locations,
        cache,
        latent.stride(0),
        key.stride(0),
        cache.stride(0),
        cache.stride(1),
        latent.shape[-1],
        key.shape[-1],
        page_size,
        cp_size,
        cp_rank,
        pages is not None,
        tr.next_power_of_2(latent.shape[-1] + key.shape[-1]),
        num_warps=4,
    )
