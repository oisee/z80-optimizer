# Draft Reply to Philipp Klaus Krause

## His points and our responses:

### 1. SDCC version (4.2.0 → 4.5.0)
**His point**: No CC changes since 4.2.0, but 4.5.0 has mos6502 backend.
**Response**: Thank you — we'll note that 4.5.0 confirms no CC changes (strengthens our comparison). We'd love to add mos6502 comparison with SDCC 4.5.0's new backend. We have a 6502 ISA definition in our superoptimizer already.

### 2. "Nanz compiler" citation
**His point**: No citation for the compiler we use.
**Response**: [NEED FROM MINZ: what's the public name, repo URL, any publication?]
- Options: cite as "MinZ, a whole-program compiler for Z80/6502. Source: https://github.com/..."
- Or: if no public repo yet, cite as "unpublished, available on request"

### 3. "17x smaller" terminology
**His point**: Mathematically confusing. "1x smaller" = size 0.
**Response**: He's absolutely right. Change to:
- "1/17th the size" or "5.9% of the original"
- Or: "17:1 size ratio"
- Or: "code is 94.1% smaller" (for the extreme case)
- General pattern: "N:1 size ratio" or "reduced to 1/Nth"

### 4. Multi-return comparison methodology
**His point**: What SDCC approach for multi-return? Pointer args vs struct?
**Response**: [NEED FROM MINZ: which SDCC approach was benchmarked?]
- If pointer args: say so explicitly
- If struct return: note hidden pointer overhead
- Best: show both and note the difference

### 5. Separate compilation / function pointers
**His point**: SDCC can't do PFCCO due to:
  a) Separate compilation (no whole-program view)
  b) Hand-written asm expects fixed CC
  c) Function pointers require fixed CC

**Response**: This is the key philosophical difference.
- MinZ is a whole-program compiler — it sees the entire call graph
- Function pointers: [NEED FROM MINZ: trampolines? fixed CC for fp-callable functions? forbid for small targets?]
- Static functions: Philipp himself notes SDCC could do this for static fns w/o address-taken — this is exactly our point! The benefit exists even in a C compiler, limited to that subset
- Our argument: on 8-bit targets, most code is whole-program anyway (no dynamic linking on Z80)

### 6. Small functions → inlining
**His point**: Small functions should be inlined, making CC irrelevant.
**Response**: Three-part answer:
  a) **Inlining and PFCCO are complementary, not competing**: inline tiny functions (1-3 ops), optimize CC for medium functions (4-20 ops) where inlining causes code bloat
  b) **Call-heavy architectures**: on Z80, CALL=17T, RET=10T = 27T overhead. For a 20T function body, inlining saves 27T but doubles code size at every call site. PFCCO saves 10-20T without any code duplication
  c) **Register pressure**: inlining increases register pressure at the call site. On Z80 with 7 registers, this can force spills that cost more than the call overhead saved
  d) **The "missing middle"**: SDCC inlines only `inline`-marked functions. There's a large class of functions too big to inline but small enough that CC overhead dominates. PFCCO targets exactly this class

## Tone
- Grateful, collegial
- Acknowledge all valid points (especially terminology fix)
- Don't be defensive about SDCC comparison
- Emphasize: whole-program vs separate compilation is a design choice, not a limitation of either approach
- Offer: happy to add SDCC 4.5.0 6502 comparison, send updated draft

## TODO
- [ ] Get MinZ team answers on points 2, 4, 5
- [ ] Fix "17x smaller" in paper
- [ ] Add SDCC 4.5.0 note
- [ ] Possibly add 6502 SDCC comparison
- [ ] Send updated draft with fixes
