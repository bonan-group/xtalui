# Example Structures

These CIF files are generated with ASE and are intended as quick smoke-test inputs for `xtal`.

- `silicon_diamond.cif`: Si in the diamond structure
- `gaas_zincblende.cif`: GaAs zincblende
- `diamond_cubic.cif`: elemental carbon diamond
- `graphite_hexagonal.cif`: hexagonal graphite
- `al_fcc.cif`: Al face-centered cubic
- `cu_fcc.cif`: Cu face-centered cubic
- `fe_bcc.cif`: Fe body-centered cubic
- `mg_hcp.cif`: Mg hexagonal close packed
- `nacl_rocksalt.cif`: NaCl rocksalt
- `sic_zincblende.cif`: SiC zincblende

Launch any of them with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run xtal examples/silicon_diamond.cif
```
