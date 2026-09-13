---
name: pdb-lookup
description: |
  Look up protein structures in the RCSB PDB through its public JSON APIs (no key):
  search entries by text or sequence, read an entry's title, method, resolution and
  chains, and download sequences or coordinates. Use this instead of web_search for
  any question about a PDB ID, a structure of a target, or an antibody-antigen complex.
homepage: https://data.rcsb.org/
---

# PDB lookup

Fixed, keyless endpoints. Fetch them with `web_fetch` (JSON comes back as is);
never web-search for a PDB ID when one of these answers directly.

## Find entries

Search API, GET with the query as URL-encoded JSON in `?json=`:

```
https://search.rcsb.org/rcsbsearch/v2/query?json={"query":{"type":"terminal","service":"full_text","parameters":{"value":"PD-1 pembrolizumab"}},"return_type":"entry","request_options":{"paginate":{"start":0,"rows":10}}}
```

Returns `result_set[].identifier` (PDB IDs) with scores. Useful services:
`full_text` (free text), `text` with `attribute` (e.g.
`rcsb_entity_source_organism.taxonomy_lineage.name`, `struct.title`),
`sequence` (`{"value":"<sequence>","identity_cutoff":0.9,"evalue_cutoff":0.1,"sequence_type":"protein"}`).
`return_type` can be `entry` or `polymer_entity`.

## Read one entry

```
https://data.rcsb.org/rest/v1/core/entry/5GGS
```

Read `struct.title`, `exptl[].method`, `rcsb_entry_info.resolution_combined`,
`rcsb_entry_info.polymer_entity_count`, `citation[0]` (title, journal, year,
`pdbx_database_id_DOI`), `rcsb_entry_container_identifiers.polymer_entity_ids`.

Per chain: `https://data.rcsb.org/rest/v1/core/polymer_entity/5GGS/1` gives
`entity_poly.pdbx_seq_one_letter_code` (sequence),
`rcsb_polymer_entity.pdbx_description`, `rcsb_entity_source_organism[].scientific_name`
and `rcsb_polymer_entity_container_identifiers.auth_asym_ids` (chain letters).

## Sequences and coordinates

```
https://www.rcsb.org/fasta/entry/5GGS          # FASTA, one record per entity with chain letters
https://files.rcsb.org/download/5GGS.cif       # mmCIF coordinates (large: use bash + curl -o to save it)
https://files.rcsb.org/download/5GGS.pdb       # legacy PDB format when available
```

## Keep results small

Ask the Data API only for the entry or entity you need, and quote the fields
above rather than the whole document. A structure question is usually answered
by one search plus one entry read.
