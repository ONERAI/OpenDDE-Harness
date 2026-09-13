---
name: uniprot-lookup
description: |
  Look up protein sequences and annotations in UniProt through its public REST API
  (no key): find an accession by gene or protein name and organism, read a chosen set
  of fields (sequence, domains, signal peptide, glycosylation, PDB cross-references),
  or get the FASTA. Use this instead of web_search for any target sequence, domain
  boundary or accession question.
homepage: https://rest.uniprot.org/
---

# UniProt lookup

Fixed, keyless endpoints. Fetch them with `web_fetch`; JSON, TSV and FASTA
come back as is. Always ask for `fields` -- a whole entry is 60KB+ and gets
truncated, a field selection is a few hundred bytes.

## Find the accession

```
https://rest.uniprot.org/uniprotkb/search?query=gene:PDCD1+AND+organism_id:9606+AND+reviewed:true&fields=accession,protein_name,length&format=tsv&size=5
```

Query syntax: `gene:`, `protein_name:`, `organism_id:` (9606 human, 10090
mouse), `accession:`, `reviewed:true` (Swiss-Prot only). Combine with `AND`.

## Read an entry

```
https://rest.uniprot.org/uniprotkb/Q15116?fields=accession,protein_name,length,sequence,ft_domain,ft_signal,ft_chain,ft_topo_dom,ft_transmem,ft_carbohyd,xref_pdb&format=json
```

Field names: `sequence`, `ft_domain` (domains with positions), `ft_signal`,
`ft_chain` (mature chain), `ft_topo_dom` (the regions outside the membrane:
extracellular / cytoplasmic), `ft_transmem` (the membrane-spanning
segments), `ft_carbohyd` (glycosylation sites), `ft_disulfid`, `xref_pdb`
(PDB entries with chains and ranges), `cc_function`. Positions are 1-based on
the full precursor sequence, signal peptide included.

FASTA only:

```
https://rest.uniprot.org/uniprotkb/Q15116.fasta
```

## Cutting an extracellular domain

Read `ft_signal` and `ft_topo_dom` from the entry, then slice the sequence
from `ft_topo_dom` (Extracellular) start to end; do not guess boundaries from
a PDB construct, which may carry tags or truncations.
