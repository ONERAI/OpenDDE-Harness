---
name: pubmed-lookup
description: |
  Find and read biomedical literature through NCBI E-utilities and Europe PMC (no key):
  search PubMed for papers on a target, antibody or method, read titles, journals,
  years and abstracts, and fetch open-access full text. Use this instead of web_search
  for any question about published structures, epitopes, affinities or protocols.
homepage: https://www.ncbi.nlm.nih.gov/books/NBK25501/
---

# PubMed lookup

Fixed, keyless endpoints. Fetch them with `web_fetch`. NCBI allows about
three requests per second without a key; a search plus a summary is enough
for most questions.

## Search

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=PD-1+nanobody+crystal+structure&retmode=json&retmax=10&sort=relevance
```

Returns `esearchresult.idlist` (PMIDs) and `count`. Field tags narrow a term:
`PD-1[Title]`, `Zak KM[Author]`, `2020:2026[dp]` (date published),
`"Nature Communications"[Journal]`.

## Titles, journals, years

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=26602187,28280600&retmode=json
```

Read `result[<pmid>].title`, `.source` (journal), `.pubdate`, `.authors[].name`,
`.elocationid` (DOI).

## Abstract

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=26602187&rettype=abstract&retmode=text
```

Plain text, several PMIDs at once with commas.

## Open-access full text (Europe PMC)

```
https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PD-1+nanobody+structure+AND+OPEN_ACCESS:y&format=json&pageSize=10&resultType=lite
```

`OPEN_ACCESS:y` keeps the search to articles whose full text may be
fetched; a PMCID alone does not mean that. For each result with
`isOpenAccess` = `Y`, the full text is
`https://www.ebi.ac.uk/europepmc/webservices/rest/<pmcid>/fullTextXML`
with the `pmcid` value verbatim (it already carries the `PMC` prefix). For
anything else, the PubMed abstract above is what is available.
