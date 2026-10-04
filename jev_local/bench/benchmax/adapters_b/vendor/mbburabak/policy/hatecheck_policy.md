# HateCheck labeling policy

Sources: paper https://arxiv.org/abs/2012.15606 (ACL 2021) · card https://huggingface.co/datasets/Paul/hatecheck · Retrieved 2026-09-21.

## Task

Binary hate-speech detection on single English documents. Gold label `hateful` vs `non-hateful` (`label_gold`). Hate speech is abuse targeted at a protected group — age, disability, gender identity, familial status, pregnancy, race, national or ethnic origin, religion, sex or sexual orientation — or at a member of such a group for being part of it (paper §1; UK Equality Act 2010 / US Civil Rights Act 1964 / EU Charter). Seven protected groups: women, trans people, gay people, black people, disabled people, Muslims, immigrants.

## Counts as hateful (true)

- Expression of strong negative emotions about a group ("I hate [GROUP]"; "[GROUP] make me sick").
- Description of a group using very negative attributes; dehumanisation ("[GROUP] are scum"; caring less about a group than about dirt); implicit derogation ("[GROUP] belong in a zoo").
- Direct threats against a group or members; threats as normative statements; proposals to subject a group to harmful treatment ("we should test new drugs on [GROUP]").
- Hate expressed via slurs, profanity, pronoun reference in later clauses/sentences, negated positive statements ("[GROUP] lives do not matter"), questions, opinions, or spelling variations. Obfuscated renderings count as the word they render — digits for letters (leet), spaces between letters, dropped, doubled, or fused letters — and derogatory use of such renderings toward a protected group is hateful.
- Questions and hypotheticals that presuppose the hateful claim ("Where is the mistake in saying [GROUP] deserve to die?") assert the hate; the question form does not soften it.
- Hate directed at an individual *as a member* of a protected group.

## Does NOT count as hateful (false)

- Criticism of ideas, institutions, or practices (including religions as ideas) that does not demean adherents as such.
- Counter-speech that quotes hate to condemn it, or denounces hate by direct reference ("Statements like '[GROUP] are scum' are deeply hurtful") — including quotes attached to condemnation ("he should be in jail for wishing 'death to all [GROUP]'", "she should be kicked off the platform for saying '[GROUP] should all be killed'") and counter-violence directed at hate perpetrators ("I will cut their racist throats").
- Reclaimed slurs used non-hatefully in-group and non-hateful slur homonyms ("I had this queer feeling").
- Non-hateful profanity not aimed at a protected group.
- Negated hate ("No [GROUP] deserves to die"; "We shouldn't let [GROUP] suffer").
- Neutral or positive statements using group identifiers.
- Abuse targeted at objects, at individuals not as group members, or at non-protected groups (professions, capitalists).

Jev reads criteria literally: judge the target and effect, not the presence of slurs or profanity alone.
