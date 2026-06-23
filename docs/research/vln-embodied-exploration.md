# VLN & Embodied AI per la coverage 3DGS di gaussian-bot

- Stato: **research note** (base di lavoro, non ancora una decisione — vedi ADR per le decisioni)
- Data: 2026-06-23
- Scope: come sfruttare idee da Visual Language Navigation (VLN) ed embodied AI per
  migliorare l'esplorazione autonoma e la generazione di pose per la ricostruzione 3DGS.
- Vincolo trasversale: deve girare nel nostro stack — VLM **Qwen via vLLM**, renderer
  **gsplat**, **single-GPU**, **senza training pesante**. Preferenza forte per
  approcci **training-free / prompting / inference-time**.

> Metodo: report prodotto da una deep-research multi-agente (5 angolazioni, 23 fonti
> primarie — NeurIPS/CVPR/ECCV/RSS/IROS/ACL — 109 claim estratti, 25 verificati in modo
> avversariale: 24 confermati 3-0, 1 confutato). Le fonti sono in fondo al documento.

---

## TL;DR — le 3 leve principali

1. **NBV guidato dall'incertezza di ricostruzione, calcolato sullo splat stesso**
   (FisherRF, GauSS-MI). È la cosa più *on-target* in assoluto: il nostro obiettivo
   *è* next-best-view per ricostruzione. Sostituisce le euristiche novelty/coverage
   con un segnale principiato, real-time, **training-free**.
   → tocca `session.py` (seeding), `metrics/coverage.py`, `filters/pose_filters.py`,
   e il prompt in `nav/observation.py`.
2. **Memoria topologica + prompting in linguaggio naturale invece di coordinate**
   (MapGPT). Insight verificato: gli LLM ragionano *male* su coordinate grezze e *bene*
   su relazioni topologiche tra nodi. Più planning multi-step con backtracking →
   attacca direttamente l'**orbiting/loop**. Fully training-free.
   → `nav/explorer.py`, `nav/observation.py`.
3. **Frontier-based exploration con information-gain** (StructNav, PONI) al posto del
   seeding farthest-point. Training-free e batte baseline RL apprese su ObjectNav.
   → `session.py` (`generate_seeds`), `metrics/coverage.py`.

Quasi tutto rispetta il vincolo single-GPU / no-training / inference-time. I pochi
componenti appresi (potential network di PONI, RL-NBV) hanno sostituti training-free
espliciti (StructNav, FisherRF/GauSS-MI).

**Percorso consigliato**: partire da **P1** (memoria topologica + prompting, economico
e training-free, attacca subito l'orbiting); in parallelo spike su **P0** (FisherRF sul
gsplat) perché è l'unica cosa veramente *on-objective* e abilita anche
l'instruction-following futuro ("zone sfocate"). P2/P3 vengono quasi gratis una volta
che hai la mappa di incertezza e il grafo.

---

## Parte 1 — Survey strutturata del campo

### 1.1 VLN classico (instruction-following)
Il VLN è il paradigma per agenti che **comunicano in linguaggio naturale, percepiscono
l'ambiente ed eseguono task** ([VLN survey, ACL 2022][vln-survey]). Benchmark: R2R/RxR
(segui un'istruzione passo-passo lungo un grafo di viewpoint), REVERIE (istruzioni
"remote-object", goal di alto livello).
**Posizionamento**: il nostro task *non* è questo — facciamo coverage autonoma, non
seguiamo istruzioni. Ma le *tecniche* (memoria, grounding visione-linguaggio, waypoint
prediction) sono riusabili, ed è la base per l'instruction-following futuro (§3).

### 1.2 Embodied exploration: da frontier a semantic
- **Frontier-Based Exploration**: vai verso il confine tra noto e ignoto. Geometrico,
  classico, robusto.
- **Semantic exploration**: inietta priori semantici nella scelta del frontier.
  **SemExp** (RL appreso). **PONI** ([CVPR 2022][poni]) disaccoppia *"where to look?"*
  (rete appresa di potenziali) da *"how to navigate?"* (Fast Marching analitico) — solo
  il "where" è appreso.
- **StructNav** ([RSS 2023][structnav]): pipeline **completamente modulare e
  training-free** che inietta semantica (da BERT/CLIP pre-addestrati, zero-shot) nel
  frontier-based per scegliere il frontier più promettente. **Senza training sulle scene
  del benchmark batte SemExp del +6.6% SR e +22.7% SPL su Gibson** ⇒ il modulare
  training-free può superare l'RL appreso.

### 1.3 Active vision / Next-Best-View per ricostruzione — *la categoria più vicina a noi*
La [review NBV 2025][nbv-review] tassonomizza in 5 famiglie: **rule-based,
uncertainty-based, sampling-based, learning-based, prediction-based**. Per il nostro
vincolo (single-GPU, no-training) le famiglie giuste sono **uncertainty-based** e
**rule/sampling-based**; l'RL-NBV (learning-based) è il trade-off "training pesante" da
evitare.

Metodi definiti **direttamente sul radiance field / splat** (perfetti per noi):
- **FisherRF** ([ECCV 2024][fisherrf]): usa la **Fisher Information** per quantificare
  l'informazione osservata in un radiance field **senza ground-truth**. Model-agnostic
  (NeRF *e* 3DGS), dà incertezza pixel-wise, e seleziona la NBV a **70 fps su backend
  3DGS** (Hessiana diagonale in CUDA, ~11 ms), **nessuna policy addestrata**.
- **GauSS-MI** ([RSS 2025][gaussmi]): **Shannon Mutual Information** come criterio NBV
  real-time definito sul modello 3DGS, con un modello di incertezza probabilistico
  **per-Gaussiana** (incertezza tracciata a livello di primitiva, non con una rete
  esterna).
- **ActiveSplat** ([fonte][activesplat]): active mapping costruito *su* Gaussian
  Splatting — riferimento architetturale diretto per esplorazione attiva su splat.

### 1.4 Navigazione zero-shot con LLM/VLM (training-free)
- **MapGPT** ([ACL 2024][mapgpt]): agente VLN **zero-shot** con LLM general-purpose,
  **nessun fine-tuning**. Due idee chiave verificate: (a) tiene **solo le relazioni
  topologiche** dei nodi mappa perché *"è difficile per GPT capire dati di coordinate
  precise"*, codificando la connettività con template in linguaggio naturale;
  (b) **adaptive multi-step path planning** — combina pensiero, mappa e piano precedente
  per aggiornare un piano multi-step, abilitando esplorazione sistematica e
  **backtracking** (49% di backtracking con 80% di correzioni riuscite su REVERIE).
- **WMNav** ([IROS 2025][wmnav]): framework **training-free** che integra il VLM in un
  *World Model*. Mantiene online una **Curiosity Value Map**: ogni cella ha un valore
  0–10 (quanto vale esplorare quella zona), aggiornato fondendo predizioni del VLM con
  osservazioni passate ⇒ trasforma il VLM nell'euristica di esplorazione, con memoria
  spaziale persistente.

### 1.5 Spatial memory / mappe topologiche con i VLM
- **HAMT** ([NeurIPS 2021][hamt]): codifica gerarchica (viste singole → panorami →
  panorami storici) per usare **l'intera storia** di navigazione; risolve il fallimento
  della memoria ricorrente semplice nel trattenere feature di inizio-traiettoria su
  percorsi lunghi.
- **Structured Scene Memory** (CVPR 2021) e rappresentazioni a grafo: l'agente
  *"mantiene memoria delle aree esplorate ed eccelle nelle decisioni globali"*
  ([survey][mem-survey]).
- **Principio modulare** (tesi [Chaplot, CMU 2021][chaplot]): separare percezione /
  mappatura / planning, unendo planning classico su mappa esplicita + priori semantici
  da foundation model. ⚠️ *Confutato* (voto 0-3) il claim più forte "il modulare è SOTA":
  adottarlo come **buon principio di design, non come garanzia di performance**.

---

## Parte 2 — Raccomandazioni concrete, prioritizzate, mappate sul codice

### 🥇 P0 — NBV su incertezza di ricostruzione (FisherRF / GauSS-MI)
**Perché**: è letteralmente il nostro obiettivo. Oggi usiamo euristiche proxy (novelty
farthest-point in `metrics/coverage.py`, `med_depth × alpha` in `_select_seeds` /
`_best_origin`). Un segnale di incertezza calcolato *sullo splat* è principiato e
real-time.

**Cosa cambia**:
- Calcola incertezza (Fisher Information o MI per-Gaussiana) sullo splat corrente.
- **Seeding** (`generate_seeds` in `session.py`): scegli i seed verso zone ad alta
  incertezza invece che solo per spread.
- **Pannello prompt** (`nav/observation.py`): aggiungi una mappa "dove la ricostruzione
  è debole" — diventa il segnale visivo che guida il VLM.
- **Filtraggio finale** (`filters/pose_filters.py`): pesa la selezione delle pose per
  information-gain, non solo novelty/budget.

**Training-free**: ✅ (FisherRF non addestra nulla; GauSS-MI traccia incertezza
per-primitiva).
**Costo**: integrazione non banale — serve calcolare FI/MI sul backend gsplat nel loop
single-GPU (non è solo prompting). 70 fps riportati senza specificare la GPU → aspettarsi
variabilità. *Massimo valore, ma massimo costo ingegneristico.*

### 🥈 P1 — Memoria topologica + prompting MapGPT-style (anti-orbiting)
**Perché**: oggi diamo al VLM una sliding-window di azioni + mappa top-down ad-hoc.
MapGPT mostra che (a) **le coordinate confondono l'LLM**, (b) servono **piani multi-step
con backtracking**. Attacca direttamente l'orbiting che già combattiamo con `StuckGuard`
/ `CoveragePlateau`.

**Cosa cambia**:
- Costruisci un **grafo topologico** dei nodi visitati (seed-walk → nodi + osservazioni
  + adiacenze).
- Nel prompt (`nav/observation.py`) passa il grafo come **adiacenza in linguaggio
  naturale** ("dal nodo A sei connesso a B (coperto), C (poco coperto)…"), **non**
  coordinate grezze.
- Cambia il controller (`nav/explorer.py`) da "una azione locale" a "**piano multi-step
  adattivo**" con possibilità di **tornare** a nodi poco coperti.

**Training-free**: ✅ fully inference-time, gira su Qwen via vLLM.
**Costo**: medio (refactor del loop + formato prompt). ⚠️ I risultati MapGPT/WMNav sono
su GPT-4V via API — la *trasferibilità architetturale* è solida, la *performance su Qwen
self-hosted* è da verificare (vedi caveat).

### 🥉 P2 — Frontier-based / information-gain seeding (StructNav)
**Perché**: il seeding farthest-point garantisce spread ma è "cieco" all'informazione.
Frontier + utility lo rende mirato, ed è training-free che batte RL.

**Cosa cambia** (`generate_seeds`, `metrics/coverage.py`):
- Rileva **frontier** sulla mappa di coverage/occupancy (confine coperto/non-coperto).
- Scegli i seed con uno score di **utilità/information-gain** (idealmente l'incertezza
  P0; in alternativa coverage-gain geometrico).
- Mantieni le pose di cattura COLMAP come *prior di validità geometrica* (già fatto), ma
  seleziona *tra* di esse per frontier-utility.

**Training-free**: ✅ (StructNav usa embedding pre-addestrati zero-shot).

### P3 — Curiosity Value Map come memoria spaziale persistente (WMNav)
**Cosa**: una mappa-griglia dove ogni cella tiene un punteggio 0–10 "quanto vale
esplorare qui", **predetto dal VLM** e aggiornato fondendo predizioni + osservazioni.
Memoria persistente che guida la prossima azione/seed. Si combina naturalmente con la
mappa di incertezza P0 (curiosity = incertezza alta). **Training-free**, inference-time.

### P4 — Stop policies basate su information-gain
**Cosa**: oggi `CoverageTarget` / `QualityTarget` / `CoveragePlateau` (in `nav/stop.py`)
usano novelty/coverage. Aggiungi/sostituisci con una soglia su **information-gain
residuo** (FI/MI marginale dell'ultima vista): fermati quando l'incertezza non cala più.
Criterio di stop più principiato e allineato all'obiettivo di ricostruzione.

---

## Parte 3 — Instruction-following (obiettivo futuro)

La buona notizia: **la stessa macchinaria di P1–P3 si ri-punta su goal in linguaggio
naturale senza fine-tuning**. Essendo MapGPT e WMNav zero-shot/inference-time, basta
aggiungere il goal NL al prompt:
- *"Esplora la cucina"* → grounding visione-linguaggio sul grafo topologico (tecniche VLN
  classiche + embedding CLIP per matchare la descrizione delle stanze ai nodi).
- *"Concentrati sulle zone poco ricostruite/sfocate"* → **si ground-a direttamente sulla
  mappa di incertezza FisherRF/GauSS-MI di P0** (il "blurry" è esattamente alta
  incertezza). Punto più elegante: il segnale costruito per la coverage autonoma *è già*
  il target di questa istruzione.

Pattern riusabili: paradigma VLN (R2R/RxR/REVERIE) per la struttura, MapGPT per il
prompting topologico+planning, WMNav per il world-model con curiosity map. **Nessuna
policy da addestrare.**

---

## Caveat e rischi (dalla verifica avversariale)

1. **Qwen ≠ GPT-4V**: i risultati headline di MapGPT/WMNav sono su GPT-4/GPT-4V via API.
   Il trasferimento *architetturale* è solido, ma la performance del ragionamento
   spaziale/topologico su Qwen self-hosted **è non verificata** e potrebbe degradare →
   potrebbero servire few-shot exemplar o structured-output scaffolding.
2. **Task mismatch**: StructNav/PONI/WMNav sono ObjectGoal-nav su benchmark indoor
   (Gibson/HM3D/MP3D), non coverage-per-ricostruzione. I *meccanismi* (frontier,
   curiosity, memoria) trasferiscono bene; i *numeri assoluti* di successo no.
3. **Costo P0**: FisherRF/GauSS-MI sono il fit più diretto ma l'integrazione sul loop
   gsplat single-GPU ha costo reale; il 70 fps è riportato senza nominare la GPU.
4. **Claim confutato**: "il modulare è SOTA" è stato ucciso (0-3). Modularità = principio
   di design valido, non garanzia di performance.
5. Diversi metodi sono recentissimi (2025) e ancora in evoluzione.

---

## Domande aperte / prossimi esperimenti

1. Qwen self-hosted regge il ragionamento topologico MapGPT-style e il planning
   multi-step? (test rapido: prompt con grafo NL + few-shot, misura tasso di backtracking
   corretto).
2. Costo/latenza reale di FisherRF FI o GauSS-MI per-Gaussiana sul nostro gsplat in
   single-GPU, e si può **riassumere** la mappa di incertezza nel prompt senza saturare
   il context?
3. L'incertezza P0 **sostituisce** o **integra** le euristiche novelty+coverage attuali,
   per seeding *e* filtraggio finale? Quale combinazione dà il set di pose meglio
   distribuito per la densificazione 3DGS?
4. Per l'instruction-following: goal tipo "zone sfocate" → grounding diretto-su-prompt
   sulla mappa incertezza, oppure funzione goal→frontier-scoring?

---

## Mappatura sintetica idea → file

| Leva | File principali | Training-free | Costo |
|------|-----------------|:-------------:|:-----:|
| P0 NBV incertezza (FisherRF/GauSS-MI) | `session.py`, `metrics/coverage.py`, `filters/pose_filters.py`, `nav/observation.py` | ✅ | Alto |
| P1 Memoria topologica + prompting | `nav/explorer.py`, `nav/observation.py` | ✅ | Medio |
| P2 Frontier/information-gain seeding | `session.py` (`generate_seeds`), `metrics/coverage.py` | ✅ | Medio |
| P3 Curiosity Value Map | `nav/observation.py`, `nav/explorer.py` | ✅ | Medio |
| P4 Stop su information-gain | `nav/stop.py` | ✅ | Basso |

---

## Fonti principali (tutte primarie, voto 3-0 salvo dove indicato)

[fisherrf]: https://arxiv.org/abs/2311.17874v1
[gaussmi]: https://arxiv.org/pdf/2504.21067
[activesplat]: https://li-yuetao.github.io/ActiveSplat/ActiveSplat.pdf
[nbv-review]: https://www.sciencedirect.com/science/article/pii/S2667393225000171
[mapgpt]: https://arxiv.org/html/2401.07314v2
[wmnav]: https://arxiv.org/html/2503.02247v1
[structnav]: https://www.roboticsproceedings.org/rss19/p075.pdf
[poni]: https://www.semanticscholar.org/paper/PONI:-Potential-Functions-for-ObjectGoal-Navigation-Ramakrishnan-Chaplot/0c6af0a9da38e4af39f54d5a1455a76e38f008c9
[hamt]: https://arxiv.org/abs/2110.13309
[mem-survey]: https://arxiv.org/html/2402.14304v1
[vln-survey]: https://arxiv.org/pdf/2203.12667
[chaplot]: https://arxiv.org/pdf/2106.13415

- **FisherRF** — Fisher Information NBV su NeRF/3DGS, 70 fps, ECCV 2024: <https://arxiv.org/abs/2311.17874v1>
- **GauSS-MI** — Shannon MI per-Gaussiana, RSS 2025: <https://arxiv.org/pdf/2504.21067>
- **ActiveSplat** — active mapping su Gaussian Splatting: <https://li-yuetao.github.io/ActiveSplat/ActiveSplat.pdf>
- **Review NBV** (tassonomia 5 famiglie), 2025: <https://www.sciencedirect.com/science/article/pii/S2667393225000171>
- **MapGPT** — VLN zero-shot, mappa topologica NL + multi-step planning, ACL 2024: <https://arxiv.org/html/2401.07314v2>
- **WMNav** — VLM world-model + Curiosity Value Map, IROS 2025: <https://arxiv.org/html/2503.02247v1>
- **StructNav** — frontier semantico training-free, RSS 2023: <https://www.roboticsproceedings.org/rss19/p075.pdf>
- **PONI** — potential functions, CVPR 2022: <https://www.semanticscholar.org/paper/PONI:-Potential-Functions-for-ObjectGoal-Navigation-Ramakrishnan-Chaplot/0c6af0a9da38e4af39f54d5a1455a76e38f008c9>
- **HAMT** — history-aware multimodal transformer, NeurIPS 2021: <https://arxiv.org/abs/2110.13309>
- **Survey memoria spaziale/grafi VLN**: <https://arxiv.org/html/2402.14304v1>
- **VLN survey**, ACL 2022: <https://arxiv.org/pdf/2203.12667>
- **Chaplot thesis** (modular navigation), CMU 2021: <https://arxiv.org/pdf/2106.13415>
