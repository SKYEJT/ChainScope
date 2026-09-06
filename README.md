# ChainScope

**GLM-5.1 powered autonomous on-chain investigation agent** — give it an Ethereum address, and it autonomously plans, calls tools, follows the money across hops, self-corrects its hypotheses, reaches an evidence-backed risk verdict, and publishes a verifiable attestation on-chain.

> Z.AI Track · Web3 × Long-Horizon Task · AI × Web3 Agentic Builders Hackathon

---

## Problem

Ethereum has hundreds of millions of addresses. Deciding whether one is involved in **money laundering, mixing, or fraud** today means manually tracing funds hop by hop and checking blacklists — slow, error-prone, lagging behind new entities, and the conclusions are hard for a third party to verify.

## Solution

ChainScope hands the **entire investigation loop** to an autonomous agent driven by GLM-5.1:

- **Long-horizon autonomy** — a LangGraph state machine (`PLAN → ACT → OBSERVE → REFLECT → REPLAN`), not a fixed pipeline. GLM decides every action at runtime: which tool to call, which counterparty to chase, when to conclude.
- **Follow the money** — multi-hop fund-flow tracing that surfaces layering, funneling and circular flows.
- **Interpretable ML signal** — GB-TGAD (self-developed Grain-Ball Temporal Graph Anomaly Detector) gives a 4-component anomaly score (attribute / structural / flow / temporal). The ML score is an **advisory signal only**; the agent corroborates or overrides it with behavioural evidence.
- **Blind-label verification** — any known label matched during the run is **sealed** away from the LLM, forcing a behaviour-only verdict; the label is revealed and reconciled at report time (AGREE / CONFLICT). No label leakage, no answer shortcuts.
- **Self-correction** — explicit hypotheses with confidence, revised (confirmed / refuted) as evidence arrives; risk estimate updated continuously; a full CaseFile records every step.
- **Verifiable conclusion** — the report goes to IPFS and an EAS attestation is written on Sepolia. Anyone can verify.

**Target users**: exchange / wallet risk-control teams (counterparty due diligence), on-chain compliance analysts (SAR-style report assistance), and end users checking an address before a large transfer.

## Architecture

```
LangGraph loop:  PLAN → ACT → OBSERVE → REFLECT → REPLAN → ... → REPORT
                 (GLM plans, picks tools, observes results, self-corrects)

Toolbox (17 tools):
  Data      : get_eth_balance / get_transactions / get_token_transfers / get_internal_transactions
  Graph/ML  : build_address_graph / detect_anomaly / explain_detection
  Tracing   : trace_fund_flow (multi-hop follow-the-money) / investigate_neighbor
  Grounding : lookup_address_label (exchange / mixer / sanctioned entity; sealed in blind mode)
  Memory    : propose_hypothesis / record_finding / revise_hypothesis
              set_risk_estimate / review_case / compile_final_report
  Attest    : publish_attestation (IPFS + EAS on Sepolia)

State        : CaseFile — persistent investigation notebook (hypotheses, evidence,
               visited addresses, fund paths, risk estimate, full iteration log)
```

**LLM call site**: `chainscope/llm.py` — the only LLM entry point. The hackathon build was locked to GLM-5.1; the model is now chosen in `.env` via `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` (any OpenAI-compatible endpoint with tool calling, e.g. `glm-5.1`, `deepseek-v4-flash`, `deepseek-v4-pro`; defaults reproduce the GLM-5.1 setup). Agent core: `chainscope/agent/graph.py`.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/SKYEJT/ChainScope.git
cd ChainScope

# 2. Create environment
conda create -n chainscope python=3.12
conda activate chainscope
pip install -e .

# 3. Configure API keys
cp .env.example .env
# Fill in: LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (e.g. DeepSeek or GLM), ETHERSCAN_API_KEY (optional fallback),
#          WALLET_PRIVATE_KEY (dedicated Sepolia testnet wallet, optional — without it
#          attestation falls back to local storage and the investigation still completes)

# 4. Run the autonomous agent (CLI)
python -m chainscope.agent 0x<address>

# 5. Or the live UI — Streamlit runs locally at http://localhost:8501
streamlit run app.py
```

> Reproducibility notes: the UI is a local Streamlit app (no public deployment needed).
> Pretrained weights `data/best_model.pt` ship with the repo, so detection works out of the box.

## On-chain Evidence (real attestations from the demo runs)

| Case | Target | Blind verdict | Sepolia TX | EAS attestation | Report (repo copy) |
|------|--------|---------------|------------|-----------------|--------------------|
| A · Tornado.Cash 100 ETH (mixer) | `0xa160cdab…3f291` | **0.82 HIGH** — label reconciliation **AGREE** | [`0x8973d593…b91000`](https://sepolia.etherscan.io/tx/0x8973d593965d39dce560a20c8fd5a047a675c783737b7e80ae930ef0a4b91000) | [`d99d7a6f…da2443`](https://sepolia.easscan.org/attestation/view/d99d7a6fd35c875435112c9621854d9f4b5d939a67c6dfdab810d301d7da2443) | `data/local_reports/bafybei58d53379….json` |
| B · vitalik.eth | `0xd8dA6BF2…96045` | **0.05 LOW** — label reconciliation **AGREE** | [`0x73a61783…bc224b`](https://sepolia.etherscan.io/tx/0x73a6178386b57ee13646da78d9ab84120adfc6fd5c4df8a52157cb7942bc224b) | [`21a38541…6130b`](https://sepolia.easscan.org/attestation/view/21a38541698952e094e3556db6da5dd0fbdcfe45e81f9a6286a68578d816130b) | `data/local_reports/bafybeid24cdbf8….json` |

Both runs were fully blind: the agent never saw a label until the final report. In case A the ML total score (0.57) was *below* the 0.6 threshold — a pure-ML system would have missed it; the agent overrode the signal after tracing a 1,200+ ETH layering chain.

> CID note: report upload uses web3.storage when `IPFS_TOKEN` is set; otherwise a deterministic
> content-hash CID is generated and the report is stored locally. The two attested report files
> are committed in `data/local_reports/` so the attestations remain verifiable.

## Long-Horizon Run Records

- `data/cases/*.json` — full CaseFile per investigation: plan, every tool call, hypotheses with revision history, evidence, fund paths, risk timeline, label reconciliation.
- `data/run_records/demo_run_log_20260610.txt` — complete iteration log of the two demo runs above (10 steps / 41 tool calls and 18 steps respectively).
- The Streamlit UI can also export a standalone HTML run record per investigation.

## APIs / SDKs / AI Tools

- **AI tool**: **GLM-5.1 (Z.AI)** — used for *all* agent reasoning and inference end-to-end (planning, tool selection, reflection, verdicts). Enforced at startup; no other model is accepted.
- **Third-party APIs**: Blockscout (primary on-chain data, no key), Etherscan V2 (fallback), Ankr public RPC (balances), Alchemy (optional Sepolia RPC), web3.storage (optional IPFS upload), EAS — Ethereum Attestation Service contract on Sepolia.
- **Open-source SDKs**: LangChain + LangGraph, PyTorch + PyTorch Geometric, NetworkX, web3.py, Streamlit, plotly / pyvis / matplotlib.

## Status & Roadmap

**Completed (hackathon scope)**

- Autonomous LangGraph agent with 17 tools, reflection, nudging, forced wrap-up
- GB-TGAD detection adapted from Elliptic (165-dim) to Ethereum (38-dim) with pretrained weights
- Blind-label sealing + post-verdict reconciliation
- IPFS + EAS attestation pipeline (two real Sepolia attestations)
- Streamlit live-streaming UI, CaseFile run records, test suite

**Next**

- Retrain GB-TGAD natively on Ethereum temporal snapshots (close the Bitcoin→Ethereum domain gap; the model is actively iterating)
- Batch / monitoring mode (watch a set of addresses continuously)
- Richer label sources and multi-chain support

## Security & Boundaries

ChainScope only **reads** main-net data and only **writes** attestations to **Sepolia testnet**. The signing key is a dedicated testnet wallet in a gitignored `.env`. All verdicts are **advisory** and require human review. Full details: [`docs/SECURITY.md`](docs/SECURITY.md).

## Team

| | |
|---|---|
| Builder | **SKYEJT** — solo (graduate researcher in Graph Neural Networks / graph anomaly detection) |
| Contact | celtics7@163.com |
| Attester wallet (Sepolia testnet only) | `0xBed54f61DBa40769EF9Ceef8622ad24d22EE3eA7` |
