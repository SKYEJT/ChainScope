# ChainScope Security Boundary

## Permissions
- **Read-only main-net access**: all on-chain data comes from public read APIs — Blockscout (primary, no key), Etherscan V2 (fallback), Ankr public RPC (balances). No main-net transaction is ever built or sent.
- **Write-only testnet**: the only write operation is an EAS attestation on **Sepolia** (chainId 11155111).

## Private Key & Wallet Boundary
- The attester wallet is a **dedicated Sepolia testnet account**: `0xBed54f61DBa40769EF9Ceef8622ad24d22EE3eA7` — never a personal or main-net wallet.
- `WALLET_PRIVATE_KEY` lives only in the local `.env` (gitignored); it is never logged, printed, committed, or sent to any API.
- It signs **only** EAS attestation transactions on Sepolia; `attest_publisher.py` builds no other transaction type.
- Worst-case blast radius is limited to test ETH on Sepolia — no real funds are ever at risk.
- If `WALLET_PRIVATE_KEY` is unset, on-chain attestation is disabled and reports fall back to local storage; the investigation still completes.

## Report Storage (IPFS)
- With `IPFS_TOKEN` set, reports are uploaded to web3.storage and the returned CID is attested.
- Without it, a **deterministic content-hash CID** is generated and the report is stored under `data/local_reports/`. The two attested demo reports are committed to the repo so their attestations remain verifiable.

## Cost Control
- Blockscout: free, no API key (primary data source)
- Etherscan free tier: 5 calls/sec, 100K/day (fallback only)
- Ankr public RPC: free (balance / block queries)
- Sepolia gas: free test ETH from faucets
- GLM API: GLM-5.1 (paid Z.AI API; enforced by `config.require_glm_5_1` — no other model is accepted)
- Built-in rate limiting, on-disk caching and request retry keep API usage well inside free tiers.

## Failure Handling
- API timeout: agent retries once, then skips and continues
- Data source failure: Blockscout automatically falls back to Etherscan
- Empty results: agent notes "no data available" and proceeds
- Model error: agent reports "detection unavailable" and continues
- Attestation failure: report saved locally, on-chain publication deferred

## Human Intervention
- The agent does **not** auto-report to any authority
- All findings are advisory and require human review
- Anomaly scores are heuristic signals, not legal evidence

## Data Privacy
- No personal data stored beyond public on-chain records
- IPFS reports contain only the address + anomaly metadata
- Local run logs are stored in `data/agent_logs/` (gitignored)
